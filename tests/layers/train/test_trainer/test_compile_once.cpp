/**************************************************************************/
/* test_compile_once.cpp                                                  */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/


/**
 * @file test_compile_once.cpp
 * @brief Tests that a training step compiles once and that every replay of it dispatches work.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;

namespace
{
    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 8;
    constexpr int64_t K_FEATURES = 6;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 4;

    /** @brief The value a result buffer holds before a run, which no run writes. */
    constexpr double K_POISON = -12345.0;

    /** @brief Provides the engine, and the classifier every training case compiles on it. */
    class Medium_CompileOnce : public SaTest::EngineTest
    {
        protected:
            /** @brief Traces, binds and initialises the classifier; sets @c ready on success. */
            void prepare()
            {
                SA_ASSERT_OK_AND_ASSIGN(
                    built, Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));
                SA_ASSERT_OK_AND_ASSIGN(state, NN::ModelState::create(*engine, model));
                pool = std::make_unique<TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(slots,
                                        Train::bind_parameters(backend(), built, *state, *pool));
                SA_EXPECT_OK(NN::initialize(backend(), *state));
                ready = true;
            }

            /**
             * @brief Fits @p dataset once and expects the compile count to stay at one.
             * @param steps Receives the steps the fit ran, added to its value.
             */
            void fit_once(Train::Trainer& trainer, const Data::Dataset& dataset,
                          std::size_t& steps)
            {
                SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer.fit(dataset));
                steps += report.steps;

                EXPECT_EQ(report.compile_count, 1u);
                EXPECT_EQ(trainer.compile_count(), 1u);
            }

            /**
             * @brief Runs one step over batch @p index of @p dataset.
             * @return Whether the first parameter read differently after the step than before.
             */
            bool step_moves_the_parameter(Train::Trainer& trainer, const Data::Dataset& dataset,
                                          std::size_t index)
            {
                const std::vector<double> before = SaTest::read_tensor(slots[0].parameter);
                Data::Batch batch;
                if (!dataset.read(index % static_cast<std::size_t>(K_BATCHES), batch).is_ok())
                    return false;

                if (!trainer.run_step(batch).has_value())
                    return false;

                return SaTest::read_tensor(slots[0].parameter) != before;
            }

            /**
             * @brief Fits @p dataset, halves the learning rate and runs two single steps over
             *        each of @p first and @p second, expecting no recompile on the way.
             */
            void round_of(Train::Trainer& trainer, Optim::AdamW& optimizer,
                          const Data::Dataset& dataset, const Data::Dataset& first,
                          const Data::Dataset& second)
            {
                SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer.fit(dataset));
                EXPECT_EQ(report.compile_count, 1u);

                // The learning rate is a host-side scalar; see tests/layers/train/README.md.
                optimizer.options().learning_rate *= 0.5;

                // Individual steps between fits, mixing the two datasets.
                Data::Batch second_0;
                Data::Batch first_0;
                Data::Batch second_1;
                Data::Batch first_1;
                ASSERT_TRUE(SaTest::status_is_ok(second.read(0, second_0)));
                ASSERT_TRUE(SaTest::status_is_ok(first.read(0, first_0)));
                ASSERT_TRUE(SaTest::status_is_ok(second.read(1, second_1)));
                ASSERT_TRUE(SaTest::status_is_ok(first.read(1, first_1)));
                SA_EXPECT_OK(trainer.run_step(second_0));
                SA_EXPECT_OK(trainer.run_step(first_0));
                SA_EXPECT_OK(trainer.run_step(second_1));
                SA_EXPECT_OK(trainer.run_step(first_1));

                EXPECT_EQ(trainer.compile_count(), 1u);
            }

            /**
             * @brief Writes @p input into @p x, poisons @p y, replays @p step and reads @p y.
             * @return The elements of @p y after the replay.
             */
            std::vector<double> replay_over(Graph::CompiledStep& step, TensorPool& tensors,
                                            Graph::ValueID x, Graph::ValueID y,
                                            const std::vector<double>& input)
            {
                SaTest::write_tensor(tensors.tensor(x), input);
                SaTest::write_tensor(tensors.tensor(y),
                                     std::vector<double>(input.size(), K_POISON));
                SA_EXPECT_OK(step.run());

                return SaTest::read_tensor(tensors.tensor(y));
            }

            MLP model{NN::Linear{K_FEATURES, 12}, NN::ReLU{}, NN::Linear{12, K_CLASSES}};
            Train::TrainingGraph built;                /**< @brief Holds its training graph. */
            std::optional<NN::ModelState> state;       /**< @brief Holds its parameters. */
            std::unique_ptr<TensorPool> pool;          /**< @brief Holds the step's values. */
            std::vector<Optim::ParameterSlot> slots;   /**< @brief Pairs parameters and gradients. */
            bool ready = false;                        /**< @brief Holds whether prepare() ended. */
    };

    Data::SyntheticOptions make_data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = K_BATCH;
        options.batches = K_BATCHES;
        options.features = K_FEATURES;
        options.classes = K_CLASSES;
        return options;
    }
}

/**
 * @brief Tests that repeated fit() calls leave the trainer's compile count at one.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_CompileOnce, ManyFitCallsCompileTheStepExactlyOnce)
{
    prepare();
    ASSERT_TRUE(ready);
    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    SA_ASSERT_OK_AND_ASSIGN(auto optimizer, Optim::SGD::make(backend(), slots, sgd));
    Train::TrainerOptions options;
    options.epochs = 1;
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(auto trainer,
                            Train::Trainer::make(backend(), built, *pool, objective,
                                                 *optimizer, options));
    SA_ASSERT_OK_AND_ASSIGN(auto dataset,
                            Data::SyntheticClassification::make(backend(), make_data_options()));
    ASSERT_EQ(trainer->compile_count(), 1u);

    std::size_t steps = 0;
    fit_once(*trainer, *dataset, steps);
    fit_once(*trainer, *dataset, steps);
    fit_once(*trainer, *dataset, steps);
    fit_once(*trainer, *dataset, steps);
    fit_once(*trainer, *dataset, steps);
    fit_once(*trainer, *dataset, steps);

    EXPECT_EQ(steps, static_cast<std::size_t>(6 * K_BATCHES));
    EXPECT_EQ(trainer->step_count(), steps);
}

/**
 * @brief Tests that each of five replayed steps moves a parameter, so none reports complete
 *        without dispatching.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_CompileOnce, EveryReplayDispatchesRatherThanReportingCompleteImmediately)
{
    prepare();
    ASSERT_TRUE(ready);
    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    SA_ASSERT_OK_AND_ASSIGN(auto optimizer, Optim::SGD::make(backend(), slots, sgd));
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(auto trainer,
                            Train::Trainer::make(backend(), built, *pool, objective,
                                                 *optimizer));
    SA_ASSERT_OK_AND_ASSIGN(auto dataset,
                            Data::SyntheticClassification::make(backend(), make_data_options()));

    EXPECT_TRUE(step_moves_the_parameter(*trainer, *dataset, 0));
    EXPECT_TRUE(step_moves_the_parameter(*trainer, *dataset, 1));
    EXPECT_TRUE(step_moves_the_parameter(*trainer, *dataset, 2));
    EXPECT_TRUE(step_moves_the_parameter(*trainer, *dataset, 3));
    EXPECT_TRUE(step_moves_the_parameter(*trainer, *dataset, 4));
    EXPECT_EQ(trainer->compile_count(), 1u);
}

/**
 * @brief Tests that the first run() and three replays overwrite a poisoned output buffer.
 *
 * The graph doubles its input, which is exact in float, so each result is compared exactly.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_CompileOnce,
       TheVeryFirstRunDispatchesRatherThanReportingCompleteOverAnEmptyBaseline)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, builder.scale(x, 2.0));
    TensorPool tensors(*engine);
    SA_EXPECT_OK(tensors.materialise_leaves(graph));
    SA_ASSERT_OK_AND_ASSIGN(auto step, Graph::CompiledStep::compile(backend(), graph, tensors));
    ASSERT_FALSE(step->empty()) << "this graph has real work in it";
    ASSERT_EQ(step->compile_count(), 1u);
    ASSERT_EQ(step->run_count(), 0u);

    // Each run must track the new input; see tests/layers/train/README.md.
    EXPECT_EQ(replay_over(*step, tensors, x, y, {1, 2, 3, 4, 5, 6}),
              (std::vector<double>{2, 4, 6, 8, 10, 12}));
    EXPECT_EQ(step->run_count(), 1u);
    EXPECT_EQ(replay_over(*step, tensors, x, y, {11, 12, 13, 14, 15, 16}),
              (std::vector<double>{22, 24, 26, 28, 30, 32}));
    EXPECT_EQ(replay_over(*step, tensors, x, y, {21, 22, 23, 24, 25, 26}),
              (std::vector<double>{42, 44, 46, 48, 50, 52}));
    EXPECT_EQ(replay_over(*step, tensors, x, y, {31, 32, 33, 34, 35, 36}),
              (std::vector<double>{62, 64, 66, 68, 70, 72}));
    EXPECT_EQ(step->run_count(), 4u);
    EXPECT_EQ(step->compile_count(), 1u);
}

/**
 * @brief Tests that swapping data, changing the learning rate and resuming fit() never recompile
 *        the step.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_CompileOnce, CompileCountSurvivesEverythingTheLoopDoesBetweenSteps)
{
    prepare();
    ASSERT_TRUE(ready);
    Optim::AdamWOptions adam;
    adam.learning_rate = 0.02;
    SA_ASSERT_OK_AND_ASSIGN(auto optimizer, Optim::AdamW::make(backend(), slots, adam));
    Train::TrainerOptions options;
    options.track_metrics = false;
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(auto trainer,
                            Train::Trainer::make(backend(), built, *pool, objective,
                                                 *optimizer, options));

    // The plan reads its inputs by address; see tests/layers/train/README.md.
    Data::SyntheticOptions other = make_data_options();
    other.seed = 0xFEED'BEEFull;
    SA_ASSERT_OK_AND_ASSIGN(auto first,
                            Data::SyntheticClassification::make(backend(), make_data_options()));
    SA_ASSERT_OK_AND_ASSIGN(auto second, Data::SyntheticClassification::make(backend(), other));
    ASSERT_EQ(trainer->compile_count(), 1u);

    // fit() returns and is entered again; see tests/layers/train/README.md.
    round_of(*trainer, *optimizer, *first, *first, *second);
    round_of(*trainer, *optimizer, *second, *first, *second);
    round_of(*trainer, *optimizer, *first, *first, *second);
    round_of(*trainer, *optimizer, *second, *first, *second);

    EXPECT_EQ(trainer->step_count(), static_cast<std::size_t>(4 * (K_BATCHES + 4)));
    EXPECT_EQ(trainer->step().run_count(), trainer->step_count());
    EXPECT_EQ(trainer->compile_count(), 1u);
}
