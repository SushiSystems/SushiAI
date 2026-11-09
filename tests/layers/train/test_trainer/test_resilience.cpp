/**************************************************************************/
/* test_resilience.cpp                                                    */
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
 * @file test_resilience.cpp
 * @brief Tests loud failures, boundary extents, numerical extremes and resource discipline.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;

namespace
{
    class Medium_Resilience : public SaTest::EngineTest {};

    /** @brief Names the suite of the cases that expect the process to stop. */
    using Medium_ResilienceDeathTest = Medium_Resilience;

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    /**
     * @class Runner
     * @brief Materialises a graph's leaves, compiles once, and replays.
     *
     * The same shape as test_forward.cpp's OneShot, kept local rather than
     * shared because the two files probe different things and a shared
     * fixture would couple their evolution.
     */
    class Runner
    {
        public:
            /**
             * @brief Builds a runner over @p graph and materialises its leaves.
             * @param backend Must outlive the runner.
             * @param graph Must outlive the runner.
             * @return The runner, or the failure of a leaf's allocation.
             */
            [[nodiscard]] static Result<std::unique_ptr<Runner>> make(
                SushiAI::ExecutionBackend& backend, const Graph::Graph& graph)
            {
                std::unique_ptr<Runner> runner(new Runner(backend, graph));
                SA_TRY(runner->pool.materialise_leaves(graph));
                return runner;
            }

            /**
             * @brief Compiles the graph.
             * @pre Every leaf has been written.
             */
            [[nodiscard]] Status compile()
            {
                SA_TRY_ASSIGN(step_, CompiledStep::compile(*backend_, *graph_, pool));
                return Status();
            }

            /**
             * @brief Replays the compiled step.
             * @pre compile() has returned success.
             */
            [[nodiscard]] Status run() { return step_->run(); }

            TensorPool pool;

        private:
            /** @brief Stores the graph and the backend and builds an empty pool. */
            Runner(SushiAI::ExecutionBackend& backend, const Graph::Graph& graph)
                : pool(backend.engine()), graph_(&graph), backend_(&backend) {}

            const Graph::Graph* graph_;
            SushiAI::ExecutionBackend* backend_;
            std::unique_ptr<CompiledStep> step_;
    };

    /**
     * @class NoBatches
     * @brief A dataset with the right shapes and nothing in it.
     *
     * SyntheticClassification refuses to generate zero batches, so an empty
     * epoch cannot be reached through it - which is exactly why the trainer's
     * own guard needs a dataset that can.
     */
    class NoBatches final : public Data::Dataset
    {
        public:
            NoBatches(int64_t batch, int64_t features, int64_t classes)
                : batch_(batch), features_(features),
                  targets_{Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{},
                                                   DType::Int32, classes}} {}

            int64_t batch_size() const noexcept override { return batch_; }
            Shape sample_shape() const override { return Shape{features_}; }
            const std::vector<Data::TargetDescription>& targets() const noexcept override
            {
                return targets_;
            }
            std::size_t batch_count() const noexcept override { return 0; }
            std::string_view name() const noexcept override { return "no-batches"; }

            [[nodiscard]] Status read(std::size_t index, Data::Batch&) const override
            {
                SA_FATAL_IF(true, "Batch {} is out of range; the dataset has 0", index);
                return Status();
            }

            [[nodiscard]] Status read_rows(std::size_t index, Data::RowRange,
                                           Data::Batch&) const override
            {
                SA_FATAL_IF(true, "Batch {} is out of range; the dataset has 0", index);
                return Status();
            }

        private:
            int64_t batch_ = 0;
            int64_t features_ = 0;
            std::vector<Data::TargetDescription> targets_;
    };

    /** @brief Every element of @p values is finite. */
    ::testing::AssertionResult all_finite(const std::vector<double>& values,
                                          std::string_view what)
    {
        const auto bad = std::find_if(values.begin(), values.end(),
                                      [](double value) { return !std::isfinite(value); });
        if (bad != values.end())
            return ::testing::AssertionFailure()
                << what << "[" << (bad - values.begin()) << "] is " << *bad
                << " - a non-finite value propagates into every weight on the next "
                   "update and the loss simply stops moving";

        return ::testing::AssertionSuccess();
    }

    /** @brief A ready-to-train MLP with its state, its pool and its parameter slots. */
    struct Fixture
    {
        MLP model;
        Train::TrainingGraph built;
        std::unique_ptr<NN::ModelState> state;
        std::unique_ptr<TensorPool> pool;
        std::vector<Optim::ParameterSlot> slots;
    };

    /** @brief Traces and binds an MLP of the given extents. */
    [[nodiscard]] Result<Fixture> make_fixture(SushiAI::ExecutionBackend& backend, int64_t batch,
                                               int64_t features, int64_t hidden, int64_t classes)
    {
        Fixture fixture{MLP(NN::Linear{features, hidden}, NN::ReLU{},
                            NN::Linear{hidden, classes}), {}, nullptr, nullptr, {}};

        SA_TRY_ASSIGN(fixture.built,
                      Train::build_classifier(fixture.model, batch, features, classes));
        SA_TRY_ASSIGN(NN::ModelState created, NN::ModelState::create(backend.engine(),
                                                                     fixture.model));
        fixture.state = std::make_unique<NN::ModelState>(std::move(created));
        fixture.pool = std::make_unique<TensorPool>(backend.engine());
        SA_TRY_ASSIGN(fixture.slots,
                      Train::bind_parameters(backend, fixture.built, *fixture.state,
                                             *fixture.pool));
        SA_TRY(NN::initialize(backend, *fixture.state));
        return fixture;
    }

    /** @brief A synthetic problem of the given extents. */
    Data::SyntheticOptions data_for(int64_t batch, int64_t batches, int64_t features,
                                    int64_t classes)
    {
        Data::SyntheticOptions options;
        options.batch_size = batch;
        options.batches = batches;
        options.features = features;
        options.classes = classes;
        return options;
    }

    /** @brief Returns how many of @p got lie outside the rounding bound of @p expected. */
    std::size_t outside_relative(const std::vector<double>& got, const std::vector<double>& expected,
                                 std::size_t operations)
    {
        std::size_t outside = 0;
        for (std::size_t i = 0; i < got.size(); ++i)
            outside += TestSupport::measure_relative_error<float>(got[i], expected[i], operations)
                           .is_within() ? 0u : 1u;

        return outside;
    }

    /** @brief Counts the roundings of a log-softmax row of up to four classes: two per class. */
    constexpr std::size_t K_LOG_SOFTMAX_OPERATIONS = 8;

    /** @brief Counts the roundings of a cross-entropy over a batch: log-softmax, product, sums. */
    constexpr std::size_t K_CROSS_ENTROPY_OPERATIONS = 10;

    /** @brief Holds the extent of a GEMM block of the local-tiled kernel. */
    constexpr int64_t K_BLOCK = 64;

    /**
     * @brief Log-softmax of @p logits, read back, for the stability checks.
     * @param rows Rows of the logit matrix.
     * @param columns Columns of the logit matrix.
     * @param logits The values to feed, row-major.
     * @return The [rows, columns] log-probabilities, or the failure of the compilation or the run.
     */
    [[nodiscard]] Result<std::vector<double>> log_softmax_of(SushiAI::ExecutionBackend& backend,
                                                             int64_t rows, int64_t columns,
                                                             const std::vector<double>& logits)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueID x = builder.input(Shape{rows, columns}, "logits");
        SA_TRY_ASSIGN(const ValueID y, builder.log_softmax(x));

        SA_TRY_ASSIGN(const std::unique_ptr<Runner> run, Runner::make(backend, graph));
        SaTest::write_tensor(run->pool.tensor(x), logits);
        SA_TRY(run->compile());
        SA_TRY(run->run());

        return SaTest::read_tensor(run->pool.tensor(y));
    }

    /** @brief One out-of-range AdamW hyperparameter: its options and the words its refusal holds. */
    struct HyperparameterRow
    {
        const char* name;            /**< @brief Names the row in the test name. */
        double beta1;                /**< @brief Holds the first decay rate. */
        double beta2;                /**< @brief Holds the second decay rate. */
        double eps;                  /**< @brief Holds the denominator offset. */
        const char* first_needle;    /**< @brief Holds a word the message names. */
        const char* second_needle;   /**< @brief Holds another. */
    };

    /** @brief One uniform-row magnitude. */
    struct MagnitudeRow
    {
        const char* name;    /**< @brief Names the row in the test name. */
        double magnitude;    /**< @brief Holds every logit of the row. */
    };

    /** @brief Refuses each out-of-range hyperparameter of AdamW. */
    class Medium_ResilienceHyperparameter : public ::testing::TestWithParam<HyperparameterRow> {};

    /** @brief Steps an optimizer before any backward ran, one parameter per row. */
    class Medium_ResilienceFirstStep : public ::testing::TestWithParam<std::size_t> {};

    /** @brief Runs the GEMM or the trainer at one extent. */
    class Medium_ResilienceExtent : public ::testing::TestWithParam<int64_t> {};

    /** @brief Runs a log-softmax over a uniform row. */
    class Medium_ResilienceUniformRow : public ::testing::TestWithParam<MagnitudeRow> {};

    /** @brief Provides a trainer compiled for a batch of 8 and 4 classes, and what it refuses. */
    class Medium_ResilienceFit : public SaTest::EngineTest
    {
        protected:
            /** @brief Builds the trainer. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                SA_ASSERT_OK_AND_ASSIGN(fixture, make_fixture(backend(), 8, 6, 10, 4));
                SA_ASSERT_OK_AND_ASSIGN(
                    optimizer, Optim::SGD::make(backend(), fixture->slots, Optim::SGDOptions{}));
                objective = std::make_unique<Train::CrossEntropyObjective>(
                    SaTest::classifier_objective(fixture->built));
                SA_ASSERT_OK_AND_ASSIGN(
                    trainer, Train::Trainer::make(backend(), fixture->built, *fixture->pool,
                                                  *objective, *optimizer));
            }

            std::optional<Fixture> fixture;                           /**< @brief The classifier. */
            std::unique_ptr<Optim::SGD> optimizer;                    /**< @brief The optimizer. */
            std::unique_ptr<Train::CrossEntropyObjective> objective;  /**< @brief The objective. */
            std::unique_ptr<Train::Trainer> trainer;                  /**< @brief The trainer. */
    };

    /** @brief Provides a trainer over one sample of one feature and three classes. */
    class Medium_ResilienceSingleSample : public SaTest::EngineTest
    {
        protected:
            /** @brief Builds the trainer and the dataset of two batches. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                SA_ASSERT_OK_AND_ASSIGN(built, Train::build_classifier(layer, 1, 1, 3));
                SA_ASSERT_OK_AND_ASSIGN(state, NN::ModelState::create(*engine, layer));
                pool = std::make_unique<TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(slots,
                                        Train::bind_parameters(backend(), built, *state, *pool));
                SA_EXPECT_OK(NN::initialize(backend(), *state));

                Optim::SGDOptions sgd;
                sgd.learning_rate = 0.5;
                SA_ASSERT_OK_AND_ASSIGN(optimizer, Optim::SGD::make(backend(), slots, sgd));

                Train::TrainerOptions options;
                options.track_metrics = false;
                objective = std::make_unique<Train::CrossEntropyObjective>(
                    SaTest::classifier_objective(built));
                SA_ASSERT_OK_AND_ASSIGN(
                    trainer, Train::Trainer::make(backend(), built, *pool, *objective,
                                                  *optimizer, options));
                SA_ASSERT_OK_AND_ASSIGN(
                    dataset, Data::SyntheticClassification::make(backend(), data_for(1, 2, 1, 3)));
            }

            /** @brief Runs one step over batch @p index and returns its loss. */
            double loss_of_step(std::size_t index)
            {
                Data::Batch batch;
                EXPECT_TRUE(SaTest::status_is_ok(dataset->read(index, batch)));
                const Result<Train::StepMetrics> metrics = trainer->run_step(batch);
                EXPECT_TRUE(metrics.has_value());
                return metrics.has_value() ? metrics.value().loss
                                           : std::numeric_limits<double>::quiet_NaN();
            }

            /** @brief Runs one step over batch @p index and expects a finite loss. */
            void expect_finite_step(std::size_t index)
            {
                EXPECT_TRUE(std::isfinite(loss_of_step(index))) << "at batch " << index;
            }

            NN::Linear layer{1, 3};                                    /**< @brief The model. */
            Train::TrainingGraph built;                                /**< @brief The graph. */
            std::optional<NN::ModelState> state;                       /**< @brief The parameters. */
            std::unique_ptr<TensorPool> pool;                          /**< @brief The values. */
            std::vector<Optim::ParameterSlot> slots;                   /**< @brief The slots. */
            std::unique_ptr<Optim::SGD> optimizer;                     /**< @brief The optimizer. */
            std::unique_ptr<Train::CrossEntropyObjective> objective;   /**< @brief The objective. */
            std::unique_ptr<Train::Trainer> trainer;                   /**< @brief The trainer. */
            std::unique_ptr<Data::SyntheticClassification> dataset;    /**< @brief The data. */
    };

    /** @brief Provides the live allocation count and a way to build and destroy a trainer. */
    class Medium_ResilienceLifetime : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns the live allocations of the runtime. */
            static std::size_t live_allocations()
            {
                return SaTest::TestEnvironment::runtime().advanced().live_allocation_count();
            }

            /** @brief Builds a trainer, fits it once and destroys it; the first round sets the baseline. */
            void round(bool first)
            {
                {
                    SA_ASSERT_OK_AND_ASSIGN(Fixture fixture,
                                            make_fixture(backend(), 8, 6, 12, 4));
                    SA_ASSERT_OK_AND_ASSIGN(
                        const std::unique_ptr<Optim::AdamW> optimizer,
                        Optim::AdamW::make(backend(), fixture.slots, Optim::AdamWOptions{}));

                    Train::TrainerOptions options;
                    options.track_metrics = false;
                    const auto objective = SaTest::classifier_objective(fixture.built);
                    SA_ASSERT_OK_AND_ASSIGN(
                        const std::unique_ptr<Train::Trainer> trainer,
                        Train::Trainer::make(backend(), fixture.built, *fixture.pool, objective,
                                             *optimizer, options));
                    SA_ASSERT_OK_AND_ASSIGN(
                        const std::unique_ptr<Data::SyntheticClassification> dataset,
                        Data::SyntheticClassification::make(backend(), data_for(8, 2, 6, 4)));
                    SA_EXPECT_OK(trainer->fit(*dataset));

                    if (first)
                        baseline_bytes = fixture.pool->allocated_bytes();
                    EXPECT_EQ(fixture.pool->allocated_bytes(), baseline_bytes)
                        << "the same model allocated a different amount";
                }

                SaTest::TestEnvironment::runtime().wait().value();

                if (first)
                    baseline = live_allocations();
                else
                    EXPECT_EQ(live_allocations(), baseline)
                        << "a round changed the number of live buffers; a trainer's buffers "
                           "must all die with it";
            }

            std::size_t baseline = 0;         /**< @brief Holds the live buffers after round one. */
            std::size_t baseline_bytes = 0;   /**< @brief Holds the pool bytes of round one. */
    };

    /** @brief Names a row of the hyperparameter table. */
    std::string hyperparameter_name(const ::testing::TestParamInfo<HyperparameterRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a row of the magnitude table. */
    std::string magnitude_name(const ::testing::TestParamInfo<MagnitudeRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a row by its index or extent. */
    std::string number_name(const ::testing::TestParamInfo<int64_t>& info)
    {
        return std::to_string(info.param);
    }

    /** @brief Names a row by its slot index. */
    std::string slot_name(const ::testing::TestParamInfo<std::size_t>& info)
    {
        return std::to_string(info.param);
    }
}

TEST_F(Medium_Resilience, AnOptimizerNamesTheParameterWhoseGradientDoesNotMatch)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter, allocate_tensor(*engine, Shape{4, 3}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient, allocate_tensor(*engine, Shape{4, 2}));

    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"0.weight", parameter, gradient});

    SA_EXPECT_FAILURE(Optim::SGD::make(backend(), slots, Optim::SGDOptions{}),
                      StatusCode::InvalidDescription,
                      "0.weight", "12 elements", "gradient has 8");
}

TEST_F(Medium_Resilience, AnOptimizerNamesBothDtypesWhenAGradientIsTheWrongType)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter,
                            allocate_tensor(*engine, Shape{4}, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient,
                            allocate_tensor(*engine, Shape{4}, DType::Float64));

    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"0.bias", parameter, gradient});

    SA_EXPECT_FAILURE(Optim::SGD::make(backend(), slots, Optim::SGDOptions{}),
                      StatusCode::InvalidDescription, "0.bias", "float32", "float64");
}

/** @brief Tests that SGD refuses a complex parameter and says it serves real floats only. */
TEST_F(Medium_Resilience, SGDRefusesAComplexParameterAndExplainsWhy)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter,
                            allocate_tensor(*engine, Shape{4}, DType::Complex32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient,
                            allocate_tensor(*engine, Shape{4}, DType::Complex32));

    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"0.weight", parameter, gradient});

    SA_EXPECT_FAILURE(Optim::SGD::make(backend(), slots, Optim::SGDOptions{}),
                      StatusCode::InvalidDescription,
                      "0.weight", "complex32", "real floating tensors only");
}

/** @brief Tests that AdamW refuses a half parameter and says why. */
TEST_F(Medium_Resilience, AdamWRefusesAHalfParameterAndExplainsWhy)
{
    // The half refusal must say why in its message; see tests/layers/train/README.md.
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter,
                            allocate_tensor(*engine, Shape{4}, DType::Half));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient,
                            allocate_tensor(*engine, Shape{4}, DType::Half));

    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"0.weight", parameter, gradient});

    SA_EXPECT_FAILURE(Optim::AdamW::make(backend(), slots, Optim::AdamWOptions{}),
                      StatusCode::InvalidDescription,
                      "0.weight", "half precision", "float32");
}

/** @brief Tests that AdamW names the hyperparameter of the row that is out of range. */
TEST_P(Medium_ResilienceHyperparameter, AdamWNamesTheHyperparameterThatIsOutOfRange)
{
    SushiBLAS::Engine& engine = SaTest::TestEnvironment::engine();
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter, allocate_tensor(engine, Shape{4}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient, allocate_tensor(engine, Shape{4}));

    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"0.weight", parameter, gradient});

    Optim::AdamWOptions options;
    options.beta1 = GetParam().beta1;
    options.beta2 = GetParam().beta2;
    options.eps = GetParam().eps;
    SA_EXPECT_FAILURE(Optim::AdamW::make(SaTest::TestEnvironment::backend(), slots, options),
                      StatusCode::InvalidDescription, GetParam().first_needle,
                      GetParam().second_needle);
}

INSTANTIATE_TEST_SUITE_P(Options, Medium_ResilienceHyperparameter,
                         ::testing::Values(HyperparameterRow{"Beta1", 1.0, 0.999, 1e-8, "beta1",
                                                             "[0, 1)"},
                                           HyperparameterRow{"Beta2", 0.9, -0.5, 1e-8, "beta2",
                                                             "[0, 1)"},
                                           HyperparameterRow{"Eps", 0.9, 0.999, 0.0,
                                                             "eps must be positive", "eps"}),
                         hyperparameter_name);

TEST_F(Medium_Resilience, ADisconnectedParameterNamesItselfAndSaysWhatToDo)
{
    NN::Linear layer{4, 3};
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built, Train::build_classifier(layer, 4, 4, 3));
    built.parameters.push_back(Train::ParameterBinding{"ghost", layer.weight.id_in(built.graph),
                                                       Graph::K_NO_VALUE});

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, layer));
    TensorPool pool(*engine);
    SA_EXPECT_FAILURE(Train::bind_parameters(backend(), built, state, pool),
                      StatusCode::InvalidDescription,
                      "'ghost'", "receives no gradient", "no optimizer could ever move it");
}

/** @brief Tests that a step before any backward leaves the row's parameter as it was. */
TEST_P(Medium_ResilienceFirstStep, AnOptimizerStepBeforeAnyBackwardIsANoOpRatherThanGarbage)
{
    // An unwritten gradient must read as zero; see tests/layers/train/README.md.
    SA_ASSERT_OK_AND_ASSIGN(Fixture fixture,
                            make_fixture(SaTest::TestEnvironment::backend(), 8, 6, 12, 4));
    ASSERT_GT(fixture.slots.size(), GetParam());
    const Optim::ParameterSlot& slot = fixture.slots[GetParam()];

    const std::vector<double> gradient = SaTest::read_tensor(slot.gradient);
    ASSERT_TRUE(all_finite(gradient, slot.name + ".grad"));
    EXPECT_TRUE(std::all_of(gradient.begin(), gradient.end(),
                            [](double value) { return value == 0.0; }))
        << "'" << slot.name << "' gradient holds a value before any backward pass has run";

    const std::vector<double> before = SaTest::read_tensor(slot.parameter);

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Optim::SGD> optimizer,
        Optim::SGD::make(SaTest::TestEnvironment::backend(), fixture.slots, sgd));
    SA_EXPECT_OK(optimizer->step());

    EXPECT_EQ(SaTest::read_tensor(slot.parameter), before)
        << "'" << slot.name << "' moved on a step with no gradient";
}

INSTANTIATE_TEST_SUITE_P(Parameters, Medium_ResilienceFirstStep, ::testing::Values(0u, 1u, 2u, 3u),
                         slot_name);

TEST_F(Medium_Resilience, TheTrainerNamesTheBatchThatDoesNotFitTheCompiledStep)
{
    SA_ASSERT_OK_AND_ASSIGN(Fixture fixture, make_fixture(backend(), 8, 6, 10, 4));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend(), fixture.slots, Optim::SGDOptions{}));
    const auto objective = SaTest::classifier_objective(fixture.built);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(backend(), fixture.built, *fixture.pool,
                                                 objective, *optimizer));

    // A wrong-sized batch can only be reported; see tests/layers/train/README.md.
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> wrong,
                            Data::SyntheticClassification::make(backend(), data_for(4, 1, 6, 4)));

    Data::Batch wrong_batch;
    ASSERT_TRUE(SaTest::status_is_ok(wrong->read(0, wrong_batch)));
    SA_EXPECT_FAILURE(trainer->run_step(wrong_batch), StatusCode::Incompatible,
                      "[4, 6]", "[8, 6]");
}

/** @brief Tests that fit refuses a dataset without batches. */
TEST_F(Medium_ResilienceFit, FitNamesTheEmptyDataset)
{
    const NoBatches empty(8, 6, 4);
    SA_EXPECT_FAILURE(trainer->fit(empty), StatusCode::InvalidDescription,
                      "The dataset holds no batches");
}

/** @brief Tests that fit refuses a dataset whose batch size is not the compiled one. */
TEST_F(Medium_ResilienceFit, FitNamesTheBatchSizeThatDisagrees)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> narrow,
                            Data::SyntheticClassification::make(backend(), data_for(4, 1, 6, 4)));
    SA_EXPECT_FAILURE(trainer->fit(*narrow), StatusCode::Incompatible,
                      "batch size is 4", "compiled for 8");
}

/** @brief Tests that fit refuses a dataset whose class count is not the compiled one. */
TEST_F(Medium_ResilienceFit, FitNamesTheClassCountThatDisagrees)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> wrong_classes,
                            Data::SyntheticClassification::make(backend(), data_for(8, 1, 6, 5)));
    SA_EXPECT_FAILURE(trainer->fit(*wrong_classes), StatusCode::Incompatible,
                      "5 classes", "compiled for 4");
}

/** @brief Tests that a refused dataset runs no step. */
TEST_F(Medium_ResilienceFit, FitRunsNoStepAfterARefusedDataset)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> narrow,
                            Data::SyntheticClassification::make(backend(), data_for(4, 1, 6, 4)));

    EXPECT_FALSE(trainer->fit(*narrow).has_value());
    EXPECT_EQ(trainer->step_count(), 0u);
}

/** @brief Tests that allocation refuses a zero extent and names it. */
TEST_F(Medium_Resilience, AllocationRefusesAZeroExtent)
{
    const int64_t empty_dims[2] = {4, 0};
    SA_EXPECT_FAILURE(allocate_tensor(*engine, SushiRuntime::span<const int64_t>(empty_dims, 2)),
                      StatusCode::InvalidDescription, "Tensor extents must be positive", "got 0");
}

/** @brief Tests that allocation refuses an element count that overflows. */
TEST_F(Medium_Resilience, AllocationRefusesAnOverflowingElementCount)
{
    // Overflow is detected before the multiply; see tests/layers/train/README.md.
    const int64_t overflowing[3] = {3'000'000'000LL, 3'000'000'000LL, 3'000'000'000LL};
    SA_EXPECT_FAILURE(allocate_tensor(*engine, SushiRuntime::span<const int64_t>(overflowing, 3)),
                      StatusCode::InvalidDescription, "more than", "elements", "3000000000");
}

TEST_F(Medium_Resilience, AModelTooLargeToAllocateNamesTheTensorRatherThanTheKernel)
{
    // A petabyte: the error names shape and dtype; see tests/layers/train/README.md.
    const int64_t enormous[2] = {1LL << 25, 1LL << 25};

    SA_EXPECT_FAILURE(allocate_tensor(*engine, SushiRuntime::span<const int64_t>(enormous, 2)),
                      StatusCode::OutOfMemory,
                      "Could not allocate the float32 tensor", "[33554432, 33554432]");
}

/** @brief Tests that the first loss of one sample of one feature is ln(3). */
TEST_F(Medium_ResilienceSingleSample, TheFirstLossIsLogC)
{
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
        loss_of_step(0), std::log(3.0), K_CROSS_ENTROPY_OPERATIONS)));
}

/** @brief Tests that eight more steps keep the loss, the weight and its gradient finite. */
TEST_F(Medium_ResilienceSingleSample, EightMoreStepsStayFinite)
{
    // N = 1 and D = 1 with one Linear: every loop bound is at its minimum.
    expect_finite_step(0);
    expect_finite_step(1);
    expect_finite_step(0);
    expect_finite_step(1);
    expect_finite_step(0);
    expect_finite_step(1);
    expect_finite_step(0);
    expect_finite_step(1);

    EXPECT_TRUE(all_finite(SaTest::read_tensor(slots[0].parameter), "weight"));
    EXPECT_TRUE(all_finite(SaTest::read_tensor(slots[0].gradient), "gradient"));
}

TEST_F(Medium_Resilience, ASingleClassIsDegenerateButNotUndefined)
{
    // With C = 1 loss and gradient are exactly zero; see tests/layers/train/README.md.
    const int64_t batch = 4;
    const int64_t features = 3;
    const int64_t classes = 1;

    NN::Linear layer{features, classes};
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(layer, batch, features, classes));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, layer));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(backend(), built, state, pool));

    NN::InitPolicy policy;
    policy.zero_final_layer = false;
    SA_EXPECT_OK(NN::initialize(backend(), state, policy));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend(), slots, Optim::AdamWOptions{}));
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(backend(), built, pool, objective,
                                                 *optimizer));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(
                                backend(), data_for(batch, 1, features, classes)));

    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                            trainer->run_step(dataset_batch));

    EXPECT_EQ(metrics.loss, 0.0);
    EXPECT_EQ(metrics.metrics.value_or("accuracy", 0.0), 1.0);
    EXPECT_TRUE(all_finite(SaTest::read_tensor(slots[0].parameter), "weight"));

    const std::vector<double> gradient = SaTest::read_tensor(slots[0].gradient);
    EXPECT_TRUE(all_finite(gradient, "gradient"));
    EXPECT_TRUE(std::all_of(gradient.begin(), gradient.end(),
                            [](double value) { return value == 0.0; }));
}

/** @brief Tests that the GEMM at the row's extent matches the host reference of its own floats. */
TEST_P(Medium_ResilienceExtent, MatmulMatchesTheHostAtTheRegisterBlockEdgeAndOnePastIt)
{
    // 64 fills a GEMM block; 65 leaves a remainder; see tests/layers/train/README.md.
    const int64_t n = GetParam();
    SushiAI::ExecutionBackend& backend = SaTest::TestEnvironment::backend();

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{n, n}, "x");
    const ValueID w = builder.input(Shape{n, n}, "w");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.matmul(x, w, false, true));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Runner> run, Runner::make(backend, graph));

    std::vector<double> x_values(static_cast<std::size_t>(n * n));
    std::vector<double> w_values(static_cast<std::size_t>(n * n));
    SaTest::HostNoise(0xB10Cu + static_cast<std::uint64_t>(n)).fill(x_values, 1.0);
    SaTest::HostNoise(0xB10Du + static_cast<std::uint64_t>(n)).fill(w_values, 1.0);

    SaTest::write_tensor(run->pool.tensor(x), x_values);
    SaTest::write_tensor(run->pool.tensor(w), w_values);

    ASSERT_TRUE(SaTest::status_is_ok(run->compile()));
    ASSERT_TRUE(SaTest::status_is_ok(run->run()));

    const std::vector<double> got = SaTest::read_tensor(run->pool.tensor(y));
    ASSERT_EQ(got.size(), static_cast<std::size_t>(n * n));

    // The reference uses the device's floats; see tests/layers/train/README.md.
    const std::vector<double> x_seen = SaTest::read_tensor(run->pool.tensor(x));
    const std::vector<double> w_seen = SaTest::read_tensor(run->pool.tensor(w));

    std::size_t outside = 0;
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
        {
            double expected = 0.0;
            double magnitude = 0.0;
            for (int64_t k = 0; k < n; ++k)
            {
                const double product = x_seen[static_cast<std::size_t>(i * n + k)]
                                     * w_seen[static_cast<std::size_t>(j * n + k)];
                expected += product;
                magnitude += std::abs(product);
            }

            outside += TestSupport::measure_sum_error<float>(
                           got[static_cast<std::size_t>(i * n + j)], expected, magnitude,
                           static_cast<std::size_t>(n)).is_within() ? 0u : 1u;
        }

    EXPECT_EQ(outside, 0u) << "n=" << n;
}

/** @brief Tests that four training steps at the row's extent keep every value finite. */
TEST_P(Medium_ResilienceExtent, TrainingIsFiniteAtTheRegisterBlockEdgeAndOnePastIt)
{
    const int64_t n = GetParam();
    SushiAI::ExecutionBackend& backend = SaTest::TestEnvironment::backend();

    SA_ASSERT_OK_AND_ASSIGN(Fixture fixture, make_fixture(backend, n, n, n, 4));
    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend, fixture.slots, sgd));

    Train::TrainerOptions options;
    options.track_metrics = false;
    const auto objective = SaTest::classifier_objective(fixture.built);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(backend, fixture.built, *fixture.pool, objective,
                                                 *optimizer, options));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(backend, data_for(n, 2, n, 4)));

    Data::Batch first;
    Data::Batch second;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, first)));
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, second)));
    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics step_0, trainer->run_step(first));
    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics step_1, trainer->run_step(second));
    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics step_2, trainer->run_step(first));
    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics step_3, trainer->run_step(second));

    EXPECT_TRUE(std::isfinite(step_0.loss) && std::isfinite(step_1.loss) &&
                std::isfinite(step_2.loss) && std::isfinite(step_3.loss)) << "n=" << n;
    EXPECT_TRUE(std::all_of(fixture.slots.begin(), fixture.slots.end(),
                            [](const Optim::ParameterSlot& slot)
                            { return all_finite(SaTest::read_tensor(slot.parameter), slot.name) &&
                                     all_finite(SaTest::read_tensor(slot.gradient), slot.name); }))
        << "n=" << n;
}

INSTANTIATE_TEST_SUITE_P(Extents, Medium_ResilienceExtent,
                         ::testing::Values(K_BLOCK, K_BLOCK + 1), number_name);

/** @brief Tests that a uniform row of the row's magnitude gives -ln(C) at every element. */
TEST_P(Medium_ResilienceUniformRow, LogSoftmaxOfUniformLogitsIsMinusLogCAtAnyMagnitude)
{
    // A uniform row gives -log(C) at any offset; see tests/layers/train/README.md.
    const int64_t rows = 3;
    const int64_t columns = 4;
    const std::vector<double> logits(static_cast<std::size_t>(rows * columns),
                                     GetParam().magnitude);

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> got,
                            log_softmax_of(SaTest::TestEnvironment::backend(), rows, columns,
                                           logits));

    ASSERT_TRUE(all_finite(got, "log_softmax"));
    const std::vector<double> expected(got.size(), -std::log(static_cast<double>(columns)));
    EXPECT_EQ(outside_relative(got, expected, K_LOG_SOFTMAX_OPERATIONS), 0u);
}

INSTANTIATE_TEST_SUITE_P(Magnitudes, Medium_ResilienceUniformRow,
                         ::testing::Values(MagnitudeRow{"Zero", 0.0}, MagnitudeRow{"One", 1.0},
                                           MagnitudeRow{"MinusOne", -1.0},
                                           MagnitudeRow{"Huge", 1e30},
                                           MagnitudeRow{"MinusHuge", -1e30},
                                           MagnitudeRow{"NearFloatMax", 3.0e38}),
                         magnitude_name);

TEST_F(Medium_Resilience, LogSoftmaxSurvivesBothExtremesInsideOneRow)
{
    // The winner's log-probability is exactly zero; see tests/layers/train/README.md.
    const int64_t rows = 2;
    const int64_t columns = 3;

    std::vector<double> logits = {
        1e30, -1e30, -1e30,
        -1e30, 1e30, -1e30,
    };

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> got,
                            log_softmax_of(backend(), rows, columns, logits));

    ASSERT_TRUE(all_finite(got, "log_softmax"));
    EXPECT_EQ(got[0], 0.0);
    EXPECT_EQ(got[4], 0.0);

    // The loser sits one gap below the winner; the gap is two float(1e30).
    const double gap = -2.0 * static_cast<double>(static_cast<float>(1e30));
    EXPECT_TRUE(TestSupport::is_within(
        TestSupport::measure_relative_error<float>(got[1], gap, K_LOG_SOFTMAX_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(
        TestSupport::measure_relative_error<float>(got[3], gap, K_LOG_SOFTMAX_OPERATIONS)));
}

/** @brief Tests that rows 60 apart give the closed-form log-probabilities. */
TEST_F(Medium_Resilience, LogSoftmaxIsExactAcrossRowsThatSpanTheExponentRange)
{
    // Each row is shifted by its own maximum; see tests/layers/train/README.md.
    std::vector<double> logits = {
        60.0, 59.0, 58.0,   // the batch maximum
        0.0, -1.0, -2.0,    // 60 below it, still exact
    };

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> got,
                            log_softmax_of(backend(), 2, 3, logits));
    ASSERT_TRUE(all_finite(got, "log_softmax"));

    // Both rows have the same spread, so both give -ln(D), -1 - ln(D) and -2 - ln(D).
    const double minus_log_denominator = -std::log(1.0 + std::exp(-1.0) + std::exp(-2.0));
    const std::vector<double> closed_form = {
        minus_log_denominator, minus_log_denominator - 1.0, minus_log_denominator - 2.0,
        minus_log_denominator, minus_log_denominator - 1.0, minus_log_denominator - 2.0,
    };

    EXPECT_EQ(outside_relative(got, closed_form, K_LOG_SOFTMAX_OPERATIONS), 0u);
}

/** @brief Tests that a correct, perfectly confident prediction costs nothing. */
TEST_F(Medium_Resilience, CrossEntropyOfAConfidentCorrectPredictionIsZero)
{
    const int64_t rows = 2;
    const int64_t columns = 3;

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID logits = builder.input(Shape{rows, columns}, "logits");
    const ValueID targets = builder.input(Shape{rows, columns}, "targets");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            NN::CrossEntropyLoss{}.forward(builder, logits, targets));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Runner> run, Runner::make(backend(), graph));

    // exp(80) overflows without the shift; see tests/layers/train/README.md.
    SaTest::write_tensor(run->pool.tensor(logits),
                         std::vector<double>{80.0, -80.0, -80.0, -80.0, 80.0, -80.0});
    SaTest::write_tensor(run->pool.tensor(targets),
                         std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0});

    ASSERT_TRUE(SaTest::status_is_ok(run->compile()));
    ASSERT_TRUE(SaTest::status_is_ok(run->run()));

    // The shifted row is {0, -160, -160}, whose log-sum-exp is log(1) = 0.
    const double correct = SaTest::read_tensor(run->pool.tensor(loss))[0];
    EXPECT_TRUE(std::isfinite(correct));
    EXPECT_EQ(correct, 0.0);
}

/** @brief Tests that a confidently wrong prediction costs the gap of 160, finite. */
TEST_F(Medium_Resilience, CrossEntropyOfAConfidentWrongPredictionIsTheLogitGap)
{
    const int64_t rows = 2;
    const int64_t columns = 3;

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID logits = builder.input(Shape{rows, columns}, "logits");
    const ValueID targets = builder.input(Shape{rows, columns}, "targets");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            NN::CrossEntropyLoss{}.forward(builder, logits, targets));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Runner> run, Runner::make(backend(), graph));

    SaTest::write_tensor(run->pool.tensor(logits),
                         std::vector<double>{80.0, -80.0, -80.0, -80.0, 80.0, -80.0});
    SaTest::write_tensor(run->pool.tensor(targets),
                         std::vector<double>{0.0, 1.0, 0.0, 1.0, 0.0, 0.0});

    ASSERT_TRUE(SaTest::status_is_ok(run->compile()));
    ASSERT_TRUE(SaTest::status_is_ok(run->run()));

    // Each row's target sits 80 - (-80) = 160 under its winner, so the mean loss is 160.
    const double wrong = SaTest::read_tensor(run->pool.tensor(loss))[0];
    EXPECT_TRUE(std::isfinite(wrong)) << "loss on a confidently wrong prediction";
    EXPECT_TRUE(TestSupport::is_within(
        TestSupport::measure_relative_error<float>(wrong, 160.0, K_CROSS_ENTROPY_OPERATIONS)));
}

/** @brief Tests that AdamW's first step with a zero gradient leaves the parameter as it was. */
TEST_F(Medium_Resilience, AdamWsFirstStepWithAZeroGradientLeavesTheParameter)
{
    // eps is added after the bias correction; see tests/layers/train/README.md.
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter, allocate_tensor(*engine, Shape{6}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient, allocate_tensor(*engine, Shape{6}));

    const std::vector<double> initial = {0.5, -1.5, 0.0, 2.0, -0.25, 100.0};
    SaTest::write_tensor(parameter, initial);
    SaTest::write_tensor(gradient, std::vector<double>(6, 0.0));

    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"p", parameter, gradient});

    Optim::AdamWOptions options;
    options.learning_rate = 0.1;
    options.weight_decay = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend(), slots, options));

    SA_EXPECT_OK(optimizer->step());

    // With no decay and no gradient the update is 0 / (0 + eps) = 0, so the bytes hold.
    const std::vector<double> after = SaTest::read_tensor(parameter);
    ASSERT_TRUE(all_finite(after, "parameter"));
    EXPECT_EQ(after, initial);
}

/** @brief Tests that AdamW's second step with a zero gradient stays finite. */
TEST_F(Medium_Resilience, AdamWsSecondStepWithAZeroGradientStaysFinite)
{
    // The bias corrections differ on step two; see tests/layers/train/README.md.
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor parameter, allocate_tensor(*engine, Shape{6}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor gradient, allocate_tensor(*engine, Shape{6}));

    SaTest::write_tensor(parameter, std::vector<double>{0.5, -1.5, 0.0, 2.0, -0.25, 100.0});
    SaTest::write_tensor(gradient, std::vector<double>(6, 0.0));

    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"p", parameter, gradient});

    Optim::AdamWOptions options;
    options.learning_rate = 0.1;
    options.weight_decay = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(backend(), slots, options));

    SA_EXPECT_OK(optimizer->step());
    SA_EXPECT_OK(optimizer->step());
    EXPECT_TRUE(all_finite(SaTest::read_tensor(parameter), "parameter after two steps"));
}

/** @brief Tests that building and destroying a trainer five times leaves the allocations as round one did. */
TEST_F(Medium_ResilienceLifetime, BuildingAndDestroyingTrainersReachesASteadyState)
{
    // The runtime counts its live USM allocations; see tests/layers/train/README.md.
    round(true);
    round(false);
    round(false);
    round(false);
    round(false);
}

TEST_F(Medium_ResilienceDeathTest, BatchNormUpdateStatsIsNotAnIROperationAndSaysSo)
{
    // BATCHNORM_UPDATE_STATS has no lowering; see tests/layers/train/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID running = builder.parameter(Shape{4}, "running_mean");
    const ValueID batch = builder.input(Shape{4}, "batch_mean");
    const ValueID out = builder.like(running, "updated_running_mean");
    graph.add_node(Ops::K_BATCHNORM_UPDATE_STATS.id, {running, batch}, {out});
    SA_EXPECT_OK(graph.finalise());

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.adopt(Graph::MemoryPlan::build(graph).value()));
    SA_EXPECT_OK(pool.materialise_leaves(graph));

    SA_EXPECT_FATAL((void)CompiledStep::compile(backend(), graph, pool),
                    "No lowering for operation", "sushiai.batchnorm_update_stats");
}
