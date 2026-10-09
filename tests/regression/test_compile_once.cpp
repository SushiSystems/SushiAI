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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_compile_once.cpp
 * @brief Tests that a training step compiles once and that every replay of it dispatches work.
 * @author Mustafa Garip
 * @see tests/regression/README.md
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/regression/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;

namespace
{
    class Regression_CompileOnce : public SaTest::EngineTest {};

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 8;
    constexpr int64_t K_FEATURES = 6;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 4;

    Mlp make_model()
    {
        return Mlp(NN::Linear{K_FEATURES, 12}, NN::ReLU{}, NN::Linear{12, K_CLASSES});
    }

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
 * @see tests/regression/README.md
 */
TEST_F(Regression_CompileOnce, ManyFitCallsCompileTheStepExactlyOnce)
{
    Mlp model = make_model();
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    SA_ASSERT_OK_AND_ASSIGN(auto optimizer, Optim::SGD::make(*engine, slots, sgd));

    Train::TrainerOptions options;
    options.epochs = 1;
    SA_ASSERT_OK_AND_ASSIGN(auto trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer, options));

    SA_ASSERT_OK_AND_ASSIGN(auto dataset,
                            Data::SyntheticClassification::make(*engine, make_data_options()));

    ASSERT_EQ(trainer->compile_count(), 1u);

    std::size_t steps = 0;
    for (int epoch = 0; epoch < 6; ++epoch)
    {
        SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));
        steps += report.steps;

        EXPECT_EQ(report.compile_count, 1u) << "recompiled during epoch " << epoch;
        EXPECT_EQ(trainer->compile_count(), 1u) << "recompiled during epoch " << epoch;
    }

    EXPECT_EQ(steps, static_cast<std::size_t>(6 * K_BATCHES));
    EXPECT_EQ(trainer->step_count(), steps);
}

/**
 * @brief Tests that five replayed steps each change the loss and move a parameter.
 * @see tests/regression/README.md
 */
TEST_F(Regression_CompileOnce, EveryReplayDispatchesRatherThanReportingCompleteImmediately)
{
    Mlp model = make_model();
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    SA_ASSERT_OK_AND_ASSIGN(auto optimizer, Optim::SGD::make(*engine, slots, sgd));

    SA_ASSERT_OK_AND_ASSIGN(auto trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer));
    SA_ASSERT_OK_AND_ASSIGN(auto dataset,
                            Data::SyntheticClassification::make(*engine, make_data_options()));

    const std::vector<double> initial = SaTest::read_tensor(slots[0].parameter);

    double previous_loss = 0.0;
    for (std::size_t step = 0; step < 5; ++step)
    {
        SA_ASSERT_OK_AND_ASSIGN(
            const Train::StepMetrics metrics,
            trainer->run_step(dataset->batch(step % static_cast<std::size_t>(K_BATCHES))));

        ASSERT_TRUE(std::isfinite(metrics.loss)) << "at step " << step;
        EXPECT_GT(metrics.loss, 0.0) << "at step " << step;
        if (step > 0)
            EXPECT_NE(metrics.loss, previous_loss) << "the step stopped doing work at " << step;
        previous_loss = metrics.loss;
    }

    const std::vector<double> trained = SaTest::read_tensor(slots[0].parameter);
    ASSERT_EQ(initial.size(), trained.size());

    std::size_t moved = 0;
    for (std::size_t i = 0; i < initial.size(); ++i)
        if (initial[i] != trained[i])
            ++moved;

    EXPECT_GT(moved, 0u) << "no parameter moved across five replays";
    EXPECT_EQ(trainer->compile_count(), 1u);
}

/**
 * @brief Tests that the first run() and three replays overwrite a poisoned output buffer.
 * @see tests/regression/README.md
 */
TEST_F(Regression_CompileOnce,
       TheVeryFirstRunDispatchesRatherThanReportingCompleteOverAnEmptyBaseline)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueId x = builder.input(Shape{2, 3}, "x");
    const Graph::ValueId y = builder.scale(x, 2.0);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));

    const std::vector<double> inputs = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    SaTest::write_tensor(pool.tensor(x), inputs);

    SA_ASSERT_OK_AND_ASSIGN(auto step, Graph::CompiledStep::compile(*engine, graph, pool));
    ASSERT_FALSE(step->empty()) << "this graph has real work in it";
    ASSERT_EQ(step->compile_count(), 1u);
    ASSERT_EQ(step->run_count(), 0u);

    constexpr double K_POISON = -12345.0;
    SaTest::write_tensor(pool.tensor(y), std::vector<double>(inputs.size(), K_POISON));

    SA_EXPECT_OK(step->run());

    const std::vector<double> first = SaTest::read_tensor(pool.tensor(y));
    for (std::size_t i = 0; i < inputs.size(); ++i)
        ASSERT_NEAR(first[i], inputs[i] * 2.0, 1e-5)
            << "the first run() left element " << i << " untouched - the plan reported "
               "complete without dispatching anything";

    EXPECT_EQ(step->run_count(), 1u);

    // Each replay must track the new input; see tests/regression/README.md.
    for (int replay = 0; replay < 3; ++replay)
    {
        std::vector<double> next(inputs.size());
        for (std::size_t i = 0; i < next.size(); ++i)
            next[i] = inputs[i] + static_cast<double>(replay + 1) * 10.0;

        SaTest::write_tensor(pool.tensor(x), next);
        SaTest::write_tensor(pool.tensor(y), std::vector<double>(inputs.size(), K_POISON));

        SA_EXPECT_OK(step->run());

        const std::vector<double> got = SaTest::read_tensor(pool.tensor(y));
        for (std::size_t i = 0; i < next.size(); ++i)
            ASSERT_NEAR(got[i], next[i] * 2.0, 1e-5)
                << "replay " << replay << " did not dispatch at element " << i;
    }

    EXPECT_EQ(step->run_count(), 4u);
    EXPECT_EQ(step->compile_count(), 1u);
}

/**
 * @brief Tests that swapping data, changing the learning rate and resuming fit() never recompile
 *        the step.
 * @see tests/regression/README.md
 */
TEST_F(Regression_CompileOnce, CompileCountSurvivesEverythingTheLoopDoesBetweenSteps)
{
    Mlp model = make_model();
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    Optim::AdamWOptions adam;
    adam.learning_rate = 0.02;
    SA_ASSERT_OK_AND_ASSIGN(auto optimizer, Optim::AdamW::make(*engine, slots, adam));

    Train::TrainerOptions options;
    options.track_accuracy = false;
    SA_ASSERT_OK_AND_ASSIGN(auto trainer,
                            Train::Trainer::make(*engine, built.graph, pool, built.spec,
                                                 *optimizer, options));

    // The plan reads its inputs by address; see tests/regression/README.md.
    Data::SyntheticOptions other = make_data_options();
    other.seed = 0xFEED'BEEFull;

    SA_ASSERT_OK_AND_ASSIGN(auto first,
                            Data::SyntheticClassification::make(*engine, make_data_options()));
    SA_ASSERT_OK_AND_ASSIGN(auto second, Data::SyntheticClassification::make(*engine, other));

    ASSERT_EQ(trainer->compile_count(), 1u);

    for (int round = 0; round < 4; ++round)
    {
        // fit() returns and is entered again; see tests/regression/README.md.
        SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report,
                                trainer->fit(round % 2 == 0 ? *first : *second));
        EXPECT_EQ(report.compile_count, 1u) << "recompiled in round " << round;

        // The learning rate is a host-side scalar; see tests/regression/README.md.
        optimizer->options().learning_rate *= 0.5;

        // Individual steps between fits, mixing the two datasets.
        for (std::size_t batch = 0; batch < 2; ++batch)
        {
            SA_EXPECT_OK(trainer->run_step(second->batch(batch)));
            SA_EXPECT_OK(trainer->run_step(first->batch(batch)));
        }

        EXPECT_EQ(trainer->compile_count(), 1u) << "recompiled after round " << round;
    }

    EXPECT_EQ(trainer->step_count(), static_cast<std::size_t>(4 * (K_BATCHES + 4)));
    EXPECT_EQ(trainer->step().run_count(), trainer->step_count());
    EXPECT_EQ(trainer->compile_count(), 1u);
}
