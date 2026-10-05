/****************************************************************/
/* test_compile_once.cpp                                        */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_compile_once.cpp
 * @brief Tests that a training step compiles once and that every replay of it dispatches work.
 * @author Mustafa Garip
 * @see tests/functional/regression/README.md
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_utils.hpp"

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/functional/regression/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;

namespace
{
    class Regression_CompileOnce : public SaTest::EngineTest {};

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t BATCH = 8;
    constexpr int64_t FEATURES = 6;
    constexpr int64_t CLASSES = 4;
    constexpr int64_t BATCHES = 4;

    Mlp make_model()
    {
        return Mlp(NN::Linear{FEATURES, 12}, NN::ReLU{}, NN::Linear{12, CLASSES});
    }

    Data::SyntheticOptions make_data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = BATCH;
        options.batches = BATCHES;
        options.features = FEATURES;
        options.classes = CLASSES;
        return options;
    }
}

/**
 * @brief Tests that repeated fit() calls leave the trainer's compile count at one.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_CompileOnce, ManyFitCallsCompileTheStepExactlyOnce)
{
    Mlp model = make_model();
    Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    Optim::SGD optimizer(*engine, slots, sgd);

    Train::TrainerOptions options;
    options.epochs = 1;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    const Data::SyntheticClassification dataset(*engine, make_data_options());

    ASSERT_EQ(trainer.compile_count(), 1u);

    std::size_t steps = 0;
    for (int epoch = 0; epoch < 6; ++epoch)
    {
        const Train::RunReport report = trainer.fit(dataset);
        steps += report.steps;

        EXPECT_EQ(report.compile_count, 1u) << "recompiled during epoch " << epoch;
        EXPECT_EQ(trainer.compile_count(), 1u) << "recompiled during epoch " << epoch;
    }

    EXPECT_EQ(steps, static_cast<std::size_t>(6 * BATCHES));
    EXPECT_EQ(trainer.step_count(), steps);
}

/**
 * @brief Tests that five replayed steps each change the loss and move a parameter.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_CompileOnce, EveryReplayDispatchesRatherThanReportingCompleteImmediately)
{
    Mlp model = make_model();
    Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    Optim::SGD optimizer(*engine, slots, sgd);

    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer);
    const Data::SyntheticClassification dataset(*engine, make_data_options());

    const std::vector<double> initial = SaTest::read_tensor(slots[0].parameter);

    double previous_loss = 0.0;
    for (std::size_t step = 0; step < 5; ++step)
    {
        const Train::StepMetrics metrics =
            trainer.run_step(dataset.batch(step % static_cast<std::size_t>(BATCHES)));

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
        if (initial[i] != trained[i]) ++moved;

    EXPECT_GT(moved, 0u) << "no parameter moved across five replays";
    EXPECT_EQ(trainer.compile_count(), 1u);
}

/**
 * @brief Tests that the first run() and three replays overwrite a poisoned output buffer.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_CompileOnce, TheVeryFirstRunDispatchesRatherThanReportingCompleteOverAnEmptyBaseline)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueId x = builder.input(Shape{2, 3}, "x");
    const Graph::ValueId y = builder.scale(x, 2.0);

    TensorPool pool(*engine);
    pool.materialise_leaves(graph);

    const std::vector<double> inputs = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    SaTest::write_tensor(pool.tensor(x), inputs);

    Graph::CompiledStep step(*engine, graph, pool);
    ASSERT_FALSE(step.empty()) << "this graph has real work in it";
    ASSERT_EQ(step.compile_count(), 1u);
    ASSERT_EQ(step.run_count(), 0u);

    constexpr double POISON = -12345.0;
    SaTest::write_tensor(pool.tensor(y), std::vector<double>(inputs.size(), POISON));

    step.run();

    const std::vector<double> first = SaTest::read_tensor(pool.tensor(y));
    for (std::size_t i = 0; i < inputs.size(); ++i)
        ASSERT_NEAR(first[i], inputs[i] * 2.0, 1e-5)
            << "the first run() left element " << i << " untouched - the plan reported "
               "complete without dispatching anything";

    EXPECT_EQ(step.run_count(), 1u);

    // Each replay must track the new input; see tests/functional/regression/README.md.
    for (int replay = 0; replay < 3; ++replay)
    {
        std::vector<double> next(inputs.size());
        for (std::size_t i = 0; i < next.size(); ++i)
            next[i] = inputs[i] + static_cast<double>(replay + 1) * 10.0;

        SaTest::write_tensor(pool.tensor(x), next);
        SaTest::write_tensor(pool.tensor(y), std::vector<double>(inputs.size(), POISON));

        step.run();

        const std::vector<double> got = SaTest::read_tensor(pool.tensor(y));
        for (std::size_t i = 0; i < next.size(); ++i)
            ASSERT_NEAR(got[i], next[i] * 2.0, 1e-5)
                << "replay " << replay << " did not dispatch at element " << i;
    }

    EXPECT_EQ(step.run_count(), 4u);
    EXPECT_EQ(step.compile_count(), 1u);
}

/**
 * @brief Tests that swapping data, changing the learning rate and resuming fit() never recompile
 *        the step.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_CompileOnce, CompileCountSurvivesEverythingTheLoopDoesBetweenSteps)
{
    Mlp model = make_model();
    Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::AdamWOptions adam;
    adam.learning_rate = 0.02;
    Optim::AdamW optimizer(*engine, slots, adam);

    Train::TrainerOptions options;
    options.track_accuracy = false;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    // The plan reads its inputs by address; see tests/functional/regression/README.md.
    Data::SyntheticOptions other = make_data_options();
    other.seed = 0xFEED'BEEFull;

    const Data::SyntheticClassification first(*engine, make_data_options());
    const Data::SyntheticClassification second(*engine, other);

    ASSERT_EQ(trainer.compile_count(), 1u);

    for (int round = 0; round < 4; ++round)
    {
        // fit() returns and is entered again; see tests/functional/regression/README.md.
        const Train::RunReport report = trainer.fit(round % 2 == 0 ? first : second);
        EXPECT_EQ(report.compile_count, 1u) << "recompiled in round " << round;

        // The learning rate is a host-side scalar; see tests/functional/regression/README.md.
        optimizer.options().learning_rate *= 0.5;

        // Individual steps between fits, mixing the two datasets.
        for (std::size_t batch = 0; batch < 2; ++batch)
        {
            trainer.run_step(second.batch(batch));
            trainer.run_step(first.batch(batch));
        }

        EXPECT_EQ(trainer.compile_count(), 1u) << "recompiled after round " << round;
    }

    EXPECT_EQ(trainer.step_count(), static_cast<std::size_t>(4 * (BATCHES + 4)));
    EXPECT_EQ(trainer.step().run_count(), trainer.step_count());
    EXPECT_EQ(trainer.compile_count(), 1u);
}
