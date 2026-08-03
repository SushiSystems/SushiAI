/**************************************************************************/
/* test_compile_once.cpp                                                  */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// The regression tier's first inhabitant, and the two claims in this library
// whose failure is completely silent.
//
// "Trace once, differentiate once, compile once, replay for ever" is the
// premise the whole design is arranged around. Both halves of it fail without
// a symptom:
//
//  - Recompiling per step still converges. It just pays graph analysis every
//    step, and nothing in the output says so.
//  - A replay that never dispatched still reports complete. CompiledStep
//    exists because rearm() before capture_base_state() leaves the completion
//    sentinel armed at in-degree zero, so run() returns having done nothing -
//    and the loss simply stops moving, which reads like a bad learning rate.
//
// Nothing upstream can catch either: sushiruntime has no test that replays a
// raw CompiledGraph, and sushiblas cannot see a training step at all. So they
// are pinned here, in the shape the trainer and `sa demo mlp` actually use.
//
// The layout below is deliberately flat rather than shared through a fixture
// struct: Trainer holds raw pointers into the pool and the optimizer, so the
// construction order (bind -> initialise -> optimizer -> trainer) is part of
// the contract, and a test that spells it out is a test that fails loudly if
// that order ever stops working. It matches functional/integration/
// test_training.cpp line for line for the same reason.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_utils.hpp"

using namespace SushiAI;

// `Test` alone resolves to ::testing::Test inside a TEST_F body, where the
// fixture's base injects the name; the alias keeps the helpers reachable.
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

TEST_F(Regression_CompileOnce, ManyFitCallsCompileTheStepExactlyOnce)
{
    // `sa demo mlp` drives one fit() per epoch so it can print a real
    // per-epoch trajectory. That is a different call pattern from a single
    // fit() with epochs=N, and a recompile hidden inside fit() would be
    // invisible to the integration test, which only calls it once.
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

TEST_F(Regression_CompileOnce, EveryReplayDispatchesRatherThanReportingCompleteImmediately)
{
    // The CompiledStep footgun, seen from the outside: an unarmed plan's run()
    // returns successfully having dispatched nothing, so the parameters never
    // move and the loss goes flat. Replay #2 onward is where that shows,
    // because replay #1 runs off the state the constructor captured.
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

TEST_F(Regression_CompileOnce, TheVeryFirstRunDispatchesRatherThanReportingCompleteOverAnEmptyBaseline)
{
    // The rearm/capture inversion, caught at run #1 rather than at replay #2.
    //
    // The test above notices the footgun only from the second replay onward,
    // because it reads it off a loss that stops changing. But the inversion
    // breaks the FIRST run too: with an empty base state, rearm() writes the
    // completion sentinel's in-degree as zero, the sentinel is ready before
    // anything is dispatched, and execute() returns success having run no
    // kernel at all. Nothing distinguishes that from a step that ran, except
    // the contents of the output buffer.
    //
    // So the buffer is poisoned first with a value the graph cannot produce,
    // and the assertion is that the poison is gone. A run that dispatched
    // nothing leaves it exactly where it was.
    //
    // Verified to have teeth by deleting capture_base_state() from
    // CompiledStep's constructor: this test fails on the first assertion,
    // reporting the poison still in place.
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

    // And every replay after it: re-poison, change the input, and require the
    // output to track the new input rather than the old answer or the poison.
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

TEST_F(Regression_CompileOnce, CompileCountSurvivesEverythingTheLoopDoesBetweenSteps)
{
    // Compile-once is asserted for the straightforward path elsewhere. What is
    // pinned here is that none of the things a real loop does between steps
    // reaches back into the plan: rewriting the batch buffers, interleaving an
    // optimizer that records and flushes its own graph every step, stopping
    // and resuming, and changing a hyperparameter mid-run.
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

    // Two datasets of the same shape but different contents: the compiled plan
    // reads its inputs by address, so swapping the data a step trains on must
    // be a copy into those buffers and nothing more.
    Data::SyntheticOptions other = make_data_options();
    other.seed = 0xFEED'BEEFull;

    const Data::SyntheticClassification first(*engine, make_data_options());
    const Data::SyntheticClassification second(*engine, other);

    ASSERT_EQ(trainer.compile_count(), 1u);

    for (int round = 0; round < 4; ++round)
    {
        // Stop and resume: fit() returns and is entered again, which is the
        // pattern `sa demo mlp` uses to print a per-epoch trajectory.
        const Train::RunReport report = trainer.fit(round % 2 == 0 ? first : second);
        EXPECT_EQ(report.compile_count, 1u) << "recompiled in round " << round;

        // A learning-rate schedule changes a host-side scalar the optimizer
        // reads per step. If that ever became a compiled attribute it would
        // force a recompile - see ir.hpp's warning on Attributes.
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
