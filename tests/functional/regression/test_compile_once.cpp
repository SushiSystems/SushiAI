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
