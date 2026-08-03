/**************************************************************************/
/* test_learning_curve.cpp                                                */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// The test that catches "everything passes and the model stops training".
//
// Every other test here checks a property: this gradient matches a finite
// difference, that step compiled once, this loss is finite. A model can
// satisfy all of them and still learn measurably worse than it did last month
// - a gradient scaled by the wrong constant, a momentum buffer applied a step
// late, an initialisation gain quietly changed. Each of those leaves the loss
// going down, just not the same way, and no property test has an opinion about
// "the same way".
//
// So this one pins the trajectory itself against a reference recorded from a
// real run of this code. The tolerance is the whole design of the test:
//
//  - Too tight and it fails on every legitimate kernel change. Reassociating a
//    gemm's inner sum moves an fp32 result by an ULP or two, and twelve steps
//    of momentum amplify that; a tolerance below about 1e-3 would be noise.
//  - Too loose and it stops distinguishing training from not-training. A
//    gradient wrong by a constant factor, or a backward rule reading the
//    forward output where it needed the input, moves this trajectory by
//    tenths.
//
// LOSS_TOLERANCE sits two orders of magnitude between those, and the shape
// assertions below it - the exact ln(C) start, the monotone-enough descent,
// the final value - hold even if a future toolchain widens the noise floor.
//
// Re-recording the reference is legitimate ONLY when the change that moved it
// is understood and intended. If it moved and nobody knows why, that is the
// finding, not the tolerance.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
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
    class Regression_LearningCurve : public ::testing::Test {};

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t BATCH = 8;
    constexpr int64_t FEATURES = 6;
    constexpr int64_t HIDDEN = 12;
    constexpr int64_t CLASSES = 4;
    constexpr int64_t BATCHES = 4;
    constexpr int64_t EPOCHS = 4;
    constexpr double LEARNING_RATE = 0.25;
    constexpr double MOMENTUM = 0.9;
    constexpr std::uint64_t DATA_SEED = 0x0C0FFEE'5EEDull;

    /** @brief Everything the reference depends on, in one place. */
    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = BATCH;
        options.batches = BATCHES;
        options.features = FEATURES;
        options.classes = CLASSES;
        options.seed = DATA_SEED;
        return options;
    }

    /**
     * @brief Runs the pinned configuration and returns its per-step losses.
     *
     * A fresh Engine, because reproducibility is a property of one - see
     * dataset.hpp on the RNG offset counter.
     *
     * @return One loss per step, EPOCHS * BATCHES of them.
     */
    std::vector<double> recorded_trajectory()
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});
        Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

        TensorPool pool(engine);
        std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
        NN::initialize(engine, model, pool);

        Optim::SGDOptions sgd;
        sgd.learning_rate = LEARNING_RATE;
        sgd.momentum = MOMENTUM;
        Optim::SGD optimizer(engine, slots, sgd);

        Train::TrainerOptions options;
        options.track_accuracy = false;
        Train::Trainer trainer(engine, built.graph, pool, built.spec, optimizer, options);

        const Data::SyntheticClassification dataset(engine, data_options());

        std::vector<double> losses;
        losses.reserve(static_cast<std::size_t>(EPOCHS * BATCHES));

        for (int64_t epoch = 0; epoch < EPOCHS; ++epoch)
            for (std::size_t batch = 0; batch < dataset.batch_count(); ++batch)
                losses.push_back(trainer.run_step(dataset.batch(batch)).loss);

        return losses;
    }

    /** @brief Renders a trajectory as a paste-ready initialiser, for re-recording. */
    std::string as_initialiser(const std::vector<double>& losses)
    {
        std::string text = "{\n";
        for (std::size_t i = 0; i < losses.size(); ++i)
        {
            char buffer[64] = {};
            std::snprintf(buffer, sizeof(buffer), "%.6f, ", losses[i]);
            if (i % 4 == 0) text += "        ";
            text += buffer;
            if (i % 4 == 3) text += "\n";
        }
        return text + "    }";
    }

    /**
     * @brief The reference, recorded from a run of this configuration.
     *
     * Recorded on 2026-08-03 against SushiBLAS at the storage_offset/RNG/scalar
     * -precision revision current that day. Every value is the batch's mean
     * cross-entropy as the trainer reports it.
     */
    constexpr double REFERENCE[] = {
        1.386294, 1.262154, 1.037548, 0.767985,
        0.570578, 0.373737, 0.274389, 0.152003,
        0.098416, 0.070900, 0.122845, 0.031372,
        0.029982, 0.011396, 0.055688, 0.005311,
    };

    /** @brief How far a step's loss may drift before it counts as a change. */
    constexpr double LOSS_TOLERANCE = 0.02;
}

TEST_F(Regression_LearningCurve, TheLossTrajectoryMatchesTheRecordedReference)
{
    const std::vector<double> losses = recorded_trajectory();

    ASSERT_EQ(losses.size(), std::size(REFERENCE))
        << "the configuration changed shape; re-record the reference:\n"
        << as_initialiser(losses);

    bool matched = true;
    for (std::size_t step = 0; step < losses.size(); ++step)
    {
        EXPECT_NEAR(losses[step], REFERENCE[step], LOSS_TOLERANCE) << "at step " << step;
        if (std::abs(losses[step] - REFERENCE[step]) > LOSS_TOLERANCE) matched = false;
    }

    if (!matched)
        ADD_FAILURE() << "the model no longer learns the way it did. Understand WHY before "
                         "re-recording; the observed trajectory was:\n"
                      << as_initialiser(losses);
}

TEST_F(Regression_LearningCurve, TheTrajectoryHasTheShapeALearningModelHas)
{
    // The shape assertions, stated independently of the recorded numbers so
    // they survive a legitimate re-record. Together they say: it starts where
    // a zeroed classifier must start, it goes down, and it gets most of the
    // way there.
    const std::vector<double> losses = recorded_trajectory();
    ASSERT_FALSE(losses.empty());

    EXPECT_NEAR(losses.front(), std::log(static_cast<double>(CLASSES)), 1e-4)
        << "the zeroed classifier's first loss is ln(C) exactly, or something initialised it";

    for (std::size_t step = 0; step < losses.size(); ++step)
        ASSERT_TRUE(std::isfinite(losses[step])) << "at step " << step;

    EXPECT_LT(losses.back(), losses.front() * 0.25)
        << "the loss barely moved over " << losses.size() << " steps";

    // Not monotone step by step - each step sees a different batch - but each
    // epoch's mean must beat the one before it.
    double previous_epoch = 0.0;
    for (int64_t epoch = 0; epoch < EPOCHS; ++epoch)
    {
        double total = 0.0;
        for (int64_t batch = 0; batch < BATCHES; ++batch)
            total += losses[static_cast<std::size_t>(epoch * BATCHES + batch)];

        const double mean = total / static_cast<double>(BATCHES);
        if (epoch > 0)
            EXPECT_LT(mean, previous_epoch) << "epoch " << epoch << " did not improve";
        previous_epoch = mean;
    }
}
