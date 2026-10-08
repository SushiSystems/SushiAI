/**************************************************************************/
/* test_learning_curve.cpp                                                */
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
 * @file test_learning_curve.cpp
 * @brief Tests that the per-step loss trajectory matches a recorded reference and keeps its shape.
 * @author Mustafa Garip
 * @see tests/functional/regression/README.md
 */

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

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/functional/regression/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;

namespace
{
    class Regression_LearningCurve : public ::testing::Test {};

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 8;
    constexpr int64_t K_FEATURES = 6;
    constexpr int64_t K_HIDDEN = 12;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 4;
    constexpr int64_t K_EPOCHS = 4;
    constexpr double K_LEARNING_RATE = 0.25;
    constexpr double K_MOMENTUM = 0.9;
    constexpr std::uint64_t K_DATA_SEED = 0x0C0FFEE'5EEDull;

    /** @brief Everything the reference depends on, in one place. */
    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = K_BATCH;
        options.batches = K_BATCHES;
        options.features = K_FEATURES;
        options.classes = K_CLASSES;
        options.seed = K_DATA_SEED;
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

        Mlp model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
        Train::TrainingGraph built = Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES);

        TensorPool pool(engine);
        std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
        NN::initialize(engine, model, pool);

        Optim::SGDOptions sgd;
        sgd.learning_rate = K_LEARNING_RATE;
        sgd.momentum = K_MOMENTUM;
        Optim::SGD optimizer(engine, slots, sgd);

        Train::TrainerOptions options;
        options.track_accuracy = false;
        Train::Trainer trainer(engine, built.graph, pool, built.spec, optimizer, options);

        const Data::SyntheticClassification dataset(engine, data_options());

        std::vector<double> losses;
        losses.reserve(static_cast<std::size_t>(K_EPOCHS * K_BATCHES));

        for (int64_t epoch = 0; epoch < K_EPOCHS; ++epoch)
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
    constexpr double K_REFERENCE[] = {
        1.386294, 1.262154, 1.037548, 0.767985,
        0.570578, 0.373737, 0.274389, 0.152003,
        0.098416, 0.070900, 0.122845, 0.031372,
        0.029982, 0.011396, 0.055688, 0.005311,
    };

    /** @brief How far a step's loss may drift before it counts as a change. */
    constexpr double K_LOSS_TOLERANCE = 0.02;
}

TEST_F(Regression_LearningCurve, TheLossTrajectoryMatchesTheRecordedReference)
{
    const std::vector<double> losses = recorded_trajectory();

    ASSERT_EQ(losses.size(), std::size(K_REFERENCE))
        << "the configuration changed shape; re-record the reference:\n"
        << as_initialiser(losses);

    bool matched = true;
    for (std::size_t step = 0; step < losses.size(); ++step)
    {
        EXPECT_NEAR(losses[step], K_REFERENCE[step], K_LOSS_TOLERANCE) << "at step " << step;
        if (std::abs(losses[step] - K_REFERENCE[step]) > K_LOSS_TOLERANCE) matched = false;
    }

    if (!matched)
        ADD_FAILURE() << "the model no longer learns the way it did. Understand WHY before "
                         "re-recording; the observed trajectory was:\n"
                      << as_initialiser(losses);
}

/**
 * @brief Tests that the trajectory starts at ln(C), stays finite and improves every epoch.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_LearningCurve, TheTrajectoryHasTheShapeALearningModelHas)
{
    const std::vector<double> losses = recorded_trajectory();
    ASSERT_FALSE(losses.empty());

    EXPECT_NEAR(losses.front(), std::log(static_cast<double>(K_CLASSES)), 1e-4)
        << "the zeroed classifier's first loss is ln(C) exactly, or something initialised it";

    for (std::size_t step = 0; step < losses.size(); ++step)
        ASSERT_TRUE(std::isfinite(losses[step])) << "at step " << step;

    EXPECT_LT(losses.back(), losses.front() * 0.25)
        << "the loss barely moved over " << losses.size() << " steps";

    // Each epoch's mean must beat the one before it; see tests/functional/regression/README.md.
    double previous_epoch = 0.0;
    for (int64_t epoch = 0; epoch < K_EPOCHS; ++epoch)
    {
        double total = 0.0;
        for (int64_t batch = 0; batch < K_BATCHES; ++batch)
            total += losses[static_cast<std::size_t>(epoch * K_BATCHES + batch)];

        const double mean = total / static_cast<double>(K_BATCHES);
        if (epoch > 0)
            EXPECT_LT(mean, previous_epoch) << "epoch " << epoch << " did not improve";
        previous_epoch = mean;
    }
}
