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
 * @see tests/regression/README.md
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <common/classifier_objective.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/regression/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;

namespace
{
    class Regression_LearningCurve : public ::testing::Test {};

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

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
     * @brief Runs the pinned configuration on a fresh Engine and returns its per-step losses.
     * @return One loss per step, K_EPOCHS * K_BATCHES of them, or the first failure.
     * @see tests/regression/README.md
     */
    [[nodiscard]] Result<std::vector<double>> recorded_trajectory()
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::context());

        MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        TensorPool pool(engine);
        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(built, state, pool));
        SA_TRY(NN::initialize(engine, state));

        Optim::SGDOptions sgd;
        sgd.learning_rate = K_LEARNING_RATE;
        sgd.momentum = K_MOMENTUM;
        SA_TRY_ASSIGN(std::unique_ptr<Optim::SGD> optimizer, Optim::SGD::make(engine, slots, sgd));

        Train::TrainerOptions options;
        options.track_metrics = false;
        const auto objective = SaTest::classifier_objective(built);
        SA_TRY_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(engine, built, pool, objective, *optimizer,
                                           options));

        SA_TRY_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                      Data::SyntheticClassification::make(engine, data_options()));

        std::vector<double> losses;
        losses.reserve(static_cast<std::size_t>(K_EPOCHS * K_BATCHES));

        for (int64_t epoch = 0; epoch < K_EPOCHS; ++epoch)
            for (std::size_t batch = 0; batch < dataset->batch_count(); ++batch)
            {
                Data::Batch dataset_batch;
                SA_TRY(dataset->read(batch, dataset_batch));
                SA_TRY_ASSIGN(const Train::StepMetrics metrics,
                              trainer->run_step(dataset_batch));
                losses.push_back(metrics.loss);
            }

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
            if (i % 4 == 0)
                text += "        ";

            text += buffer;
            if (i % 4 == 3)
                text += "\n";
        }
        return text + "    }";
    }

    /**
     * @brief The reference, recorded from a run of this configuration.
     *
     * Recorded on 2026-10-09 against SushiBLAS 266eeb7, whose generator draws four
     * values per Philox block. Every value is the batch's mean
     * cross-entropy as the trainer reports it.
     */
    constexpr double K_REFERENCE[] = {
        1.386294, 0.991878, 0.594361, 0.382508,
        0.237015, 0.101345, 0.015991, 0.016213,
        0.013401, 0.005473, 0.001037, 0.004709,
        0.000865, 0.000353, 0.000528, 0.002215,
    };

    /** @brief Steps from this index on are compared relatively; earlier steps absolutely. */
    constexpr std::size_t K_TAIL_FIRST_STEP = 6;

    /** @brief Absolute drift a head step's loss may show before it counts as a change. */
    constexpr double K_LOSS_TOLERANCE = 0.02;

    /**
     * @brief Relative drift a tail step's loss may show, as a fraction of its reference.
     *
     * Two per cent pins each tail value to about its second significant digit.
     */
    constexpr double K_LOSS_RELATIVE_TOLERANCE = 0.02;

    /**
     * @brief Tells whether one step's loss matches its reference within that step's tolerance.
     *
     * Head steps pass within K_LOSS_TOLERANCE; tail steps pass within K_LOSS_RELATIVE_TOLERANCE.
     */
    bool within_tolerance(std::size_t step, double observed, double reference)
    {
        const double drift = std::abs(observed - reference);
        if (step < K_TAIL_FIRST_STEP)
            return drift <= K_LOSS_TOLERANCE;

        return drift <= K_LOSS_RELATIVE_TOLERANCE * std::abs(reference);
    }
}

/**
 * @brief Tests that every step's loss is within its tolerance of the recorded reference.
 * @see tests/regression/README.md
 */
TEST_F(Regression_LearningCurve, TheLossTrajectoryMatchesTheRecordedReference)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> losses, recorded_trajectory());

    ASSERT_EQ(losses.size(), std::size(K_REFERENCE))
        << "the configuration changed shape; re-record the reference:\n"
        << as_initialiser(losses);

    bool matched = true;
    for (std::size_t step = 0; step < losses.size(); ++step)
    {
        const bool within = within_tolerance(step, losses[step], K_REFERENCE[step]);
        EXPECT_TRUE(within) << "at step " << step << ": observed " << losses[step]
                            << ", reference " << K_REFERENCE[step];
        if (!within)
            matched = false;
    }

    if (!matched)
        ADD_FAILURE() << "the model no longer learns the way it did. Understand WHY before "
                         "re-recording; the observed trajectory was:\n"
                      << as_initialiser(losses);
}

/**
 * @brief Tests that the trajectory starts at ln(C), stays finite and improves every epoch.
 * @see tests/regression/README.md
 */
TEST_F(Regression_LearningCurve, TheTrajectoryHasTheShapeALearningModelHas)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> losses, recorded_trajectory());
    ASSERT_FALSE(losses.empty());

    EXPECT_NEAR(losses.front(), std::log(static_cast<double>(K_CLASSES)), 1e-4)
        << "the zeroed classifier's first loss is ln(C) exactly, or something initialised it";

    for (std::size_t step = 0; step < losses.size(); ++step)
        ASSERT_TRUE(std::isfinite(losses[step])) << "at step " << step;

    EXPECT_LT(losses.back(), losses.front() * 0.25)
        << "the loss barely moved over " << losses.size() << " steps";

    // Each epoch's mean must beat the one before it; see tests/regression/README.md.
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
