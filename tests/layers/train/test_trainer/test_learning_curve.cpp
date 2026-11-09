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
 * @file test_learning_curve.cpp
 * @brief Tests that the per-step loss trajectory matches a recorded reference and keeps its shape.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;
using Graph::TensorPool;

namespace
{
    class Medium_LearningCurve : public ::testing::Test {};

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
     * @see tests/layers/train/README.md
     */
    [[nodiscard]] Result<std::vector<double>> recorded_trajectory()
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
        SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

        MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});
        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        TensorPool pool(engine);
        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(backend, built, state, pool));
        SA_TRY(NN::initialize(backend, state));

        Optim::SGDOptions sgd;
        sgd.learning_rate = K_LEARNING_RATE;
        sgd.momentum = K_MOMENTUM;
        SA_TRY_ASSIGN(std::unique_ptr<Optim::SGD> optimizer, Optim::SGD::make(backend, slots, sgd));

        Train::TrainerOptions options;
        options.track_metrics = false;
        const auto objective = SaTest::classifier_objective(built);
        SA_TRY_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(backend, built, pool, objective, *optimizer,
                                           options));

        SA_TRY_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                      Data::SyntheticClassification::make(backend, data_options()));

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

    /**
     * @brief Counts the roundings on the longest path to the first loss in float: the
     *        log-softmax of four classes rounds 8, the product with the target once, the sum of
     *        the 32 products 31 times at worst and the mean once.
     */
    constexpr std::size_t K_FIRST_LOSS_OPERATIONS = 41;

    /** @brief The fraction of its first loss the last loss of the run must stay under. */
    constexpr double K_FALL_FRACTION = 0.25;

    /** @brief Returns the mean of the K_BATCHES losses of epoch @p epoch. */
    double epoch_mean(const std::vector<double>& losses, int64_t epoch)
    {
        const auto first = losses.begin() + epoch * K_BATCHES;
        return std::accumulate(first, first + K_BATCHES, 0.0) / static_cast<double>(K_BATCHES);
    }
}

/** @brief Tests that the zeroed classifier's first loss is ln(C), every class equally likely. */
TEST_F(Medium_LearningCurve, TheFirstLossIsLogCOfAZeroedClassifier)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> losses, recorded_trajectory());
    ASSERT_FALSE(losses.empty());

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
        losses.front(), std::log(static_cast<double>(K_CLASSES)), K_FIRST_LOSS_OPERATIONS)));
}

/** @brief Tests that the loss ends under a quarter of its first value. */
TEST_F(Medium_LearningCurve, TheLossFallsToAQuarterOfItsStart)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> losses, recorded_trajectory());
    ASSERT_FALSE(losses.empty());

    EXPECT_LT(losses.back(), losses.front() * K_FALL_FRACTION);
}

/** @brief Tests that the mean loss of each epoch beats the one before it. */
TEST_F(Medium_LearningCurve, EachEpochImprovesOnTheOneBefore)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> losses, recorded_trajectory());
    ASSERT_EQ(losses.size(), static_cast<std::size_t>(K_EPOCHS * K_BATCHES));

    EXPECT_LT(epoch_mean(losses, 1), epoch_mean(losses, 0));
    EXPECT_LT(epoch_mean(losses, 2), epoch_mean(losses, 1));
    EXPECT_LT(epoch_mean(losses, 3), epoch_mean(losses, 2));
}
