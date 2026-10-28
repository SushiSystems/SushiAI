/**************************************************************************/
/* test_free_training_loop.cpp                                            */
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
 * @file test_free_training_loop.cpp
 * @brief Tests the loop that drives an anchor-free training step over a dataset.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/training/free_branch_weights.hpp>
#include <SushiAI/vision/training/free_training_loop.hpp>
#include <SushiAI/vision/training/truth_from_roles.hpp>

#include "detection_loop_support.hpp"
#include "free_two_level_case.hpp"

using namespace SushiAI;
using namespace SushiAI::DetectionLoopTest;
using namespace SushiAI::FreeTwoLevelTest;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;
using Vision::Training::FreeLoopOptions;
using Vision::Training::FreeProgress;
using Vision::Training::FreeRunReport;
using Vision::Training::run_free_training;

namespace
{
    /** @brief Holds the object slots of one image, M. */
    constexpr std::int64_t K_SLOTS = 2;

    /** @brief Holds the batches of one epoch of the dataset. */
    constexpr std::size_t K_BATCHES = 3;

    /** @brief Bounds the error of a rate that is one product of two doubles. */
    constexpr double K_RATE_TOLERANCE = 1e-15;

    /** @brief Returns a rate of 0.1 that halves from update 2 and again from update 5. */
    Optim::StepSchedule make_halving_schedule()
    {
        Optim::StepSchedule schedule;
        schedule.base_rate = 0.1;
        schedule.steps = {2, 5};
        schedule.scales = {0.5, 0.5};
        return schedule;
    }

    /** @brief Provides an engine, a two-level model with its step, and datasets of truths. */
    class Medium_FreeTrainingLoop : public SaTest::EngineTest
    {
        protected:
            /** @brief Builds the model, both graphs, the optimizer and the step. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                model_.prepare(*engine);
            }

            /** @brief Releases what holds tensors of the engine before the engine goes. */
            void TearDown() override
            {
                model_.release();
                SaTest::EngineTest::TearDown();
            }

            /** @brief Returns a dataset of @p batches batches of @p rows seeded images each. */
            Data::MaterialisedDataset make_dataset(
                std::size_t batches, std::int64_t rows,
                const std::vector<Data::TargetDescription>& targets)
            {
                return make_truth_dataset(*engine, batches, rows, targets, K_NETWORK_SIDE,
                                          K_INPUT_SEED, K_BATCH);
            }

            /** @brief Returns the three truth roles of K_SLOTS slots over K_CLASSES classes. */
            static std::vector<Data::TargetDescription> truth_roles()
            {
                Result<std::vector<Data::TargetDescription>> described =
                    Vision::Training::describe_truth_roles(K_SLOTS, K_CLASSES);
                EXPECT_TRUE(described.status().is_ok());
                return std::move(described).value();
            }

            /** @brief Holds the two-level model and the step over it. */
            TwoLevelCase model_;
    };
} // namespace

/** @brief Tests that each update takes the schedule's rate at its number, across epochs. */
TEST_F(Medium_FreeTrainingLoop, EachUpdateTakesTheRateOfItsIteration)
{
    ASSERT_TRUE(model_.ready);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());
    const Optim::RateSchedule schedule = make_halving_schedule();

    std::vector<FreeProgress> seen;
    FreeLoopOptions options;
    options.epochs = 2;
    options.schedule = &schedule;
    options.on_update = [&seen](const FreeProgress& progress) { seen.push_back(progress); };

    SA_ASSERT_OK_AND_ASSIGN(const FreeRunReport report,
                            run_free_training(*model_.step, dataset, options));

    const double rates[] = {0.1, 0.05, 0.05, 0.05, 0.025, 0.025};
    ASSERT_EQ(seen.size(), 6u);
    for (std::size_t update = 0; update < seen.size(); ++update)
    {
        EXPECT_EQ(seen[update].epoch, static_cast<std::int64_t>(update / K_BATCHES) + 1);
        EXPECT_EQ(seen[update].batch, update % K_BATCHES);
        EXPECT_EQ(seen[update].iteration, static_cast<std::int64_t>(update) + 1);
        EXPECT_NEAR(seen[update].rate, rates[update], K_RATE_TOLERANCE);
    }

    EXPECT_EQ(report.updates, 6u);
    EXPECT_NEAR(report.last_rate, 0.025, K_RATE_TOLERANCE);
    EXPECT_NEAR(model_.optimizer->learning_rate(), 0.025, K_RATE_TOLERANCE);
    EXPECT_EQ(model_.optimizer->step_count(), 6u);
    EXPECT_EQ(model_.step->run_count(), 6u);
}

/** @brief Tests that the report holds the loss terms of the first and the last update. */
TEST_F(Medium_FreeTrainingLoop, TheReportHoldsTheTermsOfTheFirstAndTheLastUpdate)
{
    ASSERT_TRUE(model_.ready);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    std::vector<FreeProgress> seen;
    FreeLoopOptions options;
    options.on_update = [&seen](const FreeProgress& progress) { seen.push_back(progress); };

    SA_ASSERT_OK_AND_ASSIGN(const FreeRunReport report,
                            run_free_training(*model_.step, dataset, options));

    ASSERT_EQ(seen.size(), K_BATCHES);
    EXPECT_EQ(report.updates, K_BATCHES);
    EXPECT_EQ(report.first.total, seen.front().terms.total);
    EXPECT_EQ(report.first.box, seen.front().terms.box);
    EXPECT_EQ(report.last.total, seen.back().terms.total);
    EXPECT_EQ(report.last.classes, seen.back().terms.classes);
    for (const FreeProgress& progress : seen)
    {
        EXPECT_TRUE(std::isfinite(progress.terms.total));
        EXPECT_GT(progress.terms.total, 0.0);
    }
}

/** @brief Tests that a cosine schedule drives the rate of each update of a run. */
TEST_F(Medium_FreeTrainingLoop, ACosineScheduleSetsTheRateOfEachUpdate)
{
    ASSERT_TRUE(model_.ready);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    Optim::CosineSchedule cosine;
    cosine.base_rate = 0.1;
    cosine.total_iterations = 6;
    cosine.final_rate = 0.01;
    const Optim::RateSchedule schedule = cosine;

    std::vector<FreeProgress> seen;
    FreeLoopOptions options;
    options.epochs = 2;
    options.schedule = &schedule;
    options.on_update = [&seen](const FreeProgress& progress) { seen.push_back(progress); };

    SA_ASSERT_OK_AND_ASSIGN(const FreeRunReport report,
                            run_free_training(*model_.step, dataset, options));

    // A double holds the schedule's own rate, so the two are compared exactly.
    ASSERT_EQ(seen.size(), 6u);
    for (std::size_t update = 0; update < seen.size(); ++update)
    {
        const std::int64_t iteration = static_cast<std::int64_t>(update) + 1;
        EXPECT_DOUBLE_EQ(seen[update].rate, cosine.rate_at(iteration)) << "update " << update;
    }
    EXPECT_DOUBLE_EQ(report.last_rate, cosine.rate_at(6));
}

/** @brief Tests that a run without a schedule leaves the optimizer's rate as it was made. */
TEST_F(Medium_FreeTrainingLoop, ARunWithoutAScheduleKeepsTheOptimizersRate)
{
    ASSERT_TRUE(model_.ready);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    SA_ASSERT_OK_AND_ASSIGN(const FreeRunReport report,
                            run_free_training(*model_.step, dataset));

    EXPECT_EQ(report.updates, K_BATCHES);
    EXPECT_NEAR(report.last_rate, K_LEARNING_RATE, K_RATE_TOLERANCE);
    EXPECT_NEAR(model_.optimizer->learning_rate(), K_LEARNING_RATE, K_RATE_TOLERANCE);
}

/** @brief Tests that a dataset the step cannot read is refused before any update. */
TEST_F(Medium_FreeTrainingLoop, ADatasetTheStepCannotReadIsRefusedBeforeAnyUpdate)
{
    ASSERT_TRUE(model_.ready);

    const Data::MaterialisedDataset other_batch =
        make_dataset(K_BATCHES, K_BATCH + 1, truth_roles());
    SA_EXPECT_FAILURE(run_free_training(*model_.step, other_batch), StatusCode::Incompatible,
                      "'truths'", "batch of 3", "ports were traced for 2");

    const Data::MaterialisedDataset empty = make_dataset(0, K_BATCH, truth_roles());
    SA_EXPECT_FAILURE(run_free_training(*model_.step, empty), StatusCode::InvalidDescription,
                      "'truths'", "no batches");

    const Data::MaterialisedDataset labelled = make_dataset(
        K_BATCHES, K_BATCH,
        {Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32,
                                 K_CLASSES}});
    SA_EXPECT_FAILURE(run_free_training(*model_.step, labelled), StatusCode::InvalidDescription,
                      "'truths'", "'boxes'");

    EXPECT_EQ(model_.optimizer->step_count(), 0u);
}

/** @brief Tests that the options of a run are checked: the epochs and the schedule. */
TEST_F(Medium_FreeTrainingLoop, OptionsARunCannotUseAreRefused)
{
    ASSERT_TRUE(model_.ready);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    FreeLoopOptions no_epoch;
    no_epoch.epochs = 0;
    SA_EXPECT_FAILURE(run_free_training(*model_.step, dataset, no_epoch),
                      StatusCode::InvalidDescription, "An anchor-free run", "at least one epoch");

    Optim::StepSchedule uneven_steps = make_halving_schedule();
    uneven_steps.scales.pop_back();
    const Optim::RateSchedule uneven = uneven_steps;
    FreeLoopOptions bad_schedule;
    bad_schedule.schedule = &uneven;
    SA_EXPECT_FAILURE(run_free_training(*model_.step, dataset, bad_schedule),
                      StatusCode::InvalidDescription, "2 steps", "1 scales");

    EXPECT_EQ(model_.optimizer->step_count(), 0u);
}

/** @brief Tests that a run goes on at the epoch index and the batch its options state. */
TEST_F(Medium_FreeTrainingLoop, ARunStartsAtTheEpochAndTheBatchItsOptionsState)
{
    ASSERT_TRUE(model_.ready);
    const Data::MaterialisedDataset held = make_dataset(K_BATCHES, K_BATCH, truth_roles());
    const EpochRecorder dataset(held);

    std::vector<FreeProgress> seen;
    FreeLoopOptions options;
    options.epochs = 2;
    options.first_epoch = 3;
    options.first_batch = 2;
    options.on_update = [&seen](const FreeProgress& progress) { seen.push_back(progress); };

    SA_ASSERT_OK_AND_ASSIGN(const FreeRunReport report,
                            run_free_training(*model_.step, dataset, options));

    // The first pass reads its last batch alone; the second reads all three.
    EXPECT_EQ(report.updates, 1u + K_BATCHES);
    EXPECT_EQ(dataset.epochs(), (std::vector<std::int64_t>{3, 4}));
    ASSERT_EQ(seen.size(), 1u + K_BATCHES);
    EXPECT_EQ(seen[0].epoch, 1);
    EXPECT_EQ(seen[0].batch, 2u);
    EXPECT_EQ(seen[1].epoch, 2);
    EXPECT_EQ(seen[1].batch, 0u);
}

/** @brief Tests that a start the run cannot take is refused before any update. */
TEST_F(Medium_FreeTrainingLoop, AStartTheDatasetLacksIsRefused)
{
    ASSERT_TRUE(model_.ready);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    FreeLoopOptions negative;
    negative.first_epoch = -1;
    SA_EXPECT_FAILURE(run_free_training(*model_.step, dataset, negative),
                      StatusCode::InvalidDescription, "epoch index", "-1");

    FreeLoopOptions past_the_end;
    past_the_end.first_batch = K_BATCHES;
    SA_EXPECT_FAILURE(run_free_training(*model_.step, dataset, past_the_end),
                      StatusCode::Incompatible, "batch 3", "'truths'", "3 batches");

    EXPECT_EQ(model_.optimizer->step_count(), 0u);
}

/** @brief Tests that the one-to-many weight falls over the epochs of the run it is given. */
TEST_F(Medium_FreeTrainingLoop, TheOneToManyWeightFallsOverTheEpochsOfTheRun)
{
    ASSERT_TRUE(model_.ready);
    const Data::MaterialisedDataset dataset = make_dataset(1, K_BATCH, truth_roles());
    const Vision::Training::FreeBranchWeights weights = model_.objective->branch_weights();

    std::vector<float> seen;
    FreeLoopOptions options;
    options.epochs = 3;
    options.on_update = [this, &seen](const FreeProgress&)
    {
        seen.push_back(model_.step->one_to_many_weight());
    };

    SA_EXPECT_OK(run_free_training(*model_.step, dataset, options));

    // The weight moves linearly from the start at epoch 0 to the end at the last epoch.
    ASSERT_EQ(seen.size(), 3u);
    EXPECT_FLOAT_EQ(seen[0], weights.start);
    EXPECT_FLOAT_EQ(seen[1], (weights.start + weights.end) / 2.0f);
    EXPECT_FLOAT_EQ(seen[2], weights.end);
}

/** @brief Tests that a restored run count is the count of both parts, and counts on. */
TEST_F(Medium_FreeTrainingLoop, ARestoredRunCountCountsOn)
{
    ASSERT_TRUE(model_.ready);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    EXPECT_EQ(model_.step->run_count(), 0u);
    SA_EXPECT_OK(model_.step->restore_run_count(4));
    EXPECT_EQ(model_.step->run_count(), 4u);

    SA_EXPECT_OK(run_free_training(*model_.step, dataset));

    EXPECT_EQ(model_.step->run_count(), 4u + K_BATCHES);
}

/** @brief Tests that a callback that throws ends the run as a failure after its update. */
TEST_F(Medium_FreeTrainingLoop, ACallbackThatThrowsEndsTheRunAsAFailure)
{
    ASSERT_TRUE(model_.ready);
    const Data::MaterialisedDataset dataset = make_dataset(K_BATCHES, K_BATCH, truth_roles());

    FreeLoopOptions options;
    options.on_update = [](const FreeProgress&) { throw std::runtime_error("stop here"); };

    const Result<FreeRunReport> report = run_free_training(*model_.step, dataset, options);

    EXPECT_FALSE(report.status().is_ok());
    EXPECT_NE(report.status().message().find("stop here"), std::string::npos);
    EXPECT_EQ(model_.optimizer->step_count(), 1u);
}
