/**************************************************************************/
/* test_detection_run_check.cpp                                           */
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

/**
 * @file test_detection_run_check.cpp
 * @brief Tests the check of a dataset and a run plan that both detection training loops run.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/vision/training/detection_run_check.hpp>
#include <SushiAI/vision/training/truth_from_roles.hpp>

using namespace SushiAI;
using Vision::Training::check_detection_run;
using Vision::Training::DetectionRunPlan;
using Vision::Training::DetectionRunSubject;
using Vision::Training::make_detection_run_plan;

namespace
{
    /** @brief Holds the images per batch the subject was built for. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the classes the subject holds. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Holds the object slots of one image, M. */
    constexpr std::int64_t K_SLOTS = 2;

    /** @brief Holds the batches of one epoch of the stub below. */
    constexpr std::size_t K_BATCHES = 3;

    /** @brief Describes a dataset by its facts alone; it can be checked, never read. */
    class DescribedDataset final : public Data::Dataset
    {
        public:
            /** @brief Describes @p batches batches of @p batch images over @p targets. */
            DescribedDataset(std::size_t batches, std::int64_t batch,
                             std::vector<Data::TargetDescription> targets)
                : batches_(batches), batch_(batch), targets_(std::move(targets))
            {

            }

            int64_t batch_size() const noexcept override { return batch_; }
            Shape sample_shape() const override { return Shape{1}; }
            const std::vector<Data::TargetDescription>& targets() const noexcept override
            {
                return targets_;
            }
            std::size_t batch_count() const noexcept override { return batches_; }
            std::string_view name() const noexcept override { return "stub"; }

            [[nodiscard]] Status read(std::size_t, Data::Batch&) const override
            {
                return unreadable();
            }

            [[nodiscard]] Status read_rows(std::size_t, Data::RowRange,
                                           Data::Batch&) const override
            {
                return unreadable();
            }

        private:
            /** @brief Returns the failure of a read, which no check makes. */
            static Status unreadable()
            {
                return Status::from_message(StatusCode::Unsupported, []()
                {
                    return std::string("The stub dataset cannot be read");
                });
            }

            /** @brief Holds the batches of one epoch. */
            std::size_t batches_;

            /** @brief Holds the images per batch. */
            std::int64_t batch_;

            /** @brief Holds the roles the stub publishes. */
            std::vector<Data::TargetDescription> targets_;
    };

    /** @brief Returns the three truth roles over K_SLOTS slots and K_CLASSES classes. */
    std::vector<Data::TargetDescription> truth_roles()
    {
        Result<std::vector<Data::TargetDescription>> described =
            Vision::Training::describe_truth_roles(K_SLOTS, K_CLASSES);
        EXPECT_TRUE(described.status().is_ok());
        return std::move(described).value();
    }

    /** @brief Returns the subject of the checks below. */
    DetectionRunSubject make_subject()
    {
        DetectionRunSubject subject;
        subject.run_name = "A test run";
        subject.batch_origin = "heads were declared for";
        subject.batch = K_BATCH;
        subject.classes = K_CLASSES;
        return subject;
    }
} // namespace

/** @brief Tests that a dataset of the subject's batch and roles passes with a default plan. */
TEST(Small_DetectionRunCheck, AcceptsADatasetTheSubjectCanRead)
{
    const DescribedDataset dataset(K_BATCHES, K_BATCH, truth_roles());

    SA_EXPECT_OK(check_detection_run(dataset, make_subject(), DetectionRunPlan{}));
}

/** @brief Tests that the plan copies the four fields of a loop's options. */
TEST(Small_DetectionRunCheck, ThePlanCopiesTheFieldsOfTheOptions)
{
    struct Options
    {
        std::int64_t epochs = 4;
        const Optim::RateSchedule* schedule = nullptr;
        std::int64_t first_epoch = 2;
        std::size_t first_batch = 1;
    };

    const Optim::RateSchedule schedule = Optim::StepSchedule{};
    Options options;
    options.schedule = &schedule;

    const DetectionRunPlan plan = make_detection_run_plan(options);
    EXPECT_EQ(plan.epochs, 4);
    EXPECT_EQ(plan.first_epoch, 2);
    EXPECT_EQ(plan.first_batch, 1u);
    EXPECT_EQ(plan.schedule, &schedule);
}

/** @brief Tests that no epoch and a negative first epoch are refused, naming the run. */
TEST(Small_DetectionRunCheck, EpochsTheRunCannotUseAreRefusedNamingTheRun)
{
    const DescribedDataset dataset(K_BATCHES, K_BATCH, truth_roles());

    DetectionRunPlan no_epoch;
    no_epoch.epochs = 0;
    SA_EXPECT_FAILURE(check_detection_run(dataset, make_subject(), no_epoch),
                      StatusCode::InvalidDescription, "A test run needs at least one epoch",
                      "got 0");

    DetectionRunPlan negative;
    negative.first_epoch = -1;
    SA_EXPECT_FAILURE(check_detection_run(dataset, make_subject(), negative),
                      StatusCode::InvalidDescription, "A test run starts at an epoch index",
                      "got -1");
}

/** @brief Tests that a dataset without batches is refused, naming it. */
TEST(Small_DetectionRunCheck, ADatasetWithoutBatchesIsRefused)
{
    const DescribedDataset dataset(0, K_BATCH, truth_roles());

    SA_EXPECT_FAILURE(check_detection_run(dataset, make_subject(), DetectionRunPlan{}),
                      StatusCode::InvalidDescription, "'stub'", "no batches");
}

/** @brief Tests that a first batch the dataset lacks is refused with the batch count. */
TEST(Small_DetectionRunCheck, AFirstBatchTheDatasetLacksIsRefused)
{
    const DescribedDataset dataset(K_BATCHES, K_BATCH, truth_roles());

    DetectionRunPlan plan;
    plan.first_batch = K_BATCHES;
    SA_EXPECT_FAILURE(check_detection_run(dataset, make_subject(), plan),
                      StatusCode::Incompatible, "batch 3", "'stub'", "3 batches");
}

/** @brief Tests that another batch size is refused with both sizes and the origin. */
TEST(Small_DetectionRunCheck, ABatchOfAnotherSizeNamesBothSizesAndTheOrigin)
{
    const DescribedDataset dataset(K_BATCHES, K_BATCH + 1, truth_roles());

    SA_EXPECT_FAILURE(check_detection_run(dataset, make_subject(), DetectionRunPlan{}),
                      StatusCode::Incompatible, "'stub'", "batch of 3",
                      "heads were declared for 2");
}

/** @brief Tests that a schedule the optimizer cannot follow is refused with its own reason. */
TEST(Small_DetectionRunCheck, AnInvalidScheduleIsRefused)
{
    const DescribedDataset dataset(K_BATCHES, K_BATCH, truth_roles());

    Optim::StepSchedule steps;
    steps.base_rate = 0.1;
    steps.steps = {2, 5};
    steps.scales = {0.5};
    const Optim::RateSchedule schedule = steps;

    DetectionRunPlan plan;
    plan.schedule = &schedule;
    SA_EXPECT_FAILURE(check_detection_run(dataset, make_subject(), plan),
                      StatusCode::InvalidDescription, "2 steps", "1 scales");
}

/** @brief Tests that a dataset without a boxes role of [M, 4] is refused. */
TEST(Small_DetectionRunCheck, ADatasetWithoutTheBoxesRoleIsRefused)
{
    const DescribedDataset dataset(
        K_BATCHES, K_BATCH,
        {Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32,
                                 K_CLASSES}});

    SA_EXPECT_FAILURE(check_detection_run(dataset, make_subject(), DetectionRunPlan{}),
                      StatusCode::InvalidDescription, "'stub'", "'boxes'");
}

/** @brief Tests that roles over another class count are refused by the role match. */
TEST(Small_DetectionRunCheck, RolesOverAnotherClassCountAreRefused)
{
    DetectionRunSubject subject = make_subject();
    subject.classes = K_CLASSES + 1;
    const DescribedDataset dataset(K_BATCHES, K_BATCH, truth_roles());

    EXPECT_FALSE(check_detection_run(dataset, subject, DetectionRunPlan{}).is_ok());
}
