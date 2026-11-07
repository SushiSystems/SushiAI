/**************************************************************************/
/* test_epoch_loop.cpp                                                    */
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
 * @file test_epoch_loop.cpp
 * @brief Tests run_epoch_loop, the loop the trainer run and the cluster run share.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include "epoch_loop.hpp"

using namespace SushiAI;

namespace
{
    /** @brief Holds the batches of one epoch. */
    constexpr std::size_t K_BATCHES = 3;

    /** @brief Returns the plan of @p epochs epochs that starts at @p epoch and @p batch. */
    Apps::EpochPlan plan_of(std::int64_t epochs, std::int64_t epoch = 0, std::size_t batch = 0)
    {
        Apps::EpochPlan plan;
        plan.epochs = epochs;
        plan.first_epoch = epoch;
        plan.first_batch = batch;
        plan.batches = K_BATCHES;
        return plan;
    }
} // namespace

/** @brief Tests the epochs a progress line is printed for. */
TEST(Small_EpochLoop, ReportsTheFirstEveryFifthAndTheLastEpoch)
{
    EXPECT_TRUE(Apps::is_reported_epoch(1, 12));
    EXPECT_FALSE(Apps::is_reported_epoch(2, 12));
    EXPECT_TRUE(Apps::is_reported_epoch(5, 12));
    EXPECT_TRUE(Apps::is_reported_epoch(10, 12));
    EXPECT_FALSE(Apps::is_reported_epoch(11, 12));
    EXPECT_TRUE(Apps::is_reported_epoch(12, 12));
}

/** @brief Tests that a resumed run starts at its batch and epoch, then reads every batch. */
TEST(Small_EpochLoop, StartsAtTheSavedBatchAndEntersEachEpoch)
{
    std::vector<std::int64_t> entered;
    std::vector<std::size_t> stepped;
    std::vector<std::int64_t> reported;
    Apps::EpochSteps steps;
    steps.enter_epoch = [&entered](std::int64_t epoch_index)
    {
        entered.push_back(epoch_index);
    };
    steps.run_step = [&stepped](std::size_t index)
    {
        stepped.push_back(index);
        return Status();
    };
    steps.report = [&reported](std::int64_t epoch, std::int64_t)
    {
        reported.push_back(epoch);
    };

    SA_EXPECT_OK(Apps::run_epoch_loop(plan_of(2, 4, 2), steps));

    EXPECT_EQ(entered, (std::vector<std::int64_t>{4, 5}));
    EXPECT_EQ(stepped, (std::vector<std::size_t>{2, 0, 1, 2}));
    EXPECT_EQ(reported, (std::vector<std::int64_t>{1, 2}));
}

/** @brief Tests that the rate is set before every update, and only when a setter is given. */
TEST(Small_EpochLoop, SetsTheRateBeforeEveryUpdate)
{
    std::vector<char> calls;
    Apps::EpochSteps steps;
    steps.set_rate = [&calls]()
    {
        calls.push_back('r');
    };
    steps.run_step = [&calls](std::size_t)
    {
        calls.push_back('u');
        return Status();
    };
    steps.report = [](std::int64_t, std::int64_t)
    {
    };

    SA_EXPECT_OK(Apps::run_epoch_loop(plan_of(1), steps));

    EXPECT_EQ(calls, (std::vector<char>{'r', 'u', 'r', 'u', 'r', 'u'}));
}

/** @brief Tests that the first failed update ends the loop with its status. */
TEST(Small_EpochLoop, StopsAtTheFirstFailedUpdate)
{
    std::size_t updates = 0;
    Apps::EpochSteps steps;
    steps.run_step = [&updates](std::size_t index) -> Status
    {
        ++updates;
        SA_FAIL_IF(index == 1, StatusCode::DeviceFailed, "update {} failed", index);
        return Status();
    };
    steps.report = [](std::int64_t, std::int64_t)
    {
    };

    SA_EXPECT_FAILURE(Apps::run_epoch_loop(plan_of(2), steps), StatusCode::DeviceFailed,
                      "update 1");
    EXPECT_EQ(updates, 2u);
}
