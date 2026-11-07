/**************************************************************************/
/* test_in_memory_worker_watch.cpp                                              */
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
 * @file test_in_memory_worker_watch.cpp
 * @brief Tests that InMemoryWorkerWatch keeps the WorkerWatch contract and moves its own clock.
 * @author Mustafa Garip
 */

#include <chrono>
#include <string>
#include <gtest/gtest.h>

#include "in_memory_worker_watch.hpp"
#include "worker_watch_contract.hpp"

namespace
{
    using SushiAI::Apps::InMemoryWorkerWatch;
    using SushiAI::Test::WorkerWatchContract;
    using std::chrono::milliseconds;

    /** @brief Runs the WorkerWatch contract against two workers that attached at the start. */
    class Small_WorkerWatchContract : public ::testing::Test
    {
        protected:
            /// @brief The adapter under test; each test makes its own.
            InMemoryWorkerWatch watch{{{"left", milliseconds(0)}, {"right", milliseconds(0)}}};
    };
} // namespace

/** @brief Tests that the in-memory watch lists the workers that attached. */
TEST_F(Small_WorkerWatchContract, ListsTheAttachedWorkers)
{
    WorkerWatchContract::expect_lists_the_attached_workers(watch);
}

/** @brief Tests that the in-memory watch reaches a count already attached. */
TEST_F(Small_WorkerWatchContract, ACountAlreadyAttachedIsReached)
{
    WorkerWatchContract::expect_a_count_already_attached_is_reached(watch);
}

/** @brief Tests that the in-memory watch does not reach a count above the attached ones. */
TEST_F(Small_WorkerWatchContract, ACountAboveTheAttachedIsNotReached)
{
    WorkerWatchContract::expect_a_count_above_the_attached_is_not_reached(watch);
}

/** @brief Tests that a failed wait of the in-memory watch passes its bound on its clock. */
TEST_F(Small_WorkerWatchContract, AFailedWaitPassesItsBoundOnTheClock)
{
    WorkerWatchContract::expect_a_failed_wait_passes_its_bound_on_the_clock(watch);
}

/** @brief Tests that the clock of the in-memory watch never goes back. */
TEST_F(Small_WorkerWatchContract, TheClockNeverGoesBack)
{
    WorkerWatchContract::expect_the_clock_never_goes_back(watch);
}

/** @brief Tests that a worker is listed only from the time it arrives. */
TEST(Small_InMemoryWorkerWatch, AWorkerIsListedFromItsArrival)
{
    InMemoryWorkerWatch watch({{"late", milliseconds(500)}});

    EXPECT_TRUE(watch.attached_names().empty());
    EXPECT_TRUE(watch.wait_for_count(1, milliseconds(500)));
    EXPECT_EQ(watch.attached_names().size(), 1u);
}

/** @brief Tests that a wait moves the clock to the arrival that reaches the count. */
TEST(Small_InMemoryWorkerWatch, AWaitMovesTheClockToTheArrivalThatReachesTheCount)
{
    InMemoryWorkerWatch watch({{"left", milliseconds(200)}, {"right", milliseconds(900)}});

    EXPECT_TRUE(watch.wait_for_count(2, milliseconds(5000)));

    EXPECT_EQ(watch.elapsed(), milliseconds(900));
}

/** @brief Tests that a wait whose count arrives after its bound moves the clock by the bound. */
TEST(Small_InMemoryWorkerWatch, AWaitThatTimesOutMovesTheClockByItsBound)
{
    InMemoryWorkerWatch watch({{"late", milliseconds(900)}});

    EXPECT_FALSE(watch.wait_for_count(1, milliseconds(300)));

    EXPECT_EQ(watch.elapsed(), milliseconds(300));
}
