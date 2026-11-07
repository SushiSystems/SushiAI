/**************************************************************************/
/* test_cluster_worker_watch.cpp                                              */
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
 * @file test_cluster_worker_watch.cpp
 * @brief Tests that ClusterWorkerWatch keeps the WorkerWatch contract over a real cluster.
 * @author Mustafa Garip
 */

#include <chrono>
#include <gtest/gtest.h>

#include "cluster_fabric.hpp"
#include "cluster_worker_watch.hpp"
#include "worker_watch_contract.hpp"

namespace
{
    /** @brief Runs the WorkerWatch contract against a real cluster with two attached workers. */
    class Medium_WorkerWatchContract : public ::testing::Test
    {
        protected:
            /** @brief Attaches the workers "left" and "right" and waits for both. */
            void SetUp() override
            {
                fabric.start_worker("left");
                fabric.start_worker("right");
                ASSERT_TRUE(watch.wait_for_count(2, std::chrono::seconds(10)));
            }

            /// @brief The master and the workers; they outlive the watch.
            SushiAI::Apps::ClusterTest::Fabric fabric;

            /// @brief The adapter under test, over the fabric's cluster.
            SushiAI::Apps::ClusterWorkerWatch watch{fabric.cluster()};
    };
} // namespace

/** @brief Tests that the cluster watch lists the workers that attached. */
TEST_F(Medium_WorkerWatchContract, ListsTheAttachedWorkers)
{
    SushiAI::Test::WorkerWatchContract::expect_lists_the_attached_workers(watch);
}

/** @brief Tests that the cluster watch reaches a count already attached. */
TEST_F(Medium_WorkerWatchContract, ACountAlreadyAttachedIsReached)
{
    SushiAI::Test::WorkerWatchContract::expect_a_count_already_attached_is_reached(watch);
}

/** @brief Tests that the cluster watch does not reach a count above the attached ones. */
TEST_F(Medium_WorkerWatchContract, ACountAboveTheAttachedIsNotReached)
{
    SushiAI::Test::WorkerWatchContract::expect_a_count_above_the_attached_is_not_reached(watch);
}

/** @brief Tests that a failed wait on the cluster watch passes its bound on the steady clock. */
TEST_F(Medium_WorkerWatchContract, AFailedWaitPassesItsBoundOnTheClock)
{
    SushiAI::Test::WorkerWatchContract::expect_a_failed_wait_passes_its_bound_on_the_clock(watch);
}

/** @brief Tests that the steady clock of the cluster watch never goes back. */
TEST_F(Medium_WorkerWatchContract, TheClockNeverGoesBack)
{
    SushiAI::Test::WorkerWatchContract::expect_the_clock_never_goes_back(watch);
}
