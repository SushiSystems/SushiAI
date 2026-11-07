/**************************************************************************/
/* test_trailing_ledger.cpp                                               */
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
 * @file test_trailing_ledger.cpp
 * @brief Tests which trailing work TrailingLedger plans for a step and what it remembers.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <initializer_list>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/step_executor.hpp>
#include <SushiAI/eager/trailing_ledger.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/in_memory_step_recorder.hpp>

using namespace SushiAI;
using Eager::RegionKey;
using Eager::StepWork;
using Eager::TrailingLedger;
using Eager::TrailingPlan;
using Graph::InMemoryStepRecorder;

namespace
{
    /** @brief Holds the key the trailing work is recorded under in most cases. */
    constexpr RegionKey K_TRAILING_KEY = 9;

    /**
     * @brief Builds a step with one region under each of @p region_keys.
     * @param recorder The trailing work of the step, or null.
     * @param trailing_key The key the trailing work is recorded under.
     */
    [[nodiscard]] StepWork step_with(Graph::StepRecorder* recorder, RegionKey trailing_key,
                                     std::initializer_list<RegionKey> region_keys)
    {
        StepWork work;
        for (RegionKey key : region_keys)
        {
            Eager::RegionWork region;
            region.order.key = key;
            work.regions.push_back(std::move(region));
        }

        work.trailing = recorder;
        work.trailing_key = trailing_key;
        return work;
    }

    /** @brief Makes @p ledger hold @p recorder as a finished recording of three tasks. */
    void hold(TrailingLedger& ledger, InMemoryStepRecorder& recorder, RegionKey key)
    {
        ledger.begin_recording(key);
        ledger.finish_recording(recorder, 3);
    }

    /** @brief Returns the plan of @p work, stopping the test when it is refused. */
    [[nodiscard]] TrailingPlan plan_of(const TrailingLedger& ledger, const StepWork& work)
    {
        return Test::require_value(ledger.plan(work));
    }
} // namespace

/** @brief Tests that a step with no trailing work plans nothing on a fresh ledger. */
TEST(Small_TrailingLedger, AStepWithNoTrailingWorkPlansNothing)
{
    const TrailingLedger ledger;

    const TrailingPlan plan = plan_of(ledger, step_with(nullptr, Eager::K_NO_REGION, {0, 1}));

    EXPECT_FALSE(plan.retire);
    EXPECT_FALSE(plan.record);
}

/** @brief Tests that trailing work the ledger does not hold is planned for recording. */
TEST(Small_TrailingLedger, NewTrailingWorkIsPlannedForRecordingAndRetiresNothing)
{
    const TrailingLedger ledger;
    InMemoryStepRecorder update;

    const TrailingPlan plan = plan_of(ledger, step_with(&update, K_TRAILING_KEY, {0, 1}));

    EXPECT_TRUE(plan.record);
    EXPECT_FALSE(plan.retire);
}

/** @brief Tests that the recorder and key the ledger holds are neither retired nor recorded. */
TEST(Small_TrailingLedger, HeldTrailingWorkUnderTheSameKeyIsKept)
{
    TrailingLedger ledger;
    InMemoryStepRecorder update;
    hold(ledger, update, K_TRAILING_KEY);

    const TrailingPlan plan = plan_of(ledger, step_with(&update, K_TRAILING_KEY, {0, 1}));

    EXPECT_FALSE(plan.record);
    EXPECT_FALSE(plan.retire);
}

/** @brief Tests that held trailing work under a moved key is retired and recorded again. */
TEST(Small_TrailingLedger, HeldTrailingWorkUnderAMovedKeyIsRetiredAndRecordedAgain)
{
    TrailingLedger ledger;
    InMemoryStepRecorder update;
    hold(ledger, update, K_TRAILING_KEY);

    const TrailingPlan plan = plan_of(ledger, step_with(&update, K_TRAILING_KEY + 2, {0, 1}));

    EXPECT_TRUE(plan.retire);
    EXPECT_TRUE(plan.record);
}

/** @brief Tests that another recorder replaces the held one. */
TEST(Small_TrailingLedger, AnotherRecorderRetiresTheHeldOneAndIsRecorded)
{
    TrailingLedger ledger;
    InMemoryStepRecorder held;
    InMemoryStepRecorder other;
    hold(ledger, held, K_TRAILING_KEY);

    const TrailingPlan plan = plan_of(ledger, step_with(&other, K_TRAILING_KEY, {0, 1}));

    EXPECT_TRUE(plan.retire);
    EXPECT_TRUE(plan.record);
}

/** @brief Tests that a step with no trailing work retires the held one and records nothing. */
TEST(Small_TrailingLedger, AStepWithoutTrailingWorkRetiresTheHeldOne)
{
    TrailingLedger ledger;
    InMemoryStepRecorder update;
    hold(ledger, update, K_TRAILING_KEY);

    const TrailingPlan plan = plan_of(ledger, step_with(nullptr, Eager::K_NO_REGION, {0, 1}));

    EXPECT_TRUE(plan.retire);
    EXPECT_FALSE(plan.record);
}

/** @brief Tests that a recording that began and did not finish is retired and recorded again. */
TEST(Small_TrailingLedger, AFailedRecordingIsRetiredAndRecordedAgain)
{
    TrailingLedger ledger;
    InMemoryStepRecorder update;
    ledger.begin_recording(K_TRAILING_KEY);

    const TrailingPlan plan = plan_of(ledger, step_with(&update, K_TRAILING_KEY, {0, 1}));

    EXPECT_TRUE(plan.retire);
    EXPECT_TRUE(plan.record);
}

/** @brief Tests that trailing work with no key is refused. */
TEST(Small_TrailingLedger, TrailingWorkWithNoKeyIsRefused)
{
    const TrailingLedger ledger;
    InMemoryStepRecorder update;

    SA_EXPECT_FAILURE(ledger.plan(step_with(&update, Eager::K_NO_REGION, {0, 1})),
                      StatusCode::InvalidDescription, "above every region; got none");
}

/** @brief Tests that a trailing key equal to a region key is refused, so it cannot run after it. */
TEST(Small_TrailingLedger, ATrailingKeyEqualToARegionKeyIsRefused)
{
    const TrailingLedger ledger;
    InMemoryStepRecorder update;

    SA_EXPECT_FAILURE(ledger.plan(step_with(&update, 4, {0, 4})),
                      StatusCode::InvalidDescription, "does not lie above the region under key 4");
}

/** @brief Tests that a trailing key below a region key is refused. */
TEST(Small_TrailingLedger, ATrailingKeyBelowARegionKeyIsRefused)
{
    const TrailingLedger ledger;
    InMemoryStepRecorder update;

    SA_EXPECT_FAILURE(ledger.plan(step_with(&update, 1, {0, 4})),
                      StatusCode::InvalidDescription, "does not lie above the region under key 4");
}

/** @brief Tests that trailing work already held is planned without a key check. */
TEST(Small_TrailingLedger, HeldTrailingWorkIsNotCheckedAgainstTheRegionKeys)
{
    TrailingLedger ledger;
    InMemoryStepRecorder update;
    hold(ledger, update, K_TRAILING_KEY);

    SA_EXPECT_OK(ledger.plan(step_with(&update, K_TRAILING_KEY, {0, 1})).status());
}

/** @brief Tests that each begun recording is counted once. */
TEST(Small_TrailingLedger, EachBegunRecordingIsCounted)
{
    TrailingLedger ledger;
    InMemoryStepRecorder update;

    ledger.begin_recording(K_TRAILING_KEY);
    ledger.finish_recording(update, 3);
    ledger.begin_recording(K_TRAILING_KEY + 1);

    EXPECT_EQ(ledger.record_count(), 2u);
}

/** @brief Tests that the ledger reports the tasks of the recording it holds. */
TEST(Small_TrailingLedger, ItReportsTheTasksOfTheHeldRecording)
{
    TrailingLedger ledger;
    InMemoryStepRecorder update;

    hold(ledger, update, K_TRAILING_KEY);

    EXPECT_EQ(ledger.operation_count(), 3u);
    EXPECT_EQ(ledger.key(), K_TRAILING_KEY);
}

/** @brief Tests that clear forgets the key, the recorder and its tasks. */
TEST(Small_TrailingLedger, ClearForgetsTheKeyAndTheTasks)
{
    TrailingLedger ledger;
    InMemoryStepRecorder update;
    hold(ledger, update, K_TRAILING_KEY);

    ledger.clear();

    EXPECT_EQ(ledger.key(), Eager::K_NO_REGION);
    EXPECT_EQ(ledger.operation_count(), 0u);
}

/** @brief Tests that the replay hooks reach the held recorder, once per call. */
TEST(Small_TrailingLedger, TheReplayHooksReachTheHeldRecorder)
{
    TrailingLedger ledger;
    InMemoryStepRecorder::Journal journal;
    InMemoryStepRecorder update(1, journal);
    hold(ledger, update, K_TRAILING_KEY);
    journal.clear();

    ledger.before_replay();
    ledger.after_replay();

    const InMemoryStepRecorder::Journal expected = {
        {1, InMemoryStepRecorder::Call::BeforeReplay},
        {1, InMemoryStepRecorder::Call::AfterReplay}};
    EXPECT_EQ(journal, expected);
}

/** @brief Tests that the replay hooks do nothing when no recorder is held. */
TEST(Small_TrailingLedger, TheReplayHooksDoNothingWithoutAHeldRecorder)
{
    TrailingLedger ledger;
    InMemoryStepRecorder::Journal journal;
    InMemoryStepRecorder update(1, journal);
    ledger.begin_recording(K_TRAILING_KEY);

    ledger.before_replay();
    ledger.after_replay();

    EXPECT_TRUE(journal.empty());
}
