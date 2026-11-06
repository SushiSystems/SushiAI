/**************************************************************************/
/* test_in_memory_step_recorder.cpp                                       */
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
 * @file test_in_memory_step_recorder.cpp
 * @brief Tests that InMemoryStepRecorder keeps the promise of the StepRecorder interface.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/in_memory_step_recorder.hpp>
#include <SushiAI/graph/step_recorder.hpp>

using namespace SushiAI;
using Graph::InMemoryStepRecorder;
using Call = Graph::InMemoryStepRecorder::Call;

/** @brief Tests that a fresh recorder has received no call. */
TEST(Small_InMemoryStepRecorder, StartsWithNoCall)
{
    const InMemoryStepRecorder recorder;

    EXPECT_TRUE(recorder.calls().empty());
}

/** @brief Tests that the replay hooks, called through the interface, are journaled in order. */
TEST(Small_InMemoryStepRecorder, TheReplayHooksAreJournaledInOrder)
{
    InMemoryStepRecorder recorder(3);
    Graph::StepRecorder& port = recorder;

    port.before_replay();
    port.after_replay();
    port.before_replay();

    const std::vector<Call> expected = {Call::BeforeReplay, Call::AfterReplay,
                                        Call::BeforeReplay};
    EXPECT_EQ(recorder.calls(), expected);
}

/** @brief Tests that a recorder driven by a list is told before and after a replay once each. */
TEST(Small_InMemoryStepRecorder, AListDrivesTheReplayHooksOfItsMember)
{
    InMemoryStepRecorder recorder;
    Graph::StepRecorderList list;
    SA_EXPECT_OK(list.add(recorder));

    list.before_replay();
    list.after_replay();

    const std::vector<Call> expected = {Call::BeforeReplay, Call::AfterReplay};
    EXPECT_EQ(recorder.calls(), expected);
}
