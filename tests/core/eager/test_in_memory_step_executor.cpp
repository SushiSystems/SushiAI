/**************************************************************************/
/* test_in_memory_step_executor.cpp                                       */
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
 * @file test_in_memory_step_executor.cpp
 * @brief Tests that InMemoryStepExecutor keeps the promise of StepExecutor::discard().
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiAI/eager/in_memory_step_executor.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using Eager::InMemoryStepExecutor;


/** @brief Tests that a fresh executor has seen no step and discarded nothing. */
TEST(Small_InMemoryStepExecutor, StartsWithNoStepAndNoDiscard)
{
    const InMemoryStepExecutor executor;

    EXPECT_TRUE(executor.seen().empty());
    EXPECT_EQ(executor.discard_count(), 0u);
}

/** @brief Tests that discard through the interface succeeds and is counted. */
TEST(Small_InMemoryStepExecutor, DiscardSucceedsAndIsCounted)
{
    InMemoryStepExecutor executor;
    Eager::StepExecutor& port = executor;

    SA_EXPECT_OK(port.discard());
    SA_EXPECT_OK(port.discard());

    EXPECT_EQ(executor.discard_count(), 2u);
}

/** @brief Tests that a scripted failure of discard is returned, and counted all the same. */
TEST(Small_InMemoryStepExecutor, AScriptedDiscardFailureIsReturned)
{
    InMemoryStepExecutor executor;
    executor.script_discard(Status::from_message(StatusCode::OutOfMemory, []()
    {
        return std::string("no room to list the drops");
    }));

    SA_EXPECT_FAILURE(executor.discard(), StatusCode::OutOfMemory, "no room");
    EXPECT_EQ(executor.discard_count(), 1u);
}
