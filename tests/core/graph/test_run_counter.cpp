/**************************************************************************/
/* test_run_counter.cpp                                                   */
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


/**
 * @file test_run_counter.cpp
 * @brief Tests that the run counter takes the count of a saved run once and before the first run.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/run_counter.hpp>

using namespace SushiAI;

/** @brief Tests that a new counter has counted no run. */
TEST(Small_RunCounter, ANewCounterHasCountedNoRun)
{
    const Graph::RunCounter counter;

    EXPECT_EQ(counter.count(), 0u);
}

/** @brief Tests that each advance counts one run. */
TEST(Small_RunCounter, EachAdvanceCountsOneRun)
{
    Graph::RunCounter counter;

    counter.advance();
    counter.advance();

    EXPECT_EQ(counter.count(), 2u);
}

/** @brief Tests that a count above 0 is restored once. */
TEST(Small_RunCounter, ACountAboveZeroIsRestoredOnce)
{
    Graph::RunCounter counter;

    SA_EXPECT_OK(counter.restore(5));
    SA_EXPECT_FAILURE(counter.restore(7), StatusCode::InvalidDescription, "run 5");
    EXPECT_EQ(counter.count(), 5u);
}

/** @brief Tests that restoring 0 leaves a new counter new, so a later restore is accepted. */
TEST(Small_RunCounter, RestoringZeroLeavesTheCounterNew)
{
    Graph::RunCounter counter;

    SA_EXPECT_OK(counter.restore(0));
    SA_EXPECT_OK(counter.restore(5));
    EXPECT_EQ(counter.count(), 5u);
}

/** @brief Tests that a counter that has advanced refuses a restore and keeps its count. */
TEST(Small_RunCounter, ACounterThatHasRunRefusesARestore)
{
    Graph::RunCounter counter;
    counter.advance();

    SA_EXPECT_FAILURE(counter.restore(4), StatusCode::InvalidDescription, "before its first run",
                      "run 1");
    EXPECT_EQ(counter.count(), 1u);
}
