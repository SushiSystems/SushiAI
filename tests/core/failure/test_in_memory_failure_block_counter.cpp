/**************************************************************************/
/* test_in_memory_failure_block_counter.cpp                               */
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
 * @file test_in_memory_failure_block_counter.cpp
 * @brief Tests that InMemoryFailureBlockCounter counts its own scope and restores the counter it replaced.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/in_memory_failure_block_counter.hpp>
#include <SushiAI/failure/status.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Builds a failure whose block the installed counter counts. */
    Status block_holder()
    {
        return Status::from_message(StatusCode::InvalidFile, []()
        {
            return std::string("a block to count");
        });
    }
} // namespace

/** @brief Tests that the counter reads the blocks made under it and not yet released. */
TEST(Small_InMemoryFailureBlockCounter, ReadsTheBlocksMadeUnderItAndNotYetReleased)
{
    const InMemoryFailureBlockCounter counter;

    {
        const Status first = block_holder();
        const Status second = block_holder();

        EXPECT_EQ(counter.live(), 2u);
    }

    EXPECT_EQ(counter.live(), 0u);
}

/** @brief Tests that a block made before the counter is not counted by it. */
TEST(Small_InMemoryFailureBlockCounter, DoesNotCountABlockMadeBeforeIt)
{
    const Status earlier = block_holder();

    const InMemoryFailureBlockCounter counter;

    EXPECT_EQ(counter.live(), 0u);
}

/** @brief Tests that the process counter counts again once the counter is gone. */
TEST(Small_InMemoryFailureBlockCounter, TheProcessCounterCountsAgainOnceItIsGone)
{
    const std::size_t before = Detail::live_failure_count();
    {
        const InMemoryFailureBlockCounter counter;
        const Status hidden = block_holder();

        EXPECT_EQ(Detail::live_failure_count(), before);
    }

    const Status visible = block_holder();

    EXPECT_EQ(Detail::live_failure_count(), before + 1);
}
