/**************************************************************************/
/* test_failure_block_counter.cpp                                         */
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
 * @file test_failure_block_counter.cpp
 * @brief Tests that the installed FailureBlockCounter counts the blocks of Status and can be replaced.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/failure_block_counter.hpp>
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

/** @brief Tests that installing a counter returns the one it replaced. */
TEST(Small_FailureBlockCounter, InstallingACounterReturnsTheOneItReplaced)
{
    InMemoryFailureBlockCounter first;
    InMemoryFailureBlockCounter second;

    FailureBlockCounter& replaced = set_failure_block_counter(&first);

    EXPECT_EQ(&replaced, &second);
    static_cast<void>(set_failure_block_counter(&second));
}

/** @brief Tests that a null counter installs the process counter, which Status then counts in. */
TEST(Small_FailureBlockCounter, ANullCounterInstallsTheProcessCounter)
{
    const InMemoryFailureBlockCounter counter;
    FailureBlockCounter& before = set_failure_block_counter(nullptr);
    const std::size_t process_before = live_failure_count();

    {
        const Status held = block_holder();

        EXPECT_EQ(live_failure_count(), process_before + 1);
        EXPECT_EQ(counter.live(), 0u);
    }

    static_cast<void>(set_failure_block_counter(&before));
}
