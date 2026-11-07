/**************************************************************************/
/* test_replica_position.cpp                                              */
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
 * @file test_replica_position.cpp
 * @brief Tests ReplicaPosition: the position of a step and its read from a pack's bytes.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/distributed/buffer_view.hpp>
#include <SushiAI/distribution/replica_position.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using SushiAI::StatusCode;
using SushiAI::Distribution::K_REPLICA_POSITION_BYTES;
using SushiAI::Distribution::ReplicaPosition;

namespace
{
    /** @brief Holds a view too small to read and the word the refusal must name. */
    struct SmallViewRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the base address of the view; null for a view with no memory. */
        void* memory;

        /** @brief Holds the number of bytes the view claims. */
        std::size_t bytes;

        /** @brief Holds the substring the refusal must name. */
        const char* needle;
    };

    /** @brief Returns the one word a view too short to read points at. */
    std::uint64_t* one_word()
    {
        static std::uint64_t word = 3;
        return &word;
    }

    /** @brief Lists the views read() refuses: no memory, and eight bytes too few. */
    std::vector<SmallViewRow> small_view_rows()
    {
        return {
            {"NoMemory", nullptr, K_REPLICA_POSITION_BYTES, "no memory"},
            {"EightBytes", one_word(), sizeof(std::uint64_t), "8 bytes"},
        };
    }

    /** @brief Runs each view read() refuses. */
    class Small_ReplicaPositionView : public ::testing::TestWithParam<SmallViewRow>
    {

    };
} // namespace

/** @brief Tests that a step count splits into an epoch and a batch of that epoch. */
TEST(Small_ReplicaPosition, AStepSplitsIntoEpochAndBatch)
{
    const ReplicaPosition first = ReplicaPosition::at_step(0, 5);
    EXPECT_EQ(first.epoch, 0u);
    EXPECT_EQ(first.batch_index, 0u);

    const ReplicaPosition later = ReplicaPosition::at_step(12, 5);
    EXPECT_EQ(later.epoch, 2u);
    EXPECT_EQ(later.batch_index, 2u);
    EXPECT_EQ(later.step, 12u);
}

/** @brief Tests that the draw index counts every micro-batch of every step before it. */
TEST(Small_ReplicaPosition, TheDrawIndexCountsTheMicroBatchesOfEveryEarlierStep)
{
    const ReplicaPosition position = ReplicaPosition::at_step(7, 5);

    EXPECT_EQ(position.draw_index(3, 0), 21u);
    EXPECT_EQ(position.draw_index(3, 2), 23u);
    EXPECT_EQ(ReplicaPosition::at_step(0, 5).draw_index(1, 0), 0u);
}

/** @brief Tests that read() takes the epoch, the batch and the step from the first three words. */
TEST(Small_ReplicaPosition, ReadTakesTheFirstThreeWords)
{
    std::uint64_t words[4] = {7, 41, 123, 99};
    const SushiRuntime::Distributed::BufferView view{words, sizeof(words)};

    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPosition position, ReplicaPosition::read(view));
    EXPECT_EQ(position.epoch, 7u);
    EXPECT_EQ(position.batch_index, 41u);
    EXPECT_EQ(position.step, 123u);
}

/** @brief Tests that read() refuses a view with no memory or too few bytes, naming the fault. */
TEST_P(Small_ReplicaPositionView, ReadRefusesAViewTooSmall)
{
    const SmallViewRow& row = GetParam();
    const SushiRuntime::Distributed::BufferView view{row.memory, row.bytes};

    SA_EXPECT_FAILURE(ReplicaPosition::read(view), StatusCode::InvalidDescription, row.needle);
}

INSTANTIATE_TEST_SUITE_P(Views, Small_ReplicaPositionView, ::testing::ValuesIn(small_view_rows()),
                         [](const ::testing::TestParamInfo<SmallViewRow>& info)
                         {
                             return std::string(info.param.name);
                         });
