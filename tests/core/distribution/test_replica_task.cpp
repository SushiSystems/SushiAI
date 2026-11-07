/**************************************************************************/
/* test_replica_task.cpp                                                  */
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
 * @file test_replica_task.cpp
 * @brief Tests the parameters a replica node carries, and what decode() refuses.
 * @author Mustafa Garip
 */

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/distribution/replica_task.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Distribution::K_REPLICA_TASK_WORDS;
    using SushiAI::Distribution::ReplicaTask;

    /** @brief Returns a view of @p words. */
    SushiRuntime::span<const std::uint64_t> view(const std::vector<std::uint64_t>& words)
    {
        return SushiRuntime::span<const std::uint64_t>(words.data(), words.size());
    }

    /** @brief Returns the task of replica 3 of 5, run 4, seed 0xFEED. */
    ReplicaTask example_task()
    {
        ReplicaTask task;
        task.replica_index = 3;
        task.replica_count = 5;
        task.run_index = 4;
        task.seed = 0xFEEDull;
        return task;
    }

    /** @brief Holds words decode() refuses and the substring the refusal must name. */
    struct DecodeRefusalRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the words handed to decode(). */
        std::vector<std::uint64_t> words;

        /** @brief Holds the substring the refusal must name. */
        const char* needle;
    };

    /** @brief Lists the word lists decode() refuses. */
    std::vector<DecodeRefusalRow> decode_refusal_rows()
    {
        return {
            {"RegressionAnotherWordCountNamingIt", {0, 0, 0, 1, 0, 0},
             "carries 6 parameters, not 4"},
            {"AReplicaOutsideItsStep", {2, 2, 0, 0}, "replica 2 of a step of 2"},
            {"AStepOfNoReplica", {0, 0, 0, 0}, "0 replicas"},
        };
    }

    /** @brief Runs each word list decode() refuses. */
    class Small_ReplicaTaskDecodeRefusal : public ::testing::TestWithParam<DecodeRefusalRow>
    {

    };
} // namespace

/** @brief Tests that the words follow the design's order: replica, count, run, seed. */
TEST(Small_ReplicaTask, EncodesInTheDesignOrder)
{
    const std::array<std::uint64_t, K_REPLICA_TASK_WORDS> words = example_task().encode();

    EXPECT_EQ(words, (std::array<std::uint64_t, K_REPLICA_TASK_WORDS>{3, 5, 4, 0xFEED}));
}

/** @brief Tests that the encoded words decode back to the same task. */
TEST(Small_ReplicaTask, EncodedWordsDecodeBackToTheSameTask)
{
    const std::array<std::uint64_t, K_REPLICA_TASK_WORDS> words = example_task().encode();

    SA_ASSERT_OK_AND_ASSIGN(const ReplicaTask decoded,
                            ReplicaTask::decode(SushiRuntime::span<const std::uint64_t>(
                                words.data(), words.size())));

    EXPECT_EQ(decoded.encode(), words);
}

/** @brief Tests that a task maps to the draws of its seed, replica and run. */
TEST(Small_ReplicaTask, ATaskMapsToTheDrawsOfItsSeedReplicaAndRun)
{
    const ReplicaTask task = example_task();

    EXPECT_EQ(task.draws().seed, 0xFEEDull);
    EXPECT_EQ(task.draws().replica_index, 3u);
    EXPECT_EQ(task.draws().run_index, 4u);
}

/** @brief Tests that decode() refuses a word list that breaks a rule of the task, naming it. */
TEST_P(Small_ReplicaTaskDecodeRefusal, DecodeRefusesWordsThatBreakARuleNamingIt)
{
    const DecodeRefusalRow& row = GetParam();

    SA_EXPECT_FAILURE(ReplicaTask::decode(view(row.words)), StatusCode::InvalidDescription,
                      row.needle);
}

INSTANTIATE_TEST_SUITE_P(Words, Small_ReplicaTaskDecodeRefusal,
                         ::testing::ValuesIn(decode_refusal_rows()),
                         [](const ::testing::TestParamInfo<DecodeRefusalRow>& info)
                         {
                             return std::string(info.param.name);
                         });
