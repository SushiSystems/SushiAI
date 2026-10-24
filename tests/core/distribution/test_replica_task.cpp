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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_replica_task.cpp
 * @brief Tests the parameters a replica node carries, and what decode() refuses.
 * @author Mustafa Garip
 */

#include <array>
#include <cstdint>
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
} // namespace

/** @brief Tests that the words follow the design's order and decode back to the same task. */
TEST(Small_ReplicaTask, EncodesInTheDesignOrderAndDecodesBack)
{
    ReplicaTask task;
    task.epoch = 2;
    task.batch_index = 17;
    task.replica_index = 3;
    task.replica_count = 5;
    task.run_index = 4;
    task.seed = 0xFEEDull;

    const std::array<std::uint64_t, K_REPLICA_TASK_WORDS> words = task.encode();
    EXPECT_EQ(words, (std::array<std::uint64_t, K_REPLICA_TASK_WORDS>{2, 17, 3, 5, 4, 0xFEED}));

    SA_ASSERT_OK_AND_ASSIGN(const ReplicaTask decoded,
                            ReplicaTask::decode(SushiRuntime::span<const std::uint64_t>(
                                words.data(), words.size())));
    EXPECT_EQ(decoded.encode(), words);
    EXPECT_EQ(decoded.draws().seed, 0xFEEDull);
    EXPECT_EQ(decoded.draws().replica_index, 3u);
    EXPECT_EQ(decoded.draws().run_index, 4u);
}

/** @brief Tests that decode() refuses a count of words other than six. */
TEST(Small_ReplicaTask, DecodeRefusesAnotherWordCount)
{
    const std::vector<std::uint64_t> words = {0, 0, 0, 1, 0};
    SA_EXPECT_FAILURE(ReplicaTask::decode(view(words)), StatusCode::InvalidDescription,
                      "carries 6 parameters, not 5");
}

/** @brief Tests that decode() refuses a replica outside its step and a step of no replica. */
TEST(Small_ReplicaTask, DecodeRefusesAReplicaOutsideItsStep)
{
    const std::vector<std::uint64_t> outside = {0, 0, 2, 2, 0, 0};
    SA_EXPECT_FAILURE(ReplicaTask::decode(view(outside)), StatusCode::InvalidDescription,
                      "replica 2 of a step of 2");

    const std::vector<std::uint64_t> empty = {0, 0, 0, 0, 0, 0};
    SA_EXPECT_FAILURE(ReplicaTask::decode(view(empty)), StatusCode::InvalidDescription,
                      "0 replicas");
}
