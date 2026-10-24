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
 * @file test_replica_position.cpp
 * @brief Tests ReplicaPosition: the position of a step and its read from a pack's bytes.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <SushiRuntime/distributed/buffer_view.hpp>
#include <SushiAI/distribution/replica_position.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using SushiAI::StatusCode;
using SushiAI::Distribution::K_REPLICA_POSITION_BYTES;
using SushiAI::Distribution::ReplicaPosition;

/** @brief Tests that a step count splits into an epoch and a batch of that epoch. */
TEST(Small_ReplicaPosition, AStepSplitsIntoEpochAndBatch)
{
    const ReplicaPosition first = ReplicaPosition::at_step(0, 5);
    EXPECT_EQ(first.epoch, 0u);
    EXPECT_EQ(first.batch_index, 0u);

    const ReplicaPosition later = ReplicaPosition::at_step(12, 5);
    EXPECT_EQ(later.epoch, 2u);
    EXPECT_EQ(later.batch_index, 2u);
}

/** @brief Tests that read() takes the epoch and the batch from the first two words. */
TEST(Small_ReplicaPosition, ReadTakesTheFirstTwoWords)
{
    std::uint64_t words[3] = {7, 41, 99};
    const SushiRuntime::Distributed::BufferView view{words, sizeof(words)};

    SA_ASSERT_OK_AND_ASSIGN(const ReplicaPosition position, ReplicaPosition::read(view));
    EXPECT_EQ(position.epoch, 7u);
    EXPECT_EQ(position.batch_index, 41u);
}

/** @brief Tests that read() refuses a view with no memory or too few bytes. */
TEST(Small_ReplicaPosition, ReadRefusesAViewTooSmall)
{
    std::uint64_t word = 3;
    const SushiRuntime::Distributed::BufferView empty{nullptr, K_REPLICA_POSITION_BYTES};
    const SushiRuntime::Distributed::BufferView short_view{&word, sizeof(word)};

    SA_EXPECT_FAILURE(ReplicaPosition::read(empty), StatusCode::InvalidDescription, "no memory");
    SA_EXPECT_FAILURE(ReplicaPosition::read(short_view), StatusCode::InvalidDescription,
                      "8 bytes");
}
