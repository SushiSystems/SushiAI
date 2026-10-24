/**************************************************************************/
/* test_replica_draws.cpp                                                 */
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
 * @file test_replica_draws.cpp
 * @brief Tests that a replica's seed takes the run's seed, the replica and the run index.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/distribution/replica_draws.hpp>

namespace
{
    using SushiAI::Distribution::ReplicaDraws;
    using SushiAI::Distribution::replica_seed;

    /** @brief Holds the seed every case starts from. */
    constexpr std::uint64_t K_SEED = 0x5EEDull;
} // namespace

/** @brief Tests that the same draws give the same seed, the repeatability a run relies on. */
TEST(Small_ReplicaDraws, TheSameDrawsGiveTheSameSeed)
{
    EXPECT_EQ(replica_seed(ReplicaDraws{K_SEED, 2, 1}), replica_seed(ReplicaDraws{K_SEED, 2, 1}));
}

/** @brief Tests that changing any one of the seed, the replica and the run moves the seed. */
TEST(Small_ReplicaDraws, EachOfTheThreeMovesTheSeed)
{
    const std::uint64_t base = replica_seed(ReplicaDraws{K_SEED, 2, 1});
    EXPECT_NE(replica_seed(ReplicaDraws{K_SEED + 1, 2, 1}), base);
    EXPECT_NE(replica_seed(ReplicaDraws{K_SEED, 3, 1}), base);
    EXPECT_NE(replica_seed(ReplicaDraws{K_SEED, 2, 0}), base);
}

/** @brief Tests that the replica and the run are not interchangeable. */
TEST(Small_ReplicaDraws, TheReplicaAndTheRunAreNotInterchangeable)
{
    EXPECT_NE(replica_seed(ReplicaDraws{K_SEED, 1, 2}), replica_seed(ReplicaDraws{K_SEED, 2, 1}));
}
