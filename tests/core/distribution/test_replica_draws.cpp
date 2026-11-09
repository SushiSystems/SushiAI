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
 * @file test_replica_draws.cpp
 * @brief Tests that a replica's seed takes the run's seed, the replica and the run index.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/distribution/replica_draws.hpp>
#include <support/splitmix64.hpp>
#include <support/test_names.hpp>

namespace
{
    using SushiAI::Distribution::ReplicaDraws;
    using SushiAI::Distribution::replica_seed;
    using TestSupport::reference_splitmix64;

    /** @brief Holds the seed every case starts from. */
    constexpr std::uint64_t K_SEED = 0x5EEDull;

    /** @brief Returns the seed the published SplitMix64 sequence gives @p draws. */
    std::uint64_t reference_replica_seed(const ReplicaDraws& draws)
    {
        return reference_splitmix64(reference_splitmix64(draws.seed, draws.replica_index),
                                    draws.run_index);
    }

    /** @brief Holds two draws that differ and the name of what differs. */
    struct DrawPair
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the draws the seed is compared from. */
        ReplicaDraws first;

        /** @brief Holds the draws that differ from the first. */
        ReplicaDraws second;
    };

    /** @brief Lists draws that differ in the seed, the replica, the run, and in the last two swapped. */
    std::vector<DrawPair> draw_pairs()
    {
        return {
            {"ADifferentSeed", {K_SEED, 2, 1}, {K_SEED + 1, 2, 1}},
            {"ADifferentReplica", {K_SEED, 2, 1}, {K_SEED, 3, 1}},
            {"ADifferentRun", {K_SEED, 2, 1}, {K_SEED, 2, 0}},
            {"TheReplicaAndTheRunSwapped", {K_SEED, 1, 2}, {K_SEED, 2, 1}},
        };
    }

    /** @brief Runs each pair of draws that differ. */
    class Small_ReplicaDrawsPair : public ::testing::TestWithParam<DrawPair>
    {

    };

    /** @brief Runs each draw whose seed the published sequence gives. */
    class Small_ReplicaDrawsSeed : public ::testing::TestWithParam<ReplicaDraws>
    {

    };
} // namespace

/** @brief Tests that the seed of draws is the published SplitMix64 of the replica stream at the run. */
TEST_P(Small_ReplicaDrawsSeed, TheSeedIsTheSplitMixOfTheReplicaStreamAtTheRun)
{
    EXPECT_EQ(replica_seed(GetParam()), reference_replica_seed(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(Draws, Small_ReplicaDrawsSeed,
                         ::testing::Values(ReplicaDraws{K_SEED, 2, 1}, ReplicaDraws{K_SEED, 0, 0},
                                           ReplicaDraws{0, 3, 5}));

/** @brief Tests that draws differing in the seed, the replica or the run give different seeds. */
TEST_P(Small_ReplicaDrawsPair, DrawsThatDifferGiveDifferentSeeds)
{
    const DrawPair& row = GetParam();

    EXPECT_NE(replica_seed(row.first), replica_seed(row.second));
}

INSTANTIATE_TEST_SUITE_P(Draws, Small_ReplicaDrawsPair, ::testing::ValuesIn(draw_pairs()),
                         [](const ::testing::TestParamInfo<DrawPair>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });
