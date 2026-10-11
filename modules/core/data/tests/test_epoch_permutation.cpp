/**************************************************************************/
/* test_epoch_permutation.cpp                                             */
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
 * @file test_epoch_permutation.cpp
 * @brief Tests the order of an epoch: every sample once, one order per seed and epoch.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/data/epoch_permutation.hpp>

using namespace SushiAI;

namespace
{
    /** @brief The seed of every order below. */
    constexpr std::uint64_t K_SEED = 42;

    /** @brief Returns the sample at every position of one order. */
    std::vector<std::uint64_t> order_of(std::uint64_t seed, std::int64_t epoch,
                                        std::uint64_t count)
    {
        const Data::EpochPermutation permutation(seed, epoch, count);

        std::vector<std::uint64_t> order;
        for (std::uint64_t position = 0; position < count; ++position)
            order.push_back(permutation.at(position));

        return order;
    }
} // namespace

/** @brief Tests that an order holds every sample once, for counts on both sides of 4^k. */
TEST(Unit_EpochPermutation, EverySampleIsReadOnce)
{
    for (std::uint64_t count : {1u, 2u, 3u, 7u, 16u, 17u, 1000u, 4097u})
    {
        for (std::int64_t epoch : {0, 1, 5})
        {
            std::vector<bool> is_seen(count, false);
            for (std::uint64_t sample : order_of(K_SEED, epoch, count))
            {
                ASSERT_LT(sample, count);
                EXPECT_FALSE(is_seen[sample]) << "count " << count << ", epoch " << epoch;
                is_seen[sample] = true;
            }
        }
    }
}

/** @brief Tests the values of three orders, which a change of the mapping would move. */
TEST(Unit_EpochPermutation, AnOrderIsAFunctionOfTheSeedAndTheEpoch)
{
    const std::vector<std::uint64_t> first = {4, 9, 5, 2, 1, 3, 8, 7, 6, 0};
    const std::vector<std::uint64_t> next_epoch = {2, 5, 9, 1, 4, 3, 6, 7, 8, 0};
    const std::vector<std::uint64_t> other_seed = {5, 4, 1, 2, 3, 9, 0, 6, 7, 8};

    EXPECT_EQ(order_of(K_SEED, 0, 10), first);
    EXPECT_EQ(order_of(K_SEED, 0, 10), first);
    EXPECT_EQ(order_of(K_SEED, 1, 10), next_epoch);
    EXPECT_EQ(order_of(7, 0, 10), other_seed);
}

/** @brief Tests that count() returns the count the order was built over. */
TEST(Unit_EpochPermutation, CountIsTheCountItWasBuiltOver)
{
    EXPECT_EQ(Data::EpochPermutation(K_SEED, 0, 17).count(), 17u);
}
