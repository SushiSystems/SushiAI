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
 * @file test_epoch_permutation.cpp
 * @brief Tests the order of an epoch: every sample once, one order per seed and epoch.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/data/epoch_permutation.hpp>

using namespace SushiAI;

namespace
{
    /** @brief The seed of every order below. */
    constexpr std::uint64_t K_SEED = 42;

    /** @brief A seed other than K_SEED. */
    constexpr std::uint64_t K_OTHER_SEED = 7;

    /** @brief The sample count of the orders compared below; 10! orders make a tie improbable. */
    constexpr std::uint64_t K_COMPARED_COUNT = 10;

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

    /** @brief Returns whether @p order holds each sample below @p count exactly once. */
    ::testing::AssertionResult is_a_bijection(const std::vector<std::uint64_t>& order,
                                              std::uint64_t count)
    {
        if (order.size() != count)
            return ::testing::AssertionFailure() << order.size() << " positions of " << count;

        std::vector<bool> is_seen(count, false);
        for (std::uint64_t sample : order)
        {
            if (sample >= count)
                return ::testing::AssertionFailure() << "sample " << sample << " is out of range";
            if (is_seen[sample])
                return ::testing::AssertionFailure() << "sample " << sample << " is read twice";
            is_seen[sample] = true;
        }

        return ::testing::AssertionSuccess();
    }

    /** @brief Runs each case with a sample count and an epoch. */
    class Small_EpochPermutationCover
        : public ::testing::TestWithParam<std::tuple<std::uint64_t, std::int64_t>>
    {

    };
} // namespace

/** @brief Tests that an order holds every sample once, for counts on both sides of 4^k. */
TEST_P(Small_EpochPermutationCover, EverySampleIsReadOnce)
{
    const std::uint64_t count = std::get<0>(GetParam());
    const std::int64_t epoch = std::get<1>(GetParam());

    EXPECT_TRUE(is_a_bijection(order_of(K_SEED, epoch, count), count));
}

INSTANTIATE_TEST_SUITE_P(CountsAndEpochs, Small_EpochPermutationCover,
                         ::testing::Combine(::testing::Values(1u, 2u, 3u, 7u, 16u, 17u, 1000u,
                                                              4097u),
                                            ::testing::Values(0, 1, 5)));

/** @brief Tests that one seed and epoch give the same order on every read. */
TEST(Small_EpochPermutation, AnOrderRepeatsForTheSameSeedAndEpoch)
{
    EXPECT_EQ(order_of(K_SEED, 0, K_COMPARED_COUNT), order_of(K_SEED, 0, K_COMPARED_COUNT));
}

/** @brief Tests that another epoch gives another order. */
TEST(Small_EpochPermutation, AnotherEpochGivesAnotherOrder)
{
    EXPECT_NE(order_of(K_SEED, 1, K_COMPARED_COUNT), order_of(K_SEED, 0, K_COMPARED_COUNT));
}

/** @brief Tests that another seed gives another order. */
TEST(Small_EpochPermutation, AnotherSeedGivesAnotherOrder)
{
    EXPECT_NE(order_of(K_OTHER_SEED, 0, K_COMPARED_COUNT), order_of(K_SEED, 0, K_COMPARED_COUNT));
}

/** @brief Tests that an order moves samples: it is not the order in which they are stored. */
TEST(Small_EpochPermutation, AnOrderIsNotTheOrderOfStorage)
{
    std::vector<std::uint64_t> storage(K_COMPARED_COUNT);
    std::iota(storage.begin(), storage.end(), std::uint64_t{0});

    EXPECT_NE(order_of(K_SEED, 0, K_COMPARED_COUNT), storage);
}
