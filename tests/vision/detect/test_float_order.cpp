/**************************************************************************/
/* test_float_order.cpp                                                   */
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
 * @file test_float_order.cpp
 * @brief Tests the finiteness scan and the NaN-tolerant descending order.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/float_order.hpp>

namespace
{
    using SushiAI::Vision::Detect::first_non_finite;
    using SushiAI::Vision::Detect::ranks_above;

    /** @brief Holds a quiet NaN. */
    const float K_NAN = std::numeric_limits<float>::quiet_NaN();

    /** @brief Holds a positive infinity. */
    const float K_INF = std::numeric_limits<float>::infinity();

    /** @brief Holds a list of values, the index of its first bad value and the name of the row. */
    struct ScanRow
    {
        std::string name;          /**< The row's name. */
        std::vector<float> values; /**< The values to scan. */
        std::int64_t expected;     /**< The index first_non_finite returns, or -1. */
    };

    /** @brief Holds two values, whether the first ranks above the second and the row's name. */
    struct RankRow
    {
        std::string name; /**< The row's name. */
        float first;      /**< The value on the left of the relation. */
        float second;     /**< The value on the right of the relation. */
        bool expected;    /**< Whether the first ranks above the second. */
    };

    /** @brief Runs first_non_finite over one row. */
    class Small_DetectFloatOrderScan : public ::testing::TestWithParam<ScanRow>
    {

    };

    /** @brief Runs ranks_above over one row. */
    class Small_DetectFloatOrderRank : public ::testing::TestWithParam<RankRow>
    {

    };

    /** @brief Names a scan row by its name field. */
    std::string scan_name(const ::testing::TestParamInfo<ScanRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a rank row by its name field. */
    std::string rank_name(const ::testing::TestParamInfo<RankRow>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that the scan names the index of the first bad value, or -1 when there is none. */
TEST_P(Small_DetectFloatOrderScan, NamesTheIndexOfTheFirstBadValue)
{
    EXPECT_EQ(first_non_finite(GetParam().values), GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DetectFloatOrderScan,
    ::testing::Values(
        ScanRow{"NoValuesGivesMinusOne", {}, -1},
        ScanRow{"FiniteExtremesGiveMinusOne",
                {0.0f, -2.5f, std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::lowest(), std::numeric_limits<float>::denorm_min()},
                -1},
        ScanRow{"ANaNIsNamed", {1.0f, K_NAN, 2.0f}, 1},
        ScanRow{"APositiveInfinityIsNamed", {1.0f, 2.0f, K_INF}, 2},
        ScanRow{"ANegativeInfinityIsNamed", {-K_INF, 1.0f}, 0}),
    scan_name);

/** @brief Tests that the first bad value wins when several follow it. */
TEST(Small_DetectFloatOrder, FirstNonFiniteNamesTheFirstOfSeveral)
{
    const std::vector<float> values{0.0f, K_INF, K_NAN, -K_INF};

    EXPECT_EQ(first_non_finite(values), 1);
}

/**
 * @brief Tests the relation on numbers, equal numbers and NaNs.
 *
 * The larger of two numbers ranks above the smaller and never the reverse; equal numbers rank
 * above neither way, which makes the relation strict; a NaN ranks below every number and equal to
 * another NaN.
 */
TEST_P(Small_DetectFloatOrderRank, AnswersAsTheRowStates)
{
    EXPECT_EQ(ranks_above(GetParam().first, GetParam().second), GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DetectFloatOrderRank,
    ::testing::Values(
        RankRow{"TwoRanksAboveOne", 2.0f, 1.0f, true},
        RankRow{"OneDoesNotRankAboveTwo", 1.0f, 2.0f, false},
        RankRow{"ZeroRanksAboveMinusOne", 0.0f, -1.0f, true},
        RankRow{"InfinityRanksAboveOne", K_INF, 1.0f, true},
        RankRow{"OneRanksAboveMinusInfinity", 1.0f, -K_INF, true},
        RankRow{"EqualNumbersRankAboveNeitherWay", 1.5f, 1.5f, false},
        RankRow{"ZeroDoesNotRankAboveMinusZero", 0.0f, -0.0f, false},
        RankRow{"MinusZeroDoesNotRankAboveZero", -0.0f, 0.0f, false},
        RankRow{"InfinityDoesNotRankAboveInfinity", K_INF, K_INF, false},
        RankRow{"ANaNDoesNotRankAboveANumber", K_NAN, 1.0f, false},
        RankRow{"ANumberRanksAboveANaN", 1.0f, K_NAN, true},
        RankRow{"MinusInfinityRanksAboveANaN", -K_INF, K_NAN, true},
        RankRow{"ANaNDoesNotRankAboveMinusInfinity", K_NAN, -K_INF, false},
        RankRow{"ANaNDoesNotRankAboveANaN", K_NAN, K_NAN, false}),
    rank_name);

/** @brief Tests that sorting with the relation puts the numbers first, descending, then NaNs. */
TEST(Small_DetectFloatOrder, SortingWithRanksAboveLeavesTheNaNsLast)
{
    std::vector<float> values{K_NAN, 1.0f, K_NAN, 3.0f, -1.0f, 2.0f};

    std::sort(values.begin(), values.end(), ranks_above);

    ASSERT_EQ(values.size(), 6u);
    EXPECT_EQ(values[0], 3.0f);
    EXPECT_EQ(values[1], 2.0f);
    EXPECT_EQ(values[2], 1.0f);
    EXPECT_EQ(values[3], -1.0f);
    EXPECT_TRUE(std::isnan(values[4]));
    EXPECT_TRUE(std::isnan(values[5]));
}
