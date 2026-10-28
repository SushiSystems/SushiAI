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
 * @file test_float_order.cpp
 * @brief Tests the finiteness scan and the NaN-tolerant descending order.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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
} // namespace

/** @brief Tests that an empty span has no bad value. */
TEST(Small_DetectFloatOrder, FirstNonFiniteOfNoValuesIsMinusOne)
{
    const std::vector<float> values;

    EXPECT_EQ(first_non_finite(values), -1);
}

/** @brief Tests that finite values, the extremes of float among them, report no bad index. */
TEST(Small_DetectFloatOrder, FirstNonFiniteOfFiniteValuesIsMinusOne)
{
    const std::vector<float> values{0.0f, -2.5f, std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::lowest(),
                                    std::numeric_limits<float>::denorm_min()};

    EXPECT_EQ(first_non_finite(values), -1);
}

/** @brief Tests that a NaN, a positive and a negative infinity are each named by their index. */
TEST(Small_DetectFloatOrder, FirstNonFiniteNamesANaNAndBothInfinities)
{
    EXPECT_EQ(first_non_finite(std::vector<float>{1.0f, K_NAN, 2.0f}), 1);
    EXPECT_EQ(first_non_finite(std::vector<float>{1.0f, 2.0f, K_INF}), 2);
    EXPECT_EQ(first_non_finite(std::vector<float>{-K_INF, 1.0f}), 0);
}

/** @brief Tests that the first bad value wins when several follow it. */
TEST(Small_DetectFloatOrder, FirstNonFiniteNamesTheFirstOfSeveral)
{
    const std::vector<float> values{0.0f, K_INF, K_NAN, -K_INF};

    EXPECT_EQ(first_non_finite(values), 1);
}

/** @brief Tests that the larger of two numbers ranks above the smaller and never the reverse. */
TEST(Small_DetectFloatOrder, RanksAboveOrdersNumbersLargestFirst)
{
    EXPECT_TRUE(ranks_above(2.0f, 1.0f));
    EXPECT_FALSE(ranks_above(1.0f, 2.0f));
    EXPECT_TRUE(ranks_above(0.0f, -1.0f));
    EXPECT_TRUE(ranks_above(K_INF, 1.0f));
    EXPECT_TRUE(ranks_above(1.0f, -K_INF));
}

/** @brief Tests that equal numbers rank above neither way, which makes the relation strict. */
TEST(Small_DetectFloatOrder, RanksAboveIsFalseBothWaysForEqualNumbers)
{
    EXPECT_FALSE(ranks_above(1.5f, 1.5f));
    EXPECT_FALSE(ranks_above(0.0f, -0.0f));
    EXPECT_FALSE(ranks_above(-0.0f, 0.0f));
    EXPECT_FALSE(ranks_above(K_INF, K_INF));
}

/** @brief Tests that a NaN ranks below every number and equal to another NaN. */
TEST(Small_DetectFloatOrder, RanksAboveSendsANaNBelowEveryNumber)
{
    EXPECT_FALSE(ranks_above(K_NAN, 1.0f));
    EXPECT_TRUE(ranks_above(1.0f, K_NAN));
    EXPECT_TRUE(ranks_above(-K_INF, K_NAN));
    EXPECT_FALSE(ranks_above(K_NAN, -K_INF));
    EXPECT_FALSE(ranks_above(K_NAN, K_NAN));
}

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
