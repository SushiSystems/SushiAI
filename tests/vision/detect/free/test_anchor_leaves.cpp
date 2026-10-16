/**************************************************************************/
/* test_anchor_leaves.cpp                                                 */
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
 * @file test_anchor_leaves.cpp
 * @brief Tests the anchor leaves of a one-image batch and their rebuild for a smaller batch.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/free/anchor_leaves.hpp>
#include <SushiAI/vision/detect/free/anchors.hpp>

namespace
{
    using SushiAI::Vision::Detect::Free::AnchorLeaves;
    using SushiAI::Vision::Detect::Free::build_anchor_leaves;
    using SushiAI::Vision::Detect::Free::FeatureLevel;

    /** @brief Holds the tolerance of a cell centre or a stride, each a multiple of 0.5. */
    constexpr float K_EXACT = 0.0f;

    /** @brief Expects @p actual to hold @p expected, value for value, within K_EXACT. */
    void expect_values(const std::vector<float>& actual, const std::vector<float>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(actual[i], expected[i], K_EXACT) << "value " << i;
    }
} // namespace

/** @brief Tests that one image of a 1 by 2 grid holds two cell centres and their stride. */
TEST(Small_DetectAnchorLeaves, OneImageHoldsTheCellCentresAndTheirStride)
{
    const std::vector<FeatureLevel> levels{{1, 2, 8.0f}};
    AnchorLeaves leaves;

    build_anchor_leaves(levels, 1, leaves);

    expect_values(leaves.points, {0.5f, 0.5f, 1.5f, 0.5f});
    expect_values(leaves.strides, std::vector<float>(8, 8.0f));
}

/** @brief Tests that rebuilding used leaves for a smaller batch shrinks them to that batch. */
TEST(Small_DetectAnchorLeaves, RebuildingForASmallerBatchShrinksTheLeaves)
{
    const std::vector<FeatureLevel> levels{{2, 2, 8.0f}, {1, 1, 16.0f}};
    AnchorLeaves leaves;
    build_anchor_leaves(levels, 3, leaves);
    ASSERT_EQ(leaves.points.size(), 3u * 5u * 2u);

    build_anchor_leaves(levels, 1, leaves);

    // Five anchors of one image: two values per point and four per stride row.
    EXPECT_EQ(leaves.points.size(), 5u * 2u);
    EXPECT_EQ(leaves.strides.size(), 5u * 4u);
}
