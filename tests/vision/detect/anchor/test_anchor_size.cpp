/**************************************************************************/
/* test_anchor_size.cpp                                                   */
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
 * @file test_anchor_size.cpp
 * @brief Tests the default anchor size and the bounds and symmetry of the IoU of two sizes.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <SushiAI/vision/detect/anchor/seeded_sequence.hpp>

namespace
{
    using SushiAI::Vision::Detect::Anchor::AnchorSize;
    using SushiAI::Vision::Detect::Anchor::shape_iou;
    using SushiAI::Vision::Detect::Anchor::Detail::SeededSequence;

    /** @brief Holds the seed of the random sizes the bound case draws. */
    constexpr std::uint64_t K_SEED = 20261016;

    /** @brief Holds the number of size pairs the bound case draws. */
    constexpr int K_PAIRS = 256;

    /** @brief Returns a size whose sides are the next two unit draws of @p sequence. */
    AnchorSize draw_size(SeededSequence& sequence)
    {
        const float width = static_cast<float>(sequence.next_unit());
        const float height = static_cast<float>(sequence.next_unit());
        return AnchorSize{width, height};
    }
} // namespace

/** @brief Tests that a default size has a width and a height of zero. */
TEST(Small_DetectAnchorSize, ADefaultSizeHasNoExtent)
{
    const AnchorSize size;

    EXPECT_EQ(size.width, 0.0f);
    EXPECT_EQ(size.height, 0.0f);
}

/** @brief Tests that the IoU of random sizes is the same both ways and lies in [0, 1]. */
TEST(Small_DetectAnchorSize, TheIoUIsSymmetricAndBetweenZeroAndOne)
{
    SeededSequence sequence(K_SEED);

    for (int pair = 0; pair < K_PAIRS; ++pair)
    {
        const AnchorSize a = draw_size(sequence);
        const AnchorSize b = draw_size(sequence);
        const float forward = shape_iou(a, b);

        // Both orders round the same products and sums, so they agree bit for bit.
        EXPECT_EQ(forward, shape_iou(b, a)) << "pair " << pair;
        EXPECT_GE(forward, 0.0f) << "pair " << pair;
        EXPECT_LE(forward, 1.0f) << "pair " << pair;
    }
}

/** @brief Tests that a side that is not a number gives an IoU of 0 in either position. */
TEST(Small_DetectAnchorSize, ASideThatIsNotANumberGivesNoOverlap)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const AnchorSize broken{nan, 0.5f};
    const AnchorSize size{0.25f, 0.5f};

    EXPECT_EQ(shape_iou(broken, size), 0.0f);
    EXPECT_EQ(shape_iou(size, broken), 0.0f);
}
