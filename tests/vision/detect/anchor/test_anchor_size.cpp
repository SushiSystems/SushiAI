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
 * @file test_anchor_size.cpp
 * @brief Tests the IoU of two anchor sizes: hand values, order, bounds and a side not a number.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <support/splitmix64.hpp>
#include <support/tolerance.hpp>

namespace
{
    using SushiAI::Vision::Detect::Anchor::AnchorSize;
    using SushiAI::Vision::Detect::Anchor::shape_iou;

    /** @brief Holds the seed of the random sizes the property cases draw. */
    constexpr std::uint64_t K_SEED = 20261016;

    /** @brief Holds the number of size pairs the property cases draw. */
    constexpr std::size_t K_PAIRS = 256;

    /** @brief Holds 2^53, the divisor that turns the upper 53 bits of a number into a unit. */
    constexpr double K_UNIT_DIVISOR = 9007199254740992.0;

    /**
     * @brief Holds the absolute tolerance of an IoU whose terms float32 rounds once each: 32 times
     *        the float unit roundoff, the smallest power of two not tighter than the earlier 1e-6f.
     */
    constexpr float K_IOU_TOLERANCE =
        static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());

    /** @brief Holds the IoU of every drawn pair, taken in both orders. */
    struct DrawnIoUs
    {
        std::vector<float> forward;  /**< The IoU of a with b, pair by pair. */
        std::vector<float> backward; /**< The IoU of b with a, pair by pair. */
    };

    /** @brief Returns draw @p index of the fixed sequence as a float in [0, 1). */
    float draw_side(std::uint64_t index)
    {
        return static_cast<float>(
            static_cast<double>(TestSupport::reference_splitmix64(K_SEED, index) >> 11) /
            K_UNIT_DIVISOR);
    }

    /** @brief Returns the IoU of K_PAIRS pairs of drawn sizes, each pair in both orders. */
    DrawnIoUs draw_ious()
    {
        DrawnIoUs ious;
        for (std::uint64_t pair = 0; pair < K_PAIRS; ++pair)
        {
            const AnchorSize a{draw_side(4 * pair), draw_side(4 * pair + 1)};
            const AnchorSize b{draw_side(4 * pair + 2), draw_side(4 * pair + 3)};
            ious.forward.push_back(shape_iou(a, b));
            ious.backward.push_back(shape_iou(b, a));
        }
        return ious;
    }
} // namespace

/** @brief Tests that shape_iou of a size with itself is 1. */
TEST(Small_DetectAnchorSize, ASizeWithItselfHasAnIoUOfOne)
{
    const AnchorSize size{0.3f, 0.7f};

    EXPECT_NEAR(shape_iou(size, size), 1.0f, K_IOU_TOLERANCE);
}

/** @brief Tests that shape_iou of a contained size is the ratio of the two areas. */
TEST(Small_DetectAnchorSize, AContainedSizeHasTheRatioOfTheAreasAsItsIoU)
{
    const AnchorSize outer{0.2f, 0.4f};
    const AnchorSize inner{0.1f, 0.2f};

    // The inner area is 0.02 and the union is the outer area, 0.08.
    EXPECT_NEAR(shape_iou(outer, inner), 0.25f, K_IOU_TOLERANCE);
    EXPECT_NEAR(shape_iou(inner, outer), 0.25f, K_IOU_TOLERANCE);
}

/** @brief Tests that shape_iou of a wide and a tall size counts the rectangle they share. */
TEST(Small_DetectAnchorSize, CrossedSizesCountTheSharedRectangle)
{
    const AnchorSize wide{0.4f, 0.1f};
    const AnchorSize tall{0.1f, 0.4f};

    // The shared rectangle is 0.1 by 0.1 and the union is 0.04 + 0.04 - 0.01.
    EXPECT_NEAR(shape_iou(wide, tall), 1.0f / 7.0f, K_IOU_TOLERANCE);
}

/** @brief Tests that the IoU of random sizes is the same both ways. */
TEST(Small_DetectAnchorSize, TheIoUIsTheSameInBothOrders)
{
    const DrawnIoUs ious = draw_ious();

    // Both orders round the same products and sums, so they agree bit for bit.
    EXPECT_EQ(ious.forward, ious.backward);
}

/** @brief Tests that the IoU of random sizes lies in [0, 1]. */
TEST(Small_DetectAnchorSize, TheIoULiesBetweenZeroAndOne)
{
    const DrawnIoUs ious = draw_ious();

    EXPECT_THAT(ious.forward, ::testing::Each(::testing::AllOf(::testing::Ge(0.0f),
                                                               ::testing::Le(1.0f))));
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
