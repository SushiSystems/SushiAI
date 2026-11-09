/**************************************************************************/
/* test_anchor_layout.cpp                                                 */
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
 * @file test_anchor_layout.cpp
 * @brief Tests the check of an anchor layout and the IoU of a size with no area.
 * @author Mustafa Garip
 */

#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;
    using SushiAI::Vision::Detect::Anchor::AnchorSize;
    using SushiAI::Vision::Detect::Anchor::shape_iou;
    using SushiAI::Vision::Detect::Anchor::validate_anchor_layout;

    /** @brief Returns six anchors of rising area and two heads of three slots each. */
    AnchorLayout make_two_head_layout()
    {
        AnchorLayout layout;
        layout.anchors = {{0.02f, 0.03f}, {0.05f, 0.06f}, {0.09f, 0.14f},
                          {0.2f, 0.2f},   {0.3f, 0.4f},   {0.8f, 0.7f}};
        layout.head_anchors = {{3, 4, 5}, {0, 1, 2}};
        return layout;
    }
} // namespace

/** @brief Tests that shape_iou with a zero area is 0, and 0 for two zero areas. */
TEST(Small_DetectAnchorLayout, ShapeIoUWithAZeroAreaIsZero)
{
    const AnchorSize flat{0.0f, 0.3f};
    const AnchorSize size{0.2f, 0.2f};

    EXPECT_EQ(shape_iou(flat, size), 0.0f);
    EXPECT_EQ(shape_iou(size, flat), 0.0f);
    EXPECT_EQ(shape_iou(flat, flat), 0.0f);
    EXPECT_EQ(shape_iou(AnchorSize{}, AnchorSize{}), 0.0f);
}

/** @brief Tests that a layout of six anchors split over two heads validates. */
TEST(Small_DetectAnchorLayout, AcceptsOneHeadPerGroupOfAnchors)
{
    const SushiAI::Status status = validate_anchor_layout(make_two_head_layout());

    EXPECT_TRUE(status.is_ok()) << status.message();
}

/** @brief Tests that two heads may share an anchor. */
TEST(Small_DetectAnchorLayout, AcceptsASharedAnchor)
{
    AnchorLayout layout = make_two_head_layout();
    layout.head_anchors = {{0, 1, 2, 3}, {3, 4, 5}};

    const SushiAI::Status status = validate_anchor_layout(layout);

    EXPECT_TRUE(status.is_ok()) << status.message();
}

/** @brief Tests that an anchor no head uses is accepted. */
TEST(Small_DetectAnchorLayout, AcceptsAnAnchorNoHeadUses)
{
    AnchorLayout layout = make_two_head_layout();
    layout.head_anchors = {{3, 4, 5}, {1, 2}};

    const SushiAI::Status status = validate_anchor_layout(layout);

    EXPECT_TRUE(status.is_ok()) << status.message();
}

/** @brief Tests that a head index equal to the anchor count is refused. */
TEST(Small_DetectAnchorLayout, RefusesAnIndexPastTheAnchors)
{
    AnchorLayout layout = make_two_head_layout();
    layout.head_anchors[1][2] = 6;

    EXPECT_EQ(validate_anchor_layout(layout).code(), StatusCode::InvalidDescription);
}

/** @brief Tests that a negative head index is refused. */
TEST(Small_DetectAnchorLayout, RefusesANegativeIndex)
{
    AnchorLayout layout = make_two_head_layout();
    layout.head_anchors[0][0] = -1;

    EXPECT_EQ(validate_anchor_layout(layout).code(), StatusCode::InvalidDescription);
}

/** @brief Tests that a head that uses no anchor is refused. */
TEST(Small_DetectAnchorLayout, RefusesAnEmptyHead)
{
    AnchorLayout layout = make_two_head_layout();
    layout.head_anchors[1].clear();

    EXPECT_EQ(validate_anchor_layout(layout).code(), StatusCode::InvalidDescription);
}

/** @brief Tests that a layout without anchors and one without heads are refused. */
TEST(Small_DetectAnchorLayout, RefusesALayoutWithoutAnchorsOrWithoutHeads)
{
    AnchorLayout no_anchors = make_two_head_layout();
    no_anchors.anchors.clear();
    AnchorLayout no_heads = make_two_head_layout();
    no_heads.head_anchors.clear();

    EXPECT_EQ(validate_anchor_layout(no_anchors).code(), StatusCode::InvalidDescription);
    EXPECT_EQ(validate_anchor_layout(no_heads).code(), StatusCode::InvalidDescription);
}

/** @brief Tests that an anchor with a zero, a NaN or an infinite side is refused. */
TEST(Small_DetectAnchorLayout, RefusesAnAnchorWithoutAFinitePositiveSide)
{
    AnchorLayout zero_side = make_two_head_layout();
    zero_side.anchors[2].width = 0.0f;
    AnchorLayout nan_side = make_two_head_layout();
    nan_side.anchors[4].height = std::numeric_limits<float>::quiet_NaN();
    AnchorLayout infinite_side = make_two_head_layout();
    infinite_side.anchors[0].width = std::numeric_limits<float>::infinity();

    EXPECT_EQ(validate_anchor_layout(zero_side).code(), StatusCode::InvalidDescription);
    EXPECT_EQ(validate_anchor_layout(nan_side).code(), StatusCode::InvalidDescription);
    EXPECT_EQ(validate_anchor_layout(infinite_side).code(), StatusCode::InvalidDescription);
}
