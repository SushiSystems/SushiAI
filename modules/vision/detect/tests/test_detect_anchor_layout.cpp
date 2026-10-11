/**************************************************************************/
/* test_detect_anchor_layout.cpp                                          */
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
 * @file test_detect_anchor_layout.cpp
 * @brief Tests the IoU of two anchor sizes and the check of an anchor layout on hand cases.
 * @author Mustafa Garip
 */

#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;
    using SushiAI::Vision::Detect::Anchor::AnchorSize;
    using SushiAI::Vision::Detect::Anchor::shape_iou;
    using SushiAI::Vision::Detect::Anchor::validate_anchor_layout;

    /** @brief Holds the absolute tolerance of an IoU whose terms float32 rounds once each. */
    constexpr float K_IOU_TOLERANCE = 1e-6f;

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

/** @brief Tests that shape_iou of a size with itself is 1. */
TEST(Unit_DetectAnchorLayout, ShapeIoUOfASizeWithItselfIsOne)
{
    const AnchorSize size{0.3f, 0.7f};

    EXPECT_NEAR(shape_iou(size, size), 1.0f, K_IOU_TOLERANCE);
}

/** @brief Tests that shape_iou of a contained size is the ratio of the two areas. */
TEST(Unit_DetectAnchorLayout, ShapeIoUOfAContainedSizeIsTheRatioOfTheAreas)
{
    const AnchorSize outer{0.2f, 0.4f};
    const AnchorSize inner{0.1f, 0.2f};

    // The inner area is 0.02 and the union is the outer area, 0.08.
    EXPECT_NEAR(shape_iou(outer, inner), 0.25f, K_IOU_TOLERANCE);
    EXPECT_NEAR(shape_iou(inner, outer), 0.25f, K_IOU_TOLERANCE);
}

/** @brief Tests that shape_iou of a wide and a tall size counts the rectangle they share. */
TEST(Unit_DetectAnchorLayout, ShapeIoUOfCrossedSizesCountsTheSharedRectangle)
{
    const AnchorSize wide{0.4f, 0.1f};
    const AnchorSize tall{0.1f, 0.4f};

    // The shared rectangle is 0.1 by 0.1 and the union is 0.04 + 0.04 - 0.01.
    EXPECT_NEAR(shape_iou(wide, tall), 1.0f / 7.0f, K_IOU_TOLERANCE);
}

/** @brief Tests that shape_iou with a zero area is 0, and 0 for two zero areas. */
TEST(Unit_DetectAnchorLayout, ShapeIoUWithAZeroAreaIsZero)
{
    const AnchorSize flat{0.0f, 0.3f};
    const AnchorSize size{0.2f, 0.2f};

    EXPECT_EQ(shape_iou(flat, size), 0.0f);
    EXPECT_EQ(shape_iou(size, flat), 0.0f);
    EXPECT_EQ(shape_iou(flat, flat), 0.0f);
    EXPECT_EQ(shape_iou(AnchorSize{}, AnchorSize{}), 0.0f);
}

/** @brief Tests that a layout of six anchors split over two heads validates. */
TEST(Unit_DetectAnchorLayout, AcceptsOneHeadPerGroupOfAnchors)
{
    const SushiAI::Status status = validate_anchor_layout(make_two_head_layout());

    EXPECT_TRUE(status.is_ok()) << status.message();
}

/** @brief Tests that two heads may share an anchor and an anchor may belong to no head. */
TEST(Unit_DetectAnchorLayout, AcceptsASharedAnchorAndAnAnchorNoHeadUses)
{
    AnchorLayout layout = make_two_head_layout();
    layout.head_anchors = {{3, 4, 5}, {1, 2, 3}};

    const SushiAI::Status status = validate_anchor_layout(layout);

    EXPECT_TRUE(status.is_ok()) << status.message();
}

/** @brief Tests that a head index equal to the anchor count is refused. */
TEST(Unit_DetectAnchorLayout, RefusesAnIndexPastTheAnchors)
{
    AnchorLayout layout = make_two_head_layout();
    layout.head_anchors[1][2] = 6;

    EXPECT_EQ(validate_anchor_layout(layout).code(), StatusCode::InvalidDescription);
}

/** @brief Tests that a negative head index is refused. */
TEST(Unit_DetectAnchorLayout, RefusesANegativeIndex)
{
    AnchorLayout layout = make_two_head_layout();
    layout.head_anchors[0][0] = -1;

    EXPECT_EQ(validate_anchor_layout(layout).code(), StatusCode::InvalidDescription);
}

/** @brief Tests that a head that uses no anchor is refused. */
TEST(Unit_DetectAnchorLayout, RefusesAnEmptyHead)
{
    AnchorLayout layout = make_two_head_layout();
    layout.head_anchors[1].clear();

    EXPECT_EQ(validate_anchor_layout(layout).code(), StatusCode::InvalidDescription);
}

/** @brief Tests that a layout without anchors and one without heads are refused. */
TEST(Unit_DetectAnchorLayout, RefusesALayoutWithoutAnchorsOrWithoutHeads)
{
    AnchorLayout no_anchors = make_two_head_layout();
    no_anchors.anchors.clear();
    AnchorLayout no_heads = make_two_head_layout();
    no_heads.head_anchors.clear();

    EXPECT_EQ(validate_anchor_layout(no_anchors).code(), StatusCode::InvalidDescription);
    EXPECT_EQ(validate_anchor_layout(no_heads).code(), StatusCode::InvalidDescription);
}

/** @brief Tests that an anchor with a zero, a NaN or an infinite side is refused. */
TEST(Unit_DetectAnchorLayout, RefusesAnAnchorWithoutAFinitePositiveSide)
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
