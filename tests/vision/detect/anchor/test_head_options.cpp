/**************************************************************************/
/* test_head_options.cpp                                                  */
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
 * @file test_head_options.cpp
 * @brief Tests the options of an anchor head: its channel and row counts and what it refuses.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_decode.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>

namespace
{
    using SushiAI::StatusCode;

    using SushiAI::Vision::Detect::Anchor::AnchorLayout;

    using SushiAI::Vision::Detect::Anchor::count_head_channels;

    using SushiAI::Vision::Detect::Anchor::count_head_rows;

    using SushiAI::Vision::Detect::Anchor::HeadCoding;

    using SushiAI::Vision::Detect::Anchor::HeadOptions;

    using SushiAI::Vision::Detect::Anchor::validate_head;

    /** @brief Returns a layout of one anchor that its one head uses. */
    AnchorLayout make_single_anchor_layout(float width, float height)
    {
        AnchorLayout layout;
        layout.anchors = {{width, height}};
        layout.head_anchors = {{0}};
        return layout;
    }

    /** @brief Returns head options of @p classes classes under @p coding and @p scale_xy. */
    HeadOptions make_options(std::int64_t classes, HeadCoding coding, float scale_xy)
    {
        HeadOptions options;
        options.num_classes = classes;
        options.coding = coding;
        options.scale_xy = scale_xy;
        return options;
    }

} // namespace

/** @brief Tests that the channel and row counts use the slots of the head asked for. */
TEST(Unit_DetectAnchorDecode, CountsChannelsAndRowsFromTheHeadsOwnAnchors)
{
    AnchorLayout layout;
    layout.anchors = {{0.1f, 0.1f}, {0.2f, 0.2f}, {0.3f, 0.3f}, {0.4f, 0.4f}, {0.5f, 0.5f}};
    layout.head_anchors = {{2, 3, 4}, {0, 1}};
    const HeadOptions options = make_options(80, HeadCoding::Logit, 1.0f);

    EXPECT_EQ(count_head_channels(options, layout, 0), 255);
    EXPECT_EQ(count_head_channels(options, layout, 1), 170);
    EXPECT_EQ(count_head_rows({4, 13, 13}, layout, 0), 507);
    EXPECT_EQ(count_head_rows({4, 26, 26}, layout, 1), 1352);
}

/** @brief Tests that validate_head accepts a head the layout holds. */
TEST(Unit_DetectAnchorDecode, ValidatesAHeadTheLayoutHolds)
{
    const AnchorLayout layout = make_single_anchor_layout(0.3f, 0.45f);

    const SushiAI::Status status =
        validate_head(make_options(80, HeadCoding::Activated, 2.0f), layout, 0);

    EXPECT_TRUE(status.is_ok()) << status.message();
}

/** @brief Tests that validate_head refuses a head index below 0 or past the heads. */
TEST(Unit_DetectAnchorDecode, RefusesAHeadIndexOutsideTheLayout)
{
    const AnchorLayout layout = make_single_anchor_layout(0.3f, 0.45f);
    const HeadOptions options = make_options(80, HeadCoding::Logit, 1.0f);

    EXPECT_EQ(validate_head(options, layout, 1).code(), StatusCode::InvalidDescription);
    EXPECT_EQ(validate_head(options, layout, -1).code(), StatusCode::InvalidDescription);
}

/** @brief Tests that validate_head refuses a class count of 0. */
TEST(Unit_DetectAnchorDecode, RefusesAHeadWithoutClasses)
{
    const AnchorLayout layout = make_single_anchor_layout(0.3f, 0.45f);

    EXPECT_EQ(validate_head(make_options(0, HeadCoding::Logit, 1.0f), layout, 0).code(),
              StatusCode::InvalidDescription);
}

/** @brief Tests that validate_head refuses a scale_xy of 0 and of NaN. */
TEST(Unit_DetectAnchorDecode, RefusesAScaleThatIsNotFiniteAndPositive)
{
    const AnchorLayout layout = make_single_anchor_layout(0.3f, 0.45f);
    const float not_a_number = std::numeric_limits<float>::quiet_NaN();

    EXPECT_EQ(validate_head(make_options(1, HeadCoding::Logit, 0.0f), layout, 0).code(),
              StatusCode::InvalidDescription);
    EXPECT_EQ(validate_head(make_options(1, HeadCoding::Logit, not_a_number), layout, 0).code(),
              StatusCode::InvalidDescription);
}

/** @brief Tests that validate_head returns the layout's own refusal. */
TEST(Unit_DetectAnchorDecode, RefusesAHeadWhoseLayoutDoesNotValidate)
{
    AnchorLayout layout = make_single_anchor_layout(0.3f, 0.45f);
    layout.head_anchors[0][0] = 3;

    EXPECT_EQ(validate_head(make_options(1, HeadCoding::Logit, 1.0f), layout, 0).code(),
              StatusCode::InvalidDescription);
}
