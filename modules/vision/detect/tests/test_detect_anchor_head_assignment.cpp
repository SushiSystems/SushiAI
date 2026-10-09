/**************************************************************************/
/* test_detect_anchor_head_assignment.cpp                                 */
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
 * @file test_detect_anchor_head_assignment.cpp
 * @brief Tests the assignment of anchors to heads against the masks of six cfgs and hand cases.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <SushiAI/vision/detect/anchor/head_assignment.hpp>

namespace
{
    using SushiAI::Result;
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;
    using SushiAI::Vision::Detect::Anchor::AnchorSize;
    using SushiAI::Vision::Detect::Anchor::assign_anchors_to_heads;
    using SushiAI::Vision::Detect::Anchor::validate_anchor_layout;

    /** @brief Names the indices of each head, in the order of the strides. */
    using Masks = std::vector<std::vector<std::int64_t>>;

    /** @brief Holds a pixel anchor as a cfg writes it: width, then height. */
    struct PixelAnchor
    {
        float width = 0.0f;  /**< Width in pixels of the network input. */
        float height = 0.0f; /**< Height in pixels of the network input. */
    };

    /** @brief Holds the anchors, input side, head strides and masks one cfg states. */
    struct CfgCase
    {
        const char* name = "";             /**< The cfg, for the failure message. */
        std::vector<PixelAnchor> anchors;  /**< The `anchors` key, in file order. */
        float input_side = 0.0f;           /**< `width` and `height` of `[net]`, equal. */
        std::vector<std::int64_t> strides; /**< The stride of each `[yolo]`, in file order. */
        Masks masks;                       /**< The `mask` of each `[yolo]`, in file order. */
    };

    /** @brief Returns the six anchors `yolov3-tiny` and `yolov4-tiny` share. */
    std::vector<PixelAnchor> make_tiny_anchors()
    {
        return {{10, 14}, {23, 27}, {37, 58}, {81, 82}, {135, 169}, {344, 319}};
    }

    /** @brief Returns the nine anchors `yolov3` and `yolov7-tiny` share. */
    std::vector<PixelAnchor> make_v3_anchors()
    {
        return {{10, 13}, {16, 30},  {33, 23},  {30, 61},  {62, 45},
                {59, 119}, {116, 90}, {156, 198}, {373, 326}};
    }

    /** @brief Returns the nine anchors `yolov4` and `yolov7` share. */
    std::vector<PixelAnchor> make_v4_anchors()
    {
        return {{12, 16}, {19, 36},   {40, 28},   {36, 75},  {76, 55},
                {72, 146}, {142, 110}, {192, 243}, {459, 401}};
    }

    /** @brief Returns @p anchors as fractions of a square input of @p input_side pixels. */
    std::vector<AnchorSize> to_fractions(const std::vector<PixelAnchor>& anchors, float input_side)
    {
        std::vector<AnchorSize> fractions;
        fractions.reserve(anchors.size());
        for (const PixelAnchor& anchor : anchors)
            fractions.push_back({anchor.width / input_side, anchor.height / input_side});
        return fractions;
    }

    /** @brief Returns the assignment of @p anchors over @p strides with the given counts. */
    Result<AnchorLayout> assign(const std::vector<AnchorSize>& anchors,
                                const std::vector<std::int64_t>& strides,
                                const std::vector<std::int64_t>& per_head = {})
    {
        return assign_anchors_to_heads(
            SushiRuntime::span<const AnchorSize>(anchors.data(), anchors.size()),
            SushiRuntime::span<const std::int64_t>(strides.data(), strides.size()),
            SushiRuntime::span<const std::int64_t>(per_head.data(), per_head.size()));
    }

    /** @brief Returns the status code of an assignment that is expected to fail. */
    StatusCode code_of(const Result<AnchorLayout>& result)
    {
        return result.status().code();
    }

    /** @brief Returns six anchors whose areas rise with the index, as fractions. */
    std::vector<AnchorSize> make_rising_anchors()
    {
        return to_fractions(make_tiny_anchors(), 416.0f);
    }
} // namespace

/** @brief Tests that nine anchors over strides 32, 16, 8 give 6,7,8 then 3,4,5 then 0,1,2. */
TEST(Unit_DetectAnchorHeadAssignment, NineAnchorsOverThreeHeadsGiveThreeRunsOfThree)
{
    const Result<AnchorLayout> result = assign(to_fractions(make_v3_anchors(), 416.0f),
                                               {32, 16, 8});

    ASSERT_TRUE(result.has_value()) << result.status().message();
    EXPECT_EQ(result.value().head_anchors, (Masks{{6, 7, 8}, {3, 4, 5}, {0, 1, 2}}));
}

/** @brief Tests that six anchors over strides 32, 16 give 3,4,5 then 0,1,2. */
TEST(Unit_DetectAnchorHeadAssignment, SixAnchorsOverTwoHeadsGiveTwoRunsOfThree)
{
    const Result<AnchorLayout> result = assign(make_rising_anchors(), {32, 16});

    ASSERT_TRUE(result.has_value()) << result.status().message();
    EXPECT_EQ(result.value().head_anchors, (Masks{{3, 4, 5}, {0, 1, 2}}));
}

/** @brief Tests that the masks of the five cfgs follow from their anchors and strides. */
TEST(Unit_DetectAnchorHeadAssignment, ReproducesTheMasksOfFiveCfgs)
{
    const std::vector<CfgCase> cases = {
        {"yolov3-tiny", make_tiny_anchors(), 416.0f, {32, 16}, {{3, 4, 5}, {0, 1, 2}}},
        {"yolov3", make_v3_anchors(), 416.0f, {32, 16, 8}, {{6, 7, 8}, {3, 4, 5}, {0, 1, 2}}},
        {"yolov4", make_v4_anchors(), 608.0f, {8, 16, 32}, {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}},
        {"yolov7-tiny", make_v3_anchors(), 416.0f, {8, 16, 32}, {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}},
        {"yolov7", make_v4_anchors(), 640.0f, {8, 16, 32}, {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}},
    };

    for (const CfgCase& cfg : cases)
    {
        const Result<AnchorLayout> result = assign(to_fractions(cfg.anchors, cfg.input_side),
                                                   cfg.strides);

        ASSERT_TRUE(result.has_value()) << cfg.name << ": " << result.status().message();
        EXPECT_EQ(result.value().head_anchors, cfg.masks) << cfg.name;
    }
}

/** @brief Tests that `yolov4-tiny` gets 0,1,2 where its cfg has 1,2,3, one anchor apart. */
TEST(Unit_DetectAnchorHeadAssignment, DiffersFromTheSecondMaskOfYolov4TinyByOneAnchor)
{
    const Masks cfg_masks = {{3, 4, 5}, {1, 2, 3}};

    const Result<AnchorLayout> result = assign(make_rising_anchors(), {32, 16});

    ASSERT_TRUE(result.has_value()) << result.status().message();
    const Masks& computed = result.value().head_anchors;
    EXPECT_EQ(computed[0], cfg_masks[0]);
    // The cfg's second mask is the computed one with anchor 0 dropped and anchor 3 added.
    EXPECT_EQ(computed[1], (std::vector<std::int64_t>{0, 1, 2}));
    EXPECT_NE(computed[1], cfg_masks[1]);
}

/** @brief Tests that of seven anchors over two heads the smaller stride takes four. */
TEST(Unit_DetectAnchorHeadAssignment, TheRemainderGoesToTheHeadOfSmallestStride)
{
    std::vector<AnchorSize> anchors = make_rising_anchors();
    anchors.push_back({0.9f, 0.9f});

    const Result<AnchorLayout> large_first = assign(anchors, {32, 16});
    const Result<AnchorLayout> small_first = assign(anchors, {16, 32});

    ASSERT_TRUE(large_first.has_value()) << large_first.status().message();
    ASSERT_TRUE(small_first.has_value()) << small_first.status().message();
    EXPECT_EQ(large_first.value().head_anchors, (Masks{{4, 5, 6}, {0, 1, 2, 3}}));
    EXPECT_EQ(small_first.value().head_anchors, (Masks{{0, 1, 2, 3}, {4, 5, 6}}));
}

/** @brief Tests that eleven anchors over three heads split 4, 4, 3 from the smallest stride. */
TEST(Unit_DetectAnchorHeadAssignment, ARemainderOfTwoGoesToTheTwoSmallestStrides)
{
    std::vector<AnchorSize> anchors;
    for (int index = 0; index < 11; ++index)
        anchors.push_back({0.05f * static_cast<float>(index + 1), 0.05f});

    const Result<AnchorLayout> result = assign(anchors, {16, 32, 8});

    ASSERT_TRUE(result.has_value()) << result.status().message();
    EXPECT_EQ(result.value().head_anchors, (Masks{{4, 5, 6, 7}, {8, 9, 10}, {0, 1, 2, 3}}));
}

/** @brief Tests that unsorted anchors keep their order and are named in ascending area. */
TEST(Unit_DetectAnchorHeadAssignment, UnsortedAnchorsKeepTheirOrderAndAreNamedByArea)
{
    // The areas are 0.20, 0.01, 0.06 and 0.0004, so ascending area is 3, 1, 2, 0.
    const std::vector<AnchorSize> anchors = {{0.5f, 0.4f}, {0.1f, 0.1f}, {0.2f, 0.3f},
                                             {0.02f, 0.02f}};

    const Result<AnchorLayout> result = assign(anchors, {32, 16});

    ASSERT_TRUE(result.has_value()) << result.status().message();
    const AnchorLayout& layout = result.value();
    EXPECT_EQ(layout.head_anchors, (Masks{{2, 0}, {3, 1}}));
    ASSERT_EQ(layout.anchors.size(), anchors.size());
    for (std::size_t index = 0; index < anchors.size(); ++index)
    {
        // The layout holds copies of the inputs, so the comparison is exact.
        EXPECT_EQ(layout.anchors[index].width, anchors[index].width) << index;
        EXPECT_EQ(layout.anchors[index].height, anchors[index].height) << index;
    }
}

/** @brief Tests that two anchors of one area go in the order of their indices. */
TEST(Unit_DetectAnchorHeadAssignment, AnchorsOfEqualAreaGoInIndexOrder)
{
    const std::vector<AnchorSize> anchors = {{0.25f, 0.5f}, {0.5f, 0.25f}};

    const Result<AnchorLayout> result = assign(anchors, {8, 16});

    ASSERT_TRUE(result.has_value()) << result.status().message();
    EXPECT_EQ(result.value().head_anchors, (Masks{{0}, {1}}));
}

/** @brief Tests that an assignment is a layout the layout check accepts. */
TEST(Unit_DetectAnchorHeadAssignment, TheResultIsAValidLayout)
{
    const Result<AnchorLayout> result = assign(to_fractions(make_v4_anchors(), 608.0f),
                                               {8, 16, 32});

    ASSERT_TRUE(result.has_value()) << result.status().message();
    const SushiAI::Status status = validate_anchor_layout(result.value());
    EXPECT_TRUE(status.is_ok()) << status.message();
}

/** @brief Tests that per_head gives each head its count, in the order of the strides. */
TEST(Unit_DetectAnchorHeadAssignment, PerHeadSetsTheCountOfEachHead)
{
    const Result<AnchorLayout> result = assign(make_rising_anchors(), {32, 16}, {2, 4});

    ASSERT_TRUE(result.has_value()) << result.status().message();
    EXPECT_EQ(result.value().head_anchors, (Masks{{4, 5}, {0, 1, 2, 3}}));
}

/** @brief Tests that a per_head whose sum is below or above the anchor count is refused. */
TEST(Unit_DetectAnchorHeadAssignment, RefusesAPerHeadThatDoesNotSumToTheCount)
{
    EXPECT_EQ(code_of(assign(make_rising_anchors(), {32, 16}, {2, 3})), StatusCode::Incompatible);
    EXPECT_EQ(code_of(assign(make_rising_anchors(), {32, 16}, {3, 4})), StatusCode::Incompatible);
    const std::int64_t huge = std::numeric_limits<std::int64_t>::max();
    EXPECT_EQ(code_of(assign(make_rising_anchors(), {32, 16}, {huge, huge})),
              StatusCode::Incompatible);
}

/** @brief Tests that a per_head with another length than the strides is refused. */
TEST(Unit_DetectAnchorHeadAssignment, RefusesAPerHeadOfAnotherLengthThanTheStrides)
{
    EXPECT_EQ(code_of(assign(make_rising_anchors(), {32, 16}, {6})), StatusCode::Incompatible);
    EXPECT_EQ(code_of(assign(make_rising_anchors(), {32, 16}, {2, 2, 2})),
              StatusCode::Incompatible);
}

/** @brief Tests that a per_head that leaves a head without an anchor is refused. */
TEST(Unit_DetectAnchorHeadAssignment, RefusesAPerHeadWithAnEmptyHead)
{
    EXPECT_EQ(code_of(assign(make_rising_anchors(), {32, 16}, {6, 0})),
              StatusCode::InvalidDescription);
    EXPECT_EQ(code_of(assign(make_rising_anchors(), {32, 16}, {7, -1})),
              StatusCode::InvalidDescription);
}

/** @brief Tests that fewer anchors than heads are refused. */
TEST(Unit_DetectAnchorHeadAssignment, RefusesFewerAnchorsThanHeads)
{
    const std::vector<AnchorSize> anchors = {{0.1f, 0.1f}, {0.2f, 0.2f}};

    EXPECT_EQ(code_of(assign(anchors, {8, 16, 32})), StatusCode::Incompatible);
}

/** @brief Tests that no anchor, no stride and a stride at or below zero are refused. */
TEST(Unit_DetectAnchorHeadAssignment, RefusesMissingAnchorsAndBadStrides)
{
    EXPECT_EQ(code_of(assign({}, {32, 16})), StatusCode::InvalidDescription);
    EXPECT_EQ(code_of(assign(make_rising_anchors(), {})), StatusCode::InvalidDescription);
    EXPECT_EQ(code_of(assign(make_rising_anchors(), {32, 0})), StatusCode::InvalidDescription);
    EXPECT_EQ(code_of(assign(make_rising_anchors(), {-8, 16})), StatusCode::InvalidDescription);
}

/** @brief Tests that an anchor with a zero or a NaN side is refused. */
TEST(Unit_DetectAnchorHeadAssignment, RefusesAnAnchorWithoutAFinitePositiveSide)
{
    std::vector<AnchorSize> zero_side = make_rising_anchors();
    zero_side[1].height = 0.0f;
    std::vector<AnchorSize> nan_side = make_rising_anchors();
    nan_side[4].width = std::numeric_limits<float>::quiet_NaN();

    EXPECT_EQ(code_of(assign(zero_side, {32, 16})), StatusCode::InvalidDescription);
    EXPECT_EQ(code_of(assign(nan_side, {32, 16})), StatusCode::InvalidDescription);
}
