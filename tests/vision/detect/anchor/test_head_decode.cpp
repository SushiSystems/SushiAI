/**************************************************************************/
/* test_head_decode.cpp                                                   */
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
 * @file test_head_decode.cpp
 * @brief Tests the decode of an anchor head on inputs at zero and saturation, and its row order.
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
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;

    using SushiAI::Vision::Detect::Anchor::decode_head;

    using SushiAI::Vision::Detect::Anchor::HeadCoding;

    using SushiAI::Vision::Detect::Anchor::HeadOptions;

    using SushiAI::Vision::Detect::Anchor::HeadShape;

    /** @brief Holds the absolute tolerance of every decoded value, from the unit's acceptance. */
    constexpr float K_TOLERANCE = 1e-6f;

    /** @brief Holds a logit whose float32 logistic is exactly 1, and exactly 0 when negated. */
    constexpr float K_SATURATED = 100.0f;

    /** @brief Holds the entry of the centre offset along the columns within an anchor slot. */
    constexpr std::size_t K_X = 0;

    /** @brief Holds the entry of the centre offset along the rows within an anchor slot. */
    constexpr std::size_t K_Y = 1;

    /** @brief Holds the entry of the box width within an anchor slot. */
    constexpr std::size_t K_WIDTH = 2;

    /** @brief Holds the entry of the box height within an anchor slot. */
    constexpr std::size_t K_HEIGHT = 3;

    /** @brief Holds the entry of the objectness within an anchor slot. */
    constexpr std::size_t K_OBJECTNESS = 4;

    /** @brief Holds the entry of class 0; class c follows at K_FIRST_CLASS + c. */
    constexpr std::size_t K_FIRST_CLASS = 5;

    /** @brief Holds a head output and what decode_head writes from it. */
    struct HeadCase
    {
        HeadShape shape;           /**< Extents of the head. */
        HeadOptions options;       /**< Classes, coding and scale. */
        AnchorLayout layout;       /**< Anchors; the case decodes head 0. */
        std::vector<float> head;   /**< `[B, H, W, A * (5 + C)]`, zero until set. */
        std::vector<float> boxes;  /**< `[B, H * W * A, 4]`. */
        std::vector<float> scores; /**< `[B, H * W * A, C]`. */

        /** @brief Returns the anchor slots of head 0. */
        std::size_t slots() const
        {
            return layout.head_anchors[0].size();
        }

        /** @brief Returns the class entries of a slot. */
        std::size_t classes() const
        {
            return static_cast<std::size_t>(options.num_classes);
        }

        /** @brief Returns the entry @p entry of slot @p slot in cell (@p row, @p col). */
        float& at(std::size_t image, std::size_t row, std::size_t col, std::size_t slot,
                  std::size_t entry)
        {
            const std::size_t width = static_cast<std::size_t>(shape.width);
            const std::size_t height = static_cast<std::size_t>(shape.height);
            const std::size_t per_slot = K_FIRST_CLASS + classes();
            const std::size_t cell = (image * height + row) * width + col;
            return head[(cell * slots() + slot) * per_slot + entry];
        }

        /** @brief Returns the four box values of decoded row @p row of image @p image. */
        const float* box(std::size_t image, std::size_t row) const
        {
            return boxes.data() + (image * rows_per_image() + row) * 4;
        }

        /** @brief Returns the class scores of decoded row @p row of image @p image. */
        const float* score(std::size_t image, std::size_t row) const
        {
            return scores.data() + (image * rows_per_image() + row) * classes();
        }

        /** @brief Returns the decoded rows of one image, `H * W * A`. */
        std::size_t rows_per_image() const
        {
            return static_cast<std::size_t>(shape.height * shape.width) * slots();
        }

        /** @brief Runs decode_head on head 0. */
        void decode()
        {
            decode_head(head, shape, options, layout, 0, boxes, scores);
        }
    };

    /** @brief Returns a case with a zero head and outputs sized for it. */
    HeadCase make_case(const HeadShape& shape, const HeadOptions& options,
                       const AnchorLayout& layout)
    {
        HeadCase made;
        made.shape = shape;
        made.options = options;
        made.layout = layout;

        const std::size_t rows =
            static_cast<std::size_t>(shape.batch) * made.rows_per_image();
        made.head.assign(rows * (K_FIRST_CLASS + made.classes()), 0.0f);
        made.boxes.assign(rows * 4, -1.0f);
        made.scores.assign(rows * made.classes(), -1.0f);
        return made;
    }

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

/** @brief Tests that zero logits decode to the cell centre, the anchor size and scores of 0.25. */
TEST(Unit_DetectAnchorDecode, ZeroLogitsGiveTheCellCentreTheAnchorAndQuarterScores)
{
    HeadCase head = make_case({1, 1, 1}, make_options(2, HeadCoding::Logit, 1.0f),
                              make_single_anchor_layout(0.25f, 0.5f));

    head.decode();

    // logistic(0) is 0.5 and exp(0) is 1, so the box is the cell centre and the anchor.
    EXPECT_NEAR(head.box(0, 0)[0], 0.5f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[1], 0.5f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[2], 0.25f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[3], 0.5f, K_TOLERANCE);
    EXPECT_NEAR(head.score(0, 0)[0], 0.25f, K_TOLERANCE);
    EXPECT_NEAR(head.score(0, 0)[1], 0.25f, K_TOLERANCE);
}

/** @brief Tests that saturated logits reach the cell edges and the score bounds 0 and 1. */
TEST(Unit_DetectAnchorDecode, SaturatedLogitsReachTheCellEdgesAndTheScoreBounds)
{
    HeadCase head = make_case({1, 1, 1}, make_options(2, HeadCoding::Logit, 1.0f),
                              make_single_anchor_layout(0.25f, 0.5f));
    head.at(0, 0, 0, 0, K_X) = K_SATURATED;
    head.at(0, 0, 0, 0, K_Y) = -K_SATURATED;
    head.at(0, 0, 0, 0, K_WIDTH) = -K_SATURATED;
    head.at(0, 0, 0, 0, K_OBJECTNESS) = K_SATURATED;
    head.at(0, 0, 0, 0, K_FIRST_CLASS) = K_SATURATED;
    head.at(0, 0, 0, 0, K_FIRST_CLASS + 1) = -K_SATURATED;

    head.decode();

    // exp(-100) is below 1e-43, so the width vanishes; the height keeps exp(0).
    EXPECT_NEAR(head.box(0, 0)[0], 1.0f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[1], 0.0f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[2], 0.0f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[3], 0.5f, K_TOLERANCE);
    EXPECT_NEAR(head.score(0, 0)[0], 1.0f, K_TOLERANCE);
    EXPECT_NEAR(head.score(0, 0)[1], 0.0f, K_TOLERANCE);
}

/** @brief Tests that a score is objectness times class: 0.5 times 1. */
TEST(Unit_DetectAnchorDecode, ZeroObjectnessLogitHalvesASaturatedClass)
{
    HeadCase head = make_case({1, 1, 1}, make_options(1, HeadCoding::Logit, 1.0f),
                              make_single_anchor_layout(0.25f, 0.5f));
    head.at(0, 0, 0, 0, K_FIRST_CLASS) = K_SATURATED;

    head.decode();

    EXPECT_NEAR(head.score(0, 0)[0], 0.5f, K_TOLERANCE);
}

/** @brief Tests that scale_xy 2.0 moves a saturated offset half a cell past each edge. */
TEST(Unit_DetectAnchorDecode, ScaleTwoMovesASaturatedOffsetHalfACellPastTheEdge)
{
    HeadCase head = make_case({1, 1, 1}, make_options(1, HeadCoding::Logit, 2.0f),
                              make_single_anchor_layout(0.25f, 0.5f));
    head.at(0, 0, 0, 0, K_X) = K_SATURATED;
    head.at(0, 0, 0, 0, K_Y) = -K_SATURATED;

    head.decode();

    // 1 * 2 - 0.5 is 1.5 and 0 * 2 - 0.5 is -0.5, on a grid of one cell.
    EXPECT_NEAR(head.box(0, 0)[0], 1.5f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[1], -0.5f, K_TOLERANCE);
}

/** @brief Tests that scale_xy stretches about 0.5, so a zero logit stays at the cell centre. */
TEST(Unit_DetectAnchorDecode, ScaleTwoKeepsAZeroLogitAtTheCellCentre)
{
    HeadCase head = make_case({1, 1, 1}, make_options(1, HeadCoding::Logit, 2.0f),
                              make_single_anchor_layout(0.25f, 0.5f));

    head.decode();

    EXPECT_NEAR(head.box(0, 0)[0], 0.5f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[1], 0.5f, K_TOLERANCE);
}

/** @brief Tests that the Activated size at 0, 0.5 and 1 is 0, the anchor and four anchors. */
TEST(Unit_DetectAnchorDecode, ActivatedSizeAtZeroHalfAndOneIsZeroTheAnchorAndFourAnchors)
{
    HeadCase head = make_case({1, 1, 3}, make_options(1, HeadCoding::Activated, 1.0f),
                              make_single_anchor_layout(0.125f, 0.25f));
    const float inputs[3] = {0.0f, 0.5f, 1.0f};
    const float widths[3] = {0.0f, 0.125f, 0.5f};
    const float heights[3] = {0.0f, 0.25f, 1.0f};
    for (std::size_t col = 0; col < 3; ++col)
    {
        head.at(0, 0, col, 0, K_WIDTH) = inputs[col];
        head.at(0, 0, col, 0, K_HEIGHT) = inputs[col];
    }

    head.decode();

    for (std::size_t col = 0; col < 3; ++col)
    {
        EXPECT_NEAR(head.box(0, col)[2], widths[col], K_TOLERANCE) << "column " << col;
        EXPECT_NEAR(head.box(0, col)[3], heights[col], K_TOLERANCE) << "column " << col;
    }
}

/** @brief Tests that the Activated coding reads offsets, objectness and classes as they arrive. */
TEST(Unit_DetectAnchorDecode, ActivatedOffsetsAndScoresTakeNoLogistic)
{
    HeadCase head = make_case({1, 1, 1}, make_options(2, HeadCoding::Activated, 1.0f),
                              make_single_anchor_layout(0.125f, 0.25f));
    head.at(0, 0, 0, 0, K_X) = 0.25f;
    head.at(0, 0, 0, 0, K_Y) = 1.0f;
    head.at(0, 0, 0, 0, K_OBJECTNESS) = 0.5f;
    head.at(0, 0, 0, 0, K_FIRST_CLASS) = 0.5f;
    head.at(0, 0, 0, 0, K_FIRST_CLASS + 1) = 1.0f;

    head.decode();

    EXPECT_NEAR(head.box(0, 0)[0], 0.25f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[1], 1.0f, K_TOLERANCE);
    EXPECT_NEAR(head.score(0, 0)[0], 0.25f, K_TOLERANCE);
    EXPECT_NEAR(head.score(0, 0)[1], 0.5f, K_TOLERANCE);
}

/** @brief Tests that scale_xy 2.0 maps Activated offsets 0 and 1 to -0.5 and 1.5. */
TEST(Unit_DetectAnchorDecode, ActivatedScaleTwoMapsZeroAndOnePastTheCellEdges)
{
    HeadCase head = make_case({1, 1, 1}, make_options(1, HeadCoding::Activated, 2.0f),
                              make_single_anchor_layout(0.125f, 0.25f));
    head.at(0, 0, 0, 0, K_X) = 0.0f;
    head.at(0, 0, 0, 0, K_Y) = 1.0f;

    head.decode();

    // 0 * 2 - 0.5 is -0.5 and 1 * 2 - 0.5 is 1.5.
    EXPECT_NEAR(head.box(0, 0)[0], -0.5f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[1], 1.5f, K_TOLERANCE);
}

/** @brief Tests that decoded row (row * W + col) * A + a holds that cell and slot, per image. */
TEST(Unit_DetectAnchorDecode, RowsRunCellByCellThenByAnchorSlotInEveryImage)
{
    AnchorLayout layout;
    layout.anchors = {{0.1f, 0.2f}, {0.3f, 0.3f}, {0.5f, 0.6f}};
    layout.head_anchors = {{2, 0}};
    HeadCase head = make_case({2, 2, 3}, make_options(1, HeadCoding::Logit, 1.0f), layout);
    head.at(1, 1, 2, 1, K_OBJECTNESS) = K_SATURATED;
    head.at(1, 1, 2, 1, K_FIRST_CLASS) = K_SATURATED;

    head.decode();

    const float slot_widths[2] = {0.5f, 0.1f};
    const float slot_heights[2] = {0.6f, 0.2f};
    for (std::size_t image = 0; image < 2; ++image)
    {
        for (std::size_t row = 0; row < 2; ++row)
        {
            for (std::size_t col = 0; col < 3; ++col)
            {
                for (std::size_t slot = 0; slot < 2; ++slot)
                {
                    const std::size_t decoded = (row * 3 + col) * 2 + slot;
                    const float* box = head.box(image, decoded);
                    const bool marked = image == 1 && decoded == 11;
                    SCOPED_TRACE(testing::Message() << "image " << image << " row " << decoded);

                    EXPECT_NEAR(box[0], (static_cast<float>(col) + 0.5f) / 3.0f, K_TOLERANCE);
                    EXPECT_NEAR(box[1], (static_cast<float>(row) + 0.5f) / 2.0f, K_TOLERANCE);
                    EXPECT_NEAR(box[2], slot_widths[slot], K_TOLERANCE);
                    EXPECT_NEAR(box[3], slot_heights[slot], K_TOLERANCE);
                    EXPECT_NEAR(head.score(image, decoded)[0], marked ? 1.0f : 0.25f,
                                K_TOLERANCE);
                }
            }
        }
    }
}

/** @brief Tests that one layout decodes the same fractional size on a 13 and a 19 grid. */
TEST(Unit_DetectAnchorDecode, OneLayoutGivesTheSameFractionalSizeOnTwoGrids)
{
    const AnchorLayout layout = make_single_anchor_layout(0.3f, 0.45f);
    const HeadOptions options = make_options(1, HeadCoding::Logit, 1.0f);
    HeadCase coarse = make_case({1, 13, 13}, options, layout);
    HeadCase fine = make_case({1, 19, 19}, options, layout);

    coarse.decode();
    fine.decode();

    EXPECT_NEAR(coarse.box(0, 0)[2], 0.3f, K_TOLERANCE);
    EXPECT_NEAR(coarse.box(0, 0)[3], 0.45f, K_TOLERANCE);
    EXPECT_EQ(coarse.box(0, 168)[2], fine.box(0, 360)[2]);
    EXPECT_EQ(coarse.box(0, 168)[3], fine.box(0, 360)[3]);

    // Row 168 is the last cell of 13 by 13 and row 360 the last of 19 by 19.
    EXPECT_NEAR(coarse.box(0, 168)[0], 12.5f / 13.0f, K_TOLERANCE);
    EXPECT_NEAR(fine.box(0, 360)[0], 18.5f / 19.0f, K_TOLERANCE);
}
