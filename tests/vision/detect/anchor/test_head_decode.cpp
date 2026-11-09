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
 * @file test_head_decode.cpp
 * @brief Tests the decode of an anchor head on inputs at zero and saturation, and its row order.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_decode.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <support/tolerance.hpp>

namespace
{
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;

    using SushiAI::Vision::Detect::Anchor::decode_head;

    using SushiAI::Vision::Detect::Anchor::HeadCoding;

    using SushiAI::Vision::Detect::Anchor::HeadOptions;

    using SushiAI::Vision::Detect::Anchor::HeadShape;
    using SushiAI::Vision::Detect::Anchor::SizeForm;

    /**
     * @brief Holds the absolute tolerance of every decoded value, from the unit's acceptance: 32
     *        times the float unit roundoff, the smallest power of two not tighter than the earlier
     *        1e-6f.
     */
    constexpr float K_TOLERANCE = static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());

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

    /** @brief Holds an Activated size input and the width and height it decodes to. */
    struct ActivatedSizeRow
    {
        std::string name;  /**< The row's name. */
        float input;       /**< The width and height entry of the slot. */
        float width;       /**< The decoded width. */
        float height;      /**< The decoded height. */
    };

    /** @brief Runs one Activated size input. */
    class Small_DetectAnchorDecodeActivatedSize : public ::testing::TestWithParam<ActivatedSizeRow>
    {

    };

    /** @brief Names a row by its name field. */
    std::string activated_size_name(const ::testing::TestParamInfo<ActivatedSizeRow>& info)
    {
        return info.param.name;
    }

    /** @brief Holds a decoded row of a two image, 2 by 3 head and the values it must hold. */
    struct DecodedRow
    {
        std::string name;    /**< The row's name. */
        std::size_t image;   /**< The image of the row. */
        std::size_t index;   /**< The decoded row index within the image. */
        float x;             /**< The decoded centre along the columns. */
        float y;             /**< The decoded centre along the rows. */
        float width;         /**< The decoded width. */
        float height;        /**< The decoded height. */
        float score;         /**< The decoded class score. */
    };

    /** @brief Runs one decoded row. */
    class Small_DetectAnchorDecodeRow : public ::testing::TestWithParam<DecodedRow>
    {

    };

    /** @brief Names a row by its name field. */
    std::string decoded_row_name(const ::testing::TestParamInfo<DecodedRow>& info)
    {
        return info.param.name;
    }

} // namespace

/** @brief Tests that zero logits decode to the cell centre and the anchor size. */
TEST(Small_DetectAnchorDecode, ZeroLogitsGiveTheCellCentreAndTheAnchor)
{
    HeadCase head = make_case({1, 1, 1}, make_options(2, HeadCoding::Logit, 1.0f),
                              make_single_anchor_layout(0.25f, 0.5f));

    head.decode();

    // logistic(0) is 0.5 and exp(0) is 1, so the box is the cell centre and the anchor.
    EXPECT_NEAR(head.box(0, 0)[0], 0.5f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[1], 0.5f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[2], 0.25f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[3], 0.5f, K_TOLERANCE);
}

/** @brief Tests that zero logits score every class at 0.25, the product of two halves. */
TEST(Small_DetectAnchorDecode, ZeroLogitsGiveQuarterScores)
{
    HeadCase head = make_case({1, 1, 1}, make_options(2, HeadCoding::Logit, 1.0f),
                              make_single_anchor_layout(0.25f, 0.5f));

    head.decode();

    EXPECT_NEAR(head.score(0, 0)[0], 0.25f, K_TOLERANCE);
    EXPECT_NEAR(head.score(0, 0)[1], 0.25f, K_TOLERANCE);
}

/** @brief Tests the squared-sigmoid size form at a zero, a saturated and a negative logit. */
TEST(Small_DetectAnchorDecode, SquaredSigmoidSidesAreTwiceTheLogisticSquaredTimesTheAnchor)
{
    HeadOptions options = make_options(1, HeadCoding::Logit, 2.0f);
    options.size = SizeForm::SquaredSigmoid;
    HeadCase head = make_case({1, 1, 1}, options, make_single_anchor_layout(0.25f, 0.5f));
    head.at(0, 0, 0, 0, K_WIDTH) = K_SATURATED;
    head.at(0, 0, 0, 0, K_HEIGHT) = -1.0f;

    head.decode();

    // A saturated width is (2 * 1)^2 = 4 anchors; logistic(-1) is 0.2689414.
    const float doubled = 2.0f * 0.26894142f;
    EXPECT_NEAR(head.box(0, 0)[0], 0.5f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[2], 1.0f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[3], doubled * doubled * 0.5f, K_TOLERANCE);
}

/** @brief Tests that saturated logits reach the cell edges and the score bounds 0 and 1. */
TEST(Small_DetectAnchorDecode, SaturatedLogitsReachTheCellEdgesAndTheScoreBounds)
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

/** @brief Tests that a saturated objectness times a zero class logit scores 1 times 0.5. */
TEST(Small_DetectAnchorDecode, SaturatedObjectnessScoresAZeroClassLogitAtHalf)
{
    HeadCase head = make_case({1, 1, 1}, make_options(1, HeadCoding::Logit, 1.0f),
                              make_single_anchor_layout(0.25f, 0.5f));
    head.at(0, 0, 0, 0, K_OBJECTNESS) = K_SATURATED;
    head.at(0, 0, 0, 0, K_FIRST_CLASS) = 0.0f;

    head.decode();

    EXPECT_NEAR(head.score(0, 0)[0], 0.5f, K_TOLERANCE);
}

/** @brief Tests that scale_xy 2.0 moves a saturated offset half a cell past each edge. */
TEST(Small_DetectAnchorDecode, ScaleTwoMovesASaturatedOffsetHalfACellPastTheEdge)
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
TEST(Small_DetectAnchorDecode, ScaleTwoKeepsAZeroLogitAtTheCellCentre)
{
    HeadCase head = make_case({1, 1, 1}, make_options(1, HeadCoding::Logit, 2.0f),
                              make_single_anchor_layout(0.25f, 0.5f));

    head.decode();

    EXPECT_NEAR(head.box(0, 0)[0], 0.5f, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[1], 0.5f, K_TOLERANCE);
}

/** @brief Tests that an Activated size input decodes to its row's width and height. */
TEST_P(Small_DetectAnchorDecodeActivatedSize, ASizeInputDecodesToItsFractionOfTheAnchors)
{
    HeadCase head = make_case({1, 1, 1}, make_options(1, HeadCoding::Activated, 1.0f),
                              make_single_anchor_layout(0.125f, 0.25f));
    head.at(0, 0, 0, 0, K_WIDTH) = GetParam().input;
    head.at(0, 0, 0, 0, K_HEIGHT) = GetParam().input;

    head.decode();

    EXPECT_NEAR(head.box(0, 0)[2], GetParam().width, K_TOLERANCE);
    EXPECT_NEAR(head.box(0, 0)[3], GetParam().height, K_TOLERANCE);
}

// An input of 0 gives no size, 0.5 the anchor itself and 1 four anchors.
INSTANTIATE_TEST_SUITE_P(Rows, Small_DetectAnchorDecodeActivatedSize,
                         ::testing::Values(ActivatedSizeRow{"Zero", 0.0f, 0.0f, 0.0f},
                                           ActivatedSizeRow{"Half", 0.5f, 0.125f, 0.25f},
                                           ActivatedSizeRow{"One", 1.0f, 0.5f, 1.0f}),
                         activated_size_name);

/** @brief Tests that the Activated coding reads offsets, objectness and classes as they arrive. */
TEST(Small_DetectAnchorDecode, ActivatedOffsetsAndScoresTakeNoLogistic)
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
TEST(Small_DetectAnchorDecode, ActivatedScaleTwoMapsZeroAndOnePastTheCellEdges)
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

/** @brief Tests that a decoded row holds the cell and the slot its index names, in its image. */
TEST_P(Small_DetectAnchorDecodeRow, ARowHoldsTheCellAndSlotItsIndexNames)
{
    AnchorLayout layout;
    layout.anchors = {{0.1f, 0.2f}, {0.3f, 0.3f}, {0.5f, 0.6f}};
    layout.head_anchors = {{2, 0}};
    HeadCase head = make_case({2, 2, 3}, make_options(1, HeadCoding::Logit, 1.0f), layout);
    head.at(1, 1, 2, 1, K_OBJECTNESS) = K_SATURATED;
    head.at(1, 1, 2, 1, K_FIRST_CLASS) = K_SATURATED;
    const DecodedRow& row = GetParam();

    head.decode();

    const float* box = head.box(row.image, row.index);
    EXPECT_NEAR(box[0], row.x, K_TOLERANCE);
    EXPECT_NEAR(box[1], row.y, K_TOLERANCE);
    EXPECT_NEAR(box[2], row.width, K_TOLERANCE);
    EXPECT_NEAR(box[3], row.height, K_TOLERANCE);
    EXPECT_NEAR(head.score(row.image, row.index)[0], row.score, K_TOLERANCE);
}

// Index (row * 3 + col) * 2 + slot on a 2 by 3 grid; slot 0 is anchor 2 and slot 1 anchor 0.
INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DetectAnchorDecodeRow,
    ::testing::Values(
        DecodedRow{"FirstCellFirstSlot", 0, 0, 0.5f / 3.0f, 0.5f / 2.0f, 0.5f, 0.6f, 0.25f},
        DecodedRow{"FirstCellSecondSlot", 0, 1, 0.5f / 3.0f, 0.5f / 2.0f, 0.1f, 0.2f, 0.25f},
        DecodedRow{"LastColumnOfTheFirstRow", 0, 4, 2.5f / 3.0f, 0.5f / 2.0f, 0.5f, 0.6f, 0.25f},
        DecodedRow{"SecondRowSecondSlot", 0, 7, 0.5f / 3.0f, 1.5f / 2.0f, 0.1f, 0.2f, 0.25f},
        DecodedRow{"FirstCellOfTheSecondImage", 1, 0, 0.5f / 3.0f, 0.5f / 2.0f, 0.5f, 0.6f,
                   0.25f},
        DecodedRow{"TheMarkedRowOfTheSecondImage", 1, 11, 2.5f / 3.0f, 1.5f / 2.0f, 0.1f, 0.2f,
                   1.0f}),
    decoded_row_name);

/** @brief Tests that one layout decodes the same fractional size on a 13 and a 19 grid. */
TEST(Small_DetectAnchorDecode, OneLayoutGivesTheSameFractionalSizeOnTwoGrids)
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
