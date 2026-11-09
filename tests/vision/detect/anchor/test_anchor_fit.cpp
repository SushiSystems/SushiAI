/**************************************************************************/
/* test_anchor_fit.cpp                                                    */
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
 * @file test_anchor_fit.cpp
 * @brief Tests the fit report of an anchor layout on hand-built boxes.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_fit.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <support/tolerance.hpp>

namespace
{
    using SushiAI::Result;
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Anchor::AnchorFit;
    using SushiAI::Vision::Detect::Anchor::AnchorLayout;
    using SushiAI::Vision::Detect::Anchor::AnchorSize;
    using SushiAI::Vision::Detect::Anchor::FitOptions;
    using SushiAI::Vision::Detect::Anchor::measure_anchor_fit;

    /**
     * @brief Holds the absolute tolerance of a share or a mean summed in double: 16777216 times the
     *        double unit roundoff, the smallest power of two not tighter than the earlier 1e-9.
     */
    constexpr double K_FIT_TOLERANCE = 16777216.0 * TestSupport::unit_roundoff<double>();

    /** @brief Holds a recall_iou the fit refuses and the name of its row. */
    struct RefusedRecall
    {
        std::string name; /**< The row's name. */
        float recall_iou; /**< The recall_iou outside zero to one. */
    };

    /** @brief Runs one recall_iou the fit refuses. */
    class Small_DetectAnchorFitRecall : public ::testing::TestWithParam<RefusedRecall>
    {

    };

    /** @brief Names a row by its name field. */
    std::string recall_name(const ::testing::TestParamInfo<RefusedRecall>& info)
    {
        return info.param.name;
    }

    /** @brief Returns three anchors whose half-size boxes fit no other anchor above 0.2. */
    AnchorLayout make_separated_layout()
    {
        AnchorLayout layout;
        layout.anchors = {{0.0625f, 0.5f}, {0.5f, 0.0625f}, {0.5f, 0.5f}};
        layout.head_anchors = {{0, 1, 2}};
        return layout;
    }

    /** @brief Returns one box per anchor of @p layout with each side times @p factor. */
    std::vector<AnchorSize> make_scaled_boxes(const AnchorLayout& layout, float factor)
    {
        std::vector<AnchorSize> boxes;
        for (const AnchorSize& anchor : layout.anchors)
            boxes.push_back(AnchorSize{anchor.width * factor, anchor.height * factor});
        return boxes;
    }

    /** @brief Returns the six anchors of the yolov4-tiny cfg over its 416 input, two masks. */
    AnchorLayout make_v4_tiny_layout()
    {
        const float pixels[6][2] = {{10.0f, 14.0f}, {23.0f, 27.0f},   {37.0f, 58.0f},
                                    {81.0f, 82.0f}, {135.0f, 169.0f}, {344.0f, 319.0f}};
        AnchorLayout layout;
        for (const auto& anchor : pixels)
            layout.anchors.push_back(AnchorSize{anchor[0] / 416.0f, anchor[1] / 416.0f});
        layout.head_anchors = {{3, 4, 5}, {1, 2, 3}};
        return layout;
    }

    /** @brief Returns @p copies boxes at each anchor size of @p layout, anchor by anchor. */
    std::vector<AnchorSize> repeat_anchor_sizes(const AnchorLayout& layout, std::size_t copies)
    {
        std::vector<AnchorSize> boxes;
        for (const AnchorSize& anchor : layout.anchors)
            boxes.insert(boxes.end(), copies, anchor);
        return boxes;
    }
} // namespace

/** @brief Tests that one box at each anchor's size measures 1 in every number. */
TEST(Small_DetectAnchorFit, OneBoxPerAnchorGivesOneEverywhere)
{
    const AnchorLayout layout = make_separated_layout();
    const std::vector<AnchorSize> boxes = make_scaled_boxes(layout, 1.0f);

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorFit& fit = outcome.value();
    EXPECT_EQ(fit.boxes, 3);
    EXPECT_NEAR(fit.mean_best_iou, 1.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.best_possible_recall, 1.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.best_iou_tenth_percentile, 1.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.unowned_share, 0.0, K_FIT_TOLERANCE);
    ASSERT_EQ(fit.anchor_share.size(), 3u);
    EXPECT_THAT(fit.anchor_share,
                ::testing::Each(::testing::DoubleNear(1.0 / 3.0, K_FIT_TOLERANCE)));
}

/** @brief Tests that boxes at half an anchor's sides measure 0.25 in mean and tenth percentile. */
TEST(Small_DetectAnchorFit, HalfSizedBoxesGiveAQuarter)
{
    const AnchorLayout layout = make_separated_layout();
    const std::vector<AnchorSize> boxes = make_scaled_boxes(layout, 0.5f);

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    // Every side is a power of two, so each area ratio is 0.25 without rounding.
    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_NEAR(outcome.value().mean_best_iou, 0.25, K_FIT_TOLERANCE);
    EXPECT_NEAR(outcome.value().best_iou_tenth_percentile, 0.25, K_FIT_TOLERANCE);
}

/** @brief Tests that a recall_iou of 0.25 recalls every half-sized box. */
TEST(Small_DetectAnchorFit, HalfSizedBoxesAreRecalledAtAQuarter)
{
    const AnchorLayout layout = make_separated_layout();
    const std::vector<AnchorSize> boxes = make_scaled_boxes(layout, 0.5f);
    FitOptions options;
    options.recall_iou = 0.25f;

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_NEAR(outcome.value().best_possible_recall, 1.0, K_FIT_TOLERANCE);
}

/** @brief Tests that a recall_iou of 0.3 recalls none of the half-sized boxes. */
TEST(Small_DetectAnchorFit, HalfSizedBoxesAreNotRecalledAboveAQuarter)
{
    const AnchorLayout layout = make_separated_layout();
    const std::vector<AnchorSize> boxes = make_scaled_boxes(layout, 0.5f);
    FitOptions options;
    options.recall_iou = 0.3f;

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_NEAR(outcome.value().best_possible_recall, 0.0, K_FIT_TOLERANCE);
}

/** @brief Tests that the yolov4-tiny masks leave the boxes of anchor 0 without a head. */
TEST(Small_DetectAnchorFit, V4TinyMasksLeaveASixthOfTheBoxesUnowned)
{
    const AnchorLayout layout = make_v4_tiny_layout();
    const std::vector<AnchorSize> boxes = repeat_anchor_sizes(layout, 100);

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorFit& fit = outcome.value();
    EXPECT_EQ(fit.boxes, 600);
    EXPECT_NEAR(fit.unowned_share, 1.0 / 6.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.best_possible_recall, 5.0 / 6.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.mean_best_iou, 1.0, K_FIT_TOLERANCE);
    ASSERT_EQ(fit.anchor_share.size(), 6u);
    EXPECT_THAT(fit.anchor_share,
                ::testing::Each(::testing::DoubleNear(1.0 / 6.0, K_FIT_TOLERANCE)));
}

/** @brief Tests that masks covering every anchor bring the unowned share of that case to 0. */
TEST(Small_DetectAnchorFit, MasksOverEveryAnchorLeaveNoBoxUnowned)
{
    AnchorLayout layout = make_v4_tiny_layout();
    const std::vector<AnchorSize> boxes = repeat_anchor_sizes(layout, 100);
    layout.head_anchors = {{3, 4, 5}, {0, 1, 2}};

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_NEAR(outcome.value().unowned_share, 0.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(outcome.value().best_possible_recall, 1.0, K_FIT_TOLERANCE);
}

/**
 * @brief Tests the four numbers and the shares against values worked out by hand. Every side is
 *        a power of two, so each IoU is exact. Anchors 0.25 x 0.125 (listed by no head),
 *        0.5 x 0.5, 0.5 x 0.125 and 0.125 x 0.5; the first four boxes equal an anchor each and
 *        have IoU 1, the box 0.0625 x 0.0625 has IoU 1/8 with anchor 0, and the box 0.5 x 0.0625
 *        has IoU 1/2 with anchor 2. Best IoUs are 1, 1, 1, 1, 1/8 and 1/2; boxes 0 and 4 belong
 *        to anchor 0 and so are unowned; boxes 1, 2, 3 and 5 are owned and recalled at 0.5.
 */
TEST(Small_DetectAnchorFit, FourNumbersAndTheSharesMatchHandValues)
{
    AnchorLayout layout;
    layout.anchors = {{0.25f, 0.125f}, {0.5f, 0.5f}, {0.5f, 0.125f}, {0.125f, 0.5f}};
    layout.head_anchors = {{2, 3}, {1}};
    const std::vector<AnchorSize> boxes = {{0.25f, 0.125f}, {0.5f, 0.5f},       {0.5f, 0.125f},
                                           {0.125f, 0.5f},  {0.0625f, 0.0625f}, {0.5f, 0.0625f}};
    FitOptions options;
    options.recall_iou = 0.5f;

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, options);

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    const AnchorFit& fit = outcome.value();
    EXPECT_EQ(fit.boxes, 6);
    EXPECT_NEAR(fit.mean_best_iou, 4.625 / 6.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.best_possible_recall, 4.0 / 6.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.unowned_share, 2.0 / 6.0, K_FIT_TOLERANCE);
    // Six boxes put the tenth percentile at rank ceil(0.6) = 1, the smallest best IoU.
    EXPECT_NEAR(fit.best_iou_tenth_percentile, 0.125, K_FIT_TOLERANCE);
    ASSERT_EQ(fit.anchor_share.size(), 4u);
    EXPECT_NEAR(fit.anchor_share[0], 2.0 / 6.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.anchor_share[1], 1.0 / 6.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.anchor_share[2], 2.0 / 6.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(fit.anchor_share[3], 1.0 / 6.0, K_FIT_TOLERANCE);
}

/** @brief Tests that the tenth percentile of ten boxes is the smallest best IoU. */
TEST(Small_DetectAnchorFit, TenthPercentileOfTenBoxesIsTheSmallest)
{
    AnchorLayout layout;
    layout.anchors = {{0.5f, 0.5f}};
    layout.head_anchors = {{0}};
    std::vector<AnchorSize> boxes(9, AnchorSize{0.5f, 0.5f});
    boxes.push_back(AnchorSize{0.25f, 0.25f});

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    EXPECT_NEAR(outcome.value().best_iou_tenth_percentile, 0.25, K_FIT_TOLERANCE);
    EXPECT_NEAR(outcome.value().mean_best_iou, 0.925, K_FIT_TOLERANCE);
}

/** @brief Tests that of two equal anchors the lower index takes the box. */
TEST(Small_DetectAnchorFit, EqualAnchorsGiveTheBoxToTheLowerIndex)
{
    AnchorLayout layout;
    layout.anchors = {{0.25f, 0.5f}, {0.25f, 0.5f}};
    layout.head_anchors = {{1}};
    const std::vector<AnchorSize> boxes = {{0.25f, 0.5f}};

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    ASSERT_TRUE(outcome.has_value()) << outcome.status().message();
    ASSERT_EQ(outcome.value().anchor_share.size(), 2u);
    EXPECT_NEAR(outcome.value().anchor_share[0], 1.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(outcome.value().anchor_share[1], 0.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(outcome.value().unowned_share, 1.0, K_FIT_TOLERANCE);
    EXPECT_NEAR(outcome.value().best_possible_recall, 0.0, K_FIT_TOLERANCE);
}

/** @brief Tests that an empty box list is Incompatible. */
TEST(Small_DetectAnchorFit, RefusesAnEmptyBoxList)
{
    const std::vector<AnchorSize> boxes;

    const Result<AnchorFit> outcome =
        measure_anchor_fit(boxes, make_separated_layout(), FitOptions{});

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::Incompatible);
}

/** @brief Tests that a layout whose head names a missing anchor is InvalidDescription. */
TEST(Small_DetectAnchorFit, RefusesALayoutThatDoesNotValidate)
{
    AnchorLayout layout = make_separated_layout();
    layout.head_anchors[0][2] = 3;
    const std::vector<AnchorSize> boxes = {{0.5f, 0.5f}};

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, layout, FitOptions{});

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
}

/** @brief Tests that a recall_iou below 0, above 1 or not a number is InvalidDescription. */
TEST_P(Small_DetectAnchorFitRecall, ARecallIoUOutsideZeroToOneIsRefused)
{
    const std::vector<AnchorSize> boxes = {{0.5f, 0.5f}};
    FitOptions options;
    options.recall_iou = GetParam().recall_iou;

    const Result<AnchorFit> outcome = measure_anchor_fit(boxes, make_separated_layout(), options);

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.status().code(), StatusCode::InvalidDescription);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DetectAnchorFitRecall,
                         ::testing::Values(RefusedRecall{"BelowZero", -0.1f},
                                           RefusedRecall{"AboveOne", 1.5f},
                                           RefusedRecall{"NotANumber",
                                                         std::numeric_limits<float>::quiet_NaN()}),
                         recall_name);
