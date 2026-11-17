/**************************************************************************/
/* test_yolox_decode.cpp                                                  */
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
 * @file test_yolox_decode.cpp
 * @brief Tests the decoding of a YOLOX head's outputs against boxes and scores worked by hand.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/yolox/yolox_decode.hpp>
#include <SushiAI/vision/detect/yolox/yolox_task.hpp>
#include <support/tolerance.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::ClassMode;
    using SushiAI::Vision::Detect::LabelMode;
    using SushiAI::Vision::Detect::NMSConfig;
    using SushiAI::Vision::Detect::SuppressionCriterion;
    using SushiAI::Vision::Detect::Yolox::YoloxDecoded;
    using SushiAI::Vision::Detect::Yolox::YoloxLevel;
    using SushiAI::Vision::Detect::Yolox::YoloxTask;
    using SushiAI::Vision::Detect::Yolox::decode_yolox;
    using SushiAI::Vision::Detect::Yolox::make_yolox_nms_config;
    using SushiAI::Vision::Detect::Yolox::select_yolox_detections;

    /**
     * @brief Holds the relative bound of a decoded value that passes through an exponential and a
     *        product: four float roundings, the exponential's, the product's and two spare for
     *        the sum of the cell offset.
     */
    constexpr float K_RELATIVE = static_cast<float>(4.0 * TestSupport::unit_roundoff<float>());

    /** @brief Holds a logit whose float32 logistic is exactly 1, and exactly 0 when negated. */
    constexpr float K_SATURATED = 100.0f;

    /** @brief Holds the values of one level of a batch and the view over them. */
    struct LevelData
    {
        std::vector<float> classes;    /**< `[N, H, W, C]`, zero until set. */
        std::vector<float> boxes;      /**< `[N, H, W, 4]`, zero until set. */
        std::vector<float> objectness; /**< `[N, H, W, 1]`, zero until set. */
        std::int64_t height;           /**< Rows of the grid. */
        std::int64_t width;            /**< Columns of the grid. */
        float stride;                  /**< Pixels per cell. */

        /** @brief Sizes the three arrays for @p batch images of @p class_count classes. */
        LevelData(std::int64_t batch, std::int64_t grid_height, std::int64_t grid_width,
                  std::int64_t class_count, float cell_stride)
            : classes(static_cast<std::size_t>(batch * grid_height * grid_width * class_count)),
              boxes(static_cast<std::size_t>(batch * grid_height * grid_width * 4)),
              objectness(static_cast<std::size_t>(batch * grid_height * grid_width)),
              height(grid_height), width(grid_width), stride(cell_stride)
        {

        }

        /** @brief Returns the view of the arrays that decode_yolox() reads. */
        YoloxLevel view() const
        {
            YoloxLevel level;
            level.height = height;
            level.width = width;
            level.stride = stride;
            level.classes = SushiRuntime::span<const float>(classes.data(), classes.size());
            level.boxes = SushiRuntime::span<const float>(boxes.data(), boxes.size());
            level.objectness =
                SushiRuntime::span<const float>(objectness.data(), objectness.size());

            return level;
        }
    };

    /** @brief Expects @p found to be within the relative bound K_RELATIVE of @p wanted. */
    void expect_near_relative(float found, double wanted)
    {
        EXPECT_NEAR(found, wanted, K_RELATIVE * std::fabs(wanted));
    }
} // namespace

/** @brief Tests that a cell's centre is its grid index plus the offset, times the stride. */
TEST(Small_DetectYoloxDecode, TheCentreIsTheCellPlusTheOffsetTimesTheStride)
{
    LevelData level(1, 2, 2, 1, 8.0f);
    // The cell at row 1, column 0 is the third cell in row-major order.
    level.boxes[2 * 4 + 0] = 0.25f;
    level.boxes[2 * 4 + 1] = 0.5f;
    const YoloxLevel view = level.view();

    YoloxDecoded decoded;
    SA_EXPECT_OK(decode_yolox(SushiRuntime::span<const YoloxLevel>(&view, 1), 1, 1, decoded));

    ASSERT_EQ(decoded.boxes.size(), 16u);
    // Column 0 plus 0.25 is 0.25, row 1 plus 0.5 is 1.5; both times 8, and every value is exact.
    EXPECT_EQ(decoded.boxes[2 * 4 + 0], 2.0f);
    EXPECT_EQ(decoded.boxes[2 * 4 + 1], 12.0f);
}

/** @brief Tests that the size is the exponential of the raw value times the stride. */
TEST(Small_DetectYoloxDecode, TheSizeIsTheExponentialTimesTheStride)
{
    LevelData level(1, 1, 1, 1, 16.0f);
    level.boxes[2] = 0.0f;
    level.boxes[3] = 1.0f;
    const YoloxLevel view = level.view();

    YoloxDecoded decoded;
    SA_EXPECT_OK(decode_yolox(SushiRuntime::span<const YoloxLevel>(&view, 1), 1, 1, decoded));

    EXPECT_EQ(decoded.boxes[2], 16.0f);
    expect_near_relative(decoded.boxes[3], std::exp(1.0) * 16.0);
}

/** @brief Tests that a score is the objectness probability times the class probability. */
TEST(Small_DetectYoloxDecode, AScoreIsTheObjectnessTimesTheClassProbability)
{
    LevelData level(1, 1, 2, 2, 8.0f);
    // Cell 0: objectness logit 0 is 0.5; class logits 0 and saturated give 0.5 * 0.5, 0.5 * 1.
    level.objectness[0] = 0.0f;
    level.classes[0] = 0.0f;
    level.classes[1] = K_SATURATED;
    // Cell 1: a saturated objectness gives 1, so the scores are 0 and 1 from the class logits.
    level.objectness[1] = K_SATURATED;
    level.classes[2] = -K_SATURATED;
    level.classes[3] = K_SATURATED;
    const YoloxLevel view = level.view();

    YoloxDecoded decoded;
    SA_EXPECT_OK(decode_yolox(SushiRuntime::span<const YoloxLevel>(&view, 1), 1, 2, decoded));

    ASSERT_EQ(decoded.scores.size(), 4u);
    EXPECT_EQ(decoded.scores[0], 0.25f);
    EXPECT_EQ(decoded.scores[1], 0.5f);
    EXPECT_EQ(decoded.scores[2], 0.0f);
    EXPECT_EQ(decoded.scores[3], 1.0f);
}

/** @brief Tests that anchors run level by level, each with its own stride and grid. */
TEST(Small_DetectYoloxDecode, AnchorsRunLevelByLevel)
{
    LevelData fine(1, 1, 2, 1, 8.0f);
    LevelData coarse(1, 1, 1, 1, 16.0f);
    coarse.boxes[0] = 0.5f;
    const YoloxLevel views[] = {fine.view(), coarse.view()};

    YoloxDecoded decoded;
    SA_EXPECT_OK(decode_yolox(SushiRuntime::span<const YoloxLevel>(views, 2), 1, 1, decoded));

    EXPECT_EQ(decoded.shape.batch, 1);
    EXPECT_EQ(decoded.shape.anchors, 3);
    EXPECT_EQ(decoded.shape.num_classes, 1);
    // The fine level's second cell: column 1, offset 0, stride 8.
    EXPECT_EQ(decoded.boxes[1 * 4 + 0], 8.0f);
    // The coarse level's only cell is anchor 2: 0.5 times 16.
    EXPECT_EQ(decoded.boxes[2 * 4 + 0], 8.0f);
    EXPECT_EQ(decoded.boxes[2 * 4 + 2], 16.0f);
}

/** @brief Tests that the second image of a batch decodes from its own slice of each array. */
TEST(Small_DetectYoloxDecode, EachImageReadsItsOwnSlice)
{
    LevelData level(2, 1, 1, 1, 8.0f);
    level.boxes[0] = 0.0f;
    level.boxes[4] = 0.5f;
    level.classes[1] = K_SATURATED;
    level.objectness[1] = K_SATURATED;
    const YoloxLevel view = level.view();

    YoloxDecoded decoded;
    SA_EXPECT_OK(decode_yolox(SushiRuntime::span<const YoloxLevel>(&view, 1), 2, 1, decoded));

    ASSERT_EQ(decoded.shape.anchors, 1);
    EXPECT_EQ(decoded.boxes[0], 0.0f);
    EXPECT_EQ(decoded.boxes[4], 4.0f);
    EXPECT_EQ(decoded.scores[0], 0.25f);
    EXPECT_EQ(decoded.scores[1], 1.0f);
}

/** @brief Tests that no level is refused. */
TEST(Small_DetectYoloxDecode, RefusesNoLevel)
{
    YoloxDecoded decoded;

    SA_EXPECT_FAILURE(decode_yolox(SushiRuntime::span<const YoloxLevel>(), 1, 1, decoded),
                      StatusCode::InvalidDescription, "at least one level");
}

/** @brief Tests that a span of the wrong size is refused, naming the level and both counts. */
TEST(Small_DetectYoloxDecode, RefusesASpanThatDoesNotFitItsLevel)
{
    LevelData level(1, 2, 2, 3, 8.0f);
    YoloxLevel view = level.view();
    view.classes = SushiRuntime::span<const float>(level.classes.data(), 5);
    YoloxDecoded decoded;

    SA_EXPECT_FAILURE(decode_yolox(SushiRuntime::span<const YoloxLevel>(&view, 1), 1, 3, decoded),
                      StatusCode::InvalidDescription, "level 0", "5 class values", "need 12");
}

/** @brief Tests that a stride that is not positive is refused, naming the level. */
TEST(Small_DetectYoloxDecode, RefusesAStrideThatIsNotPositive)
{
    LevelData level(1, 1, 1, 1, 0.0f);
    const YoloxLevel view = level.view();
    YoloxDecoded decoded;

    SA_EXPECT_FAILURE(decode_yolox(SushiRuntime::span<const YoloxLevel>(&view, 1), 1, 1, decoded),
                      StatusCode::InvalidDescription, "level 0", "stride");
}

/** @brief Tests that the suppression settings follow the task and the best class of an anchor. */
TEST(Small_DetectYoloxDecode, TheNmsConfigFollowsTheTask)
{
    YoloxTask task;
    task.score_threshold = 0.125f;
    task.suppression.criterion = SuppressionCriterion::DistanceIoU;
    task.suppression.threshold = 0.5f;
    task.suppression.distance_exponent = 0.75f;

    const NMSConfig config = make_yolox_nms_config(task);

    EXPECT_EQ(config.confidence_threshold, 0.125f);
    EXPECT_EQ(config.iou_threshold, 0.5f);
    EXPECT_EQ(config.criterion, SuppressionCriterion::DistanceIoU);
    EXPECT_EQ(config.distance_exponent, 0.75f);
    EXPECT_EQ(config.label_mode, LabelMode::BestClass);
    EXPECT_EQ(config.class_mode, ClassMode::ClassAware);
}

/**
 * @brief Tests the whole path on three cells of one row: the second repeats the first box with a
 *        lower score and is suppressed, the third stands apart and stays.
 */
TEST(Small_DetectYoloxDecode, ADuplicateBoxIsSuppressedAndAnIsolatedOneStays)
{
    LevelData level(1, 1, 3, 1, 8.0f);
    // Cell 0 is the box (0, 0, 8, 8) at 0.5 times 1.
    level.boxes[0] = 0.5f;
    level.boxes[1] = 0.5f;
    level.classes[0] = K_SATURATED;
    // Cell 1 shifts left half a cell: centre 4, cell 0's box, score 0.5 times 0.5.
    level.boxes[4] = -0.5f;
    level.boxes[5] = 0.5f;
    // Cell 2 is the box (16, 0, 24, 8), also at 0.5 times 0.5.
    level.boxes[8] = 0.5f;
    level.boxes[9] = 0.5f;
    const YoloxLevel view = level.view();
    YoloxTask task;
    task.score_threshold = 0.125f;

    std::vector<std::vector<float>> detections;
    SA_EXPECT_OK(select_yolox_detections(SushiRuntime::span<const YoloxLevel>(&view, 1), 1, 1,
                                         task, detections));

    ASSERT_EQ(detections.size(), 1u);
    const std::vector<float> wanted = {0.0f, 0.0f, 8.0f, 8.0f, 0.5f, 0.0f,
                                       16.0f, 0.0f, 24.0f, 8.0f, 0.25f, 0.0f};
    EXPECT_EQ(detections[0], wanted);
}
