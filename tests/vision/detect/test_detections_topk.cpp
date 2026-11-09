/**************************************************************************/
/* test_detections_topk.cpp                                               */
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
 * @file test_detections_topk.cpp
 * @brief Tests the top-k postprocess against hand cases and the reference.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/detections_topk.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>

namespace
{
    using SushiAI::Test::equal_reference;
    using SushiAI::Test::require_value;
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::PredictionShape;

    /** @brief Holds one detection row. */
    using Row = std::array<float, 6>;

    /** @brief Returns the rows of @p detections whose score exceeds @p floor, in a fixed order. */
    std::vector<float> sorted_rows_above(const std::vector<float>& detections, float floor)
    {
        std::vector<Row> rows;
        for (std::size_t at = 0; at + 6 <= detections.size(); at += 6)
        {
            if (!(detections[at + 4] > floor))
                continue;

            Row row{};
            std::copy(detections.begin() + at, detections.begin() + at + 6, row.begin());
            rows.push_back(row);
        }
        std::sort(rows.begin(), rows.end());

        std::vector<float> flat;
        for (const Row& row : rows)
            flat.insert(flat.end(), row.begin(), row.end());
        return flat;
    }

    /** @brief Returns the values at @p offset of every six-value row of @p rows. */
    std::vector<float> column(const std::vector<float>& rows, std::size_t offset)
    {
        std::vector<float> values;
        for (std::size_t at = offset; at < rows.size(); at += 6)
            values.push_back(rows[at]);
        return values;
    }

    /** @brief Reports whether @p row is some anchor's box with its score in the row's class. */
    bool is_anchor_class_pair(const float* row, const std::vector<float>& boxes,
                              const std::vector<float>& scores, std::size_t classes)
    {
        const std::size_t label = static_cast<std::size_t>(row[5]);
        for (std::size_t anchor = 0; anchor * 4 < boxes.size(); ++anchor)
        {
            if (std::equal(row, row + 4, boxes.begin() + anchor * 4) &&
                std::equal(row + 4, row + 5, scores.begin() + anchor * classes + label))
                return true;
        }
        return false;
    }

    /** @brief Reports whether every row of @p rows is some anchor's box with its class score. */
    bool is_every_row_an_anchor_class_pair(const std::vector<float>& rows,
                                           const std::vector<float>& boxes,
                                           const std::vector<float>& scores, std::size_t classes)
    {
        for (std::size_t at = 0; at < rows.size(); at += 6)
        {
            if (!is_anchor_class_pair(rows.data() + at, boxes, scores, classes))
                return false;
        }
        return true;
    }

    /**
     * @brief Holds the decoded boxes, the decoded scores and the detections of the YOLO26 forward
     *        reference, and skips every case of its suite when the file is absent.
     */
    class Medium_DetectPostprocessReference : public ::testing::Test
    {
    protected:
        /** @brief Opens the reference file and reads the three tensors the cases use. */
        void SetUp() override
        {
            SA_OPEN_REFERENCE_OR_SKIP(forward, "yolo26/forward.sacp");
            SA_READ_REFERENCE_OR_FAIL(box_tensor, forward, "one2one.decoded.boxes");
            SA_READ_REFERENCE_OR_FAIL(score_tensor, forward, "one2one.decoded.scores");
            SA_READ_REFERENCE_OR_FAIL(reference_tensor, forward, "one2one.detections");
            boxes = box_tensor.floats();
            scores = score_tensor.floats();
            reference = reference_tensor.floats();
        }

        std::vector<float> boxes;     /**< The decoded boxes of the reference, `[8400, 4]`. */
        std::vector<float> scores;    /**< The decoded scores of the reference, `[8400, 80]`. */
        std::vector<float> reference; /**< The reference detections, `[300, 6]`. */
    };

} // namespace

/** @brief Tests that the top-k path returns the best anchor and class pairs, best first. */
TEST(Medium_DetectPostprocess, SelectsTheHighestScoringAnchorClassPairs)
{
    const std::vector<float> boxes{0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f,
                                   3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f};
    const std::vector<float> scores{0.1f, 0.7f, 0.9f, 0.2f, 0.3f, 0.4f};
    const PredictionShape shape{1, 3, 2};
    std::vector<float> detections(12);
    SA_EXPECT_OK(SushiAI::Vision::Detect::select_top_detections(boxes, scores, shape, 2,
                                                                detections));

    // Anchors 1 and 0 lead with 0.9 and 0.7; anchor 1 wins with class 0 and anchor 0 with class 1.
    const std::vector<float> expected{2.0f, 2.0f, 3.0f, 3.0f, 0.9f, 0.0f,
                                      0.0f, 0.0f, 1.0f, 1.0f, 0.7f, 1.0f};
    EXPECT_TRUE(equal_reference(detections, expected));
}

/** @brief Tests that one anchor yields two detections when two of its classes score highest. */
TEST(Medium_DetectPostprocess, OneAnchorCanYieldTwoDetections)
{
    const std::vector<float> boxes{0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f,
                                   3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f};
    const std::vector<float> scores{0.6f, 0.1f, 0.8f, 0.7f, 0.2f, 0.1f};
    const PredictionShape shape{1, 3, 2};
    std::vector<float> detections(12);
    SA_EXPECT_OK(SushiAI::Vision::Detect::select_top_detections(boxes, scores, shape, 2,
                                                                detections));

    // Anchors 1 and 0 lead; the gathered rows are 0.8 0.7 0.6 0.1, so flat indices 0 and 1 win.
    const std::vector<float> expected{2.0f, 2.0f, 3.0f, 3.0f, 0.8f, 0.0f,
                                      2.0f, 2.0f, 3.0f, 3.0f, 0.7f, 1.0f};
    EXPECT_TRUE(equal_reference(detections, expected));
}

/** @brief Tests that the row count is the anchor count when the limit is larger. */
TEST(Small_DetectPostprocess, LimitsRowsToTheAnchorCount)
{
    EXPECT_EQ(require_value(
                  SushiAI::Vision::Detect::count_top_detections({1, 3, 2}, 300)),
              3);
    EXPECT_EQ(require_value(
                  SushiAI::Vision::Detect::count_top_detections({1, 8400, 80}, 300)),
              300);
}

/** @brief Tests that each image of a batch is selected from its own rows. */
TEST(Medium_DetectPostprocess, SelectsEachImageFromItsOwnRows)
{
    const std::vector<float> boxes{0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f,
                                   4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 6.0f, 7.0f, 7.0f};
    const std::vector<float> scores{0.2f, 0.3f, 0.6f, 0.1f, 0.5f, 0.4f, 0.1f, 0.9f};
    const PredictionShape shape{2, 2, 2};
    std::vector<float> detections(12);
    SA_EXPECT_OK(SushiAI::Vision::Detect::select_top_detections(boxes, scores, shape, 1,
                                                                detections));

    const std::vector<float> expected{2.0f, 2.0f, 3.0f, 3.0f, 0.6f, 0.0f,
                                      6.0f, 6.0f, 7.0f, 7.0f, 0.9f, 1.0f};
    EXPECT_TRUE(equal_reference(detections, expected));
}

/** @brief Tests that a detections array of the wrong size is refused with the needed shape. */
TEST(Small_DetectPostprocess, RejectsADetectionArrayOfTheWrongSize)
{
    const std::vector<float> boxes(12, 0.0f);
    const std::vector<float> scores(6, 0.0f);
    std::vector<float> detections(6);
    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::select_top_detections(boxes, scores, {1, 3, 2}, 2,
                                                                     detections),
                      StatusCode::InvalidDescription, "[1, 2, 6] detections", "6 values");
}

/** @brief Tests that the top-k path reproduces the reference detections up to tied scores. */
TEST_F(Medium_DetectPostprocessReference, ReproducesTheReferenceDetections)
{
    const PredictionShape shape{1, 8400, 80};
    std::vector<float> detections(reference.size());
    SA_EXPECT_OK(SushiAI::Vision::Detect::select_top_detections(
                     boxes, scores, shape, SushiAI::Vision::Detect::K_MAX_DETECTIONS, detections));

    EXPECT_TRUE(equal_reference(column(detections, 4), column(reference, 4)));
    EXPECT_TRUE(equal_reference(column(detections, 5), column(reference, 5)));

    // Rows sharing the last score are interchangeable; see modules/vision/detect/README.md.
    const float last_score = column(reference, 4).back();
    EXPECT_TRUE(equal_reference(sorted_rows_above(detections, last_score),
                                sorted_rows_above(reference, last_score)));
    EXPECT_TRUE(is_every_row_an_anchor_class_pair(detections, boxes, scores, 80));
}

/** @brief Tests that the confidence filter drops a score at the threshold and keeps one above. */
TEST(Medium_DetectPostprocess, TheConfidenceFilterDropsAScoreAtTheThreshold)
{
    const std::vector<float> rows{0.0f, 0.0f, 1.0f, 1.0f, 0.5f,  3.0f, 2.0f, 2.0f,
                                  3.0f, 3.0f, 0.25f, 1.0f, 4.0f, 4.0f, 5.0f, 5.0f,
                                  0.26f, 0.0f, 6.0f, 6.0f, 7.0f, 7.0f, 0.1f,  2.0f};
    std::vector<std::vector<float>> kept;
    SA_EXPECT_OK(SushiAI::Vision::Detect::keep_confident_detections(rows, 1, 0.25f, kept));

    // 0.5 and 0.26 exceed 0.25; the row at 0.25 and the row at 0.1 do not.
    const std::vector<float> expected{0.0f, 0.0f, 1.0f, 1.0f, 0.5f,  3.0f,
                                      4.0f, 4.0f, 5.0f, 5.0f, 0.26f, 0.0f};
    ASSERT_EQ(kept.size(), 1u);
    EXPECT_TRUE(equal_reference(kept[0], expected));
}

/** @brief Tests that the confidence filter keeps each image's rows apart. */
TEST(Medium_DetectPostprocess, TheConfidenceFilterKeepsEachImagesRows)
{
    const std::vector<float> rows{0.0f, 0.0f, 1.0f, 1.0f, 0.9f, 0.0f, 2.0f, 2.0f,
                                  3.0f, 3.0f, 0.0f, 1.0f, 4.0f, 4.0f, 5.0f, 5.0f,
                                  0.0f, 0.0f, 6.0f, 6.0f, 7.0f, 7.0f, 0.0f, 1.0f};
    std::vector<std::vector<float>> kept;
    SA_EXPECT_OK(SushiAI::Vision::Detect::keep_confident_detections(
                     rows, 2, SushiAI::Vision::Detect::K_VALIDATION_CONFIDENCE, kept));

    const std::vector<float> expected_first{0.0f, 0.0f, 1.0f, 1.0f, 0.9f, 0.0f};
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_TRUE(equal_reference(kept[0], expected_first));
    EXPECT_TRUE(kept[1].empty());
}

/** @brief Tests that the confidence filter refuses rows the batch does not divide evenly. */
TEST(Medium_DetectPostprocess, TheConfidenceFilterRejectsRowsTheBatchDoesNotDivide)
{
    // Three rows of six values cannot be split between two images.
    const std::vector<float> rows(18, 0.0f);
    std::vector<std::vector<float>> kept;

    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::keep_confident_detections(rows, 2, 0.25f, kept),
                      StatusCode::InvalidDescription, "[2, k, 6] detections", "18 values");
}

/** @brief Tests that the confidence filter refuses a batch of 0. */
TEST(Medium_DetectPostprocess, TheConfidenceFilterRejectsABatchOfZero)
{
    const std::vector<float> rows(18, 0.0f);
    std::vector<std::vector<float>> kept;

    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::keep_confident_detections(rows, 0, 0.25f, kept),
                      StatusCode::InvalidDescription, "positive batch", "got 0");
}

/** @brief Tests that the confidence filter keeps the reference's rows as its validator does. */
TEST_F(Medium_DetectPostprocessReference, TheConfidenceFilterKeepsTheRowsTheReferenceValidatorKeeps)
{
    std::vector<std::vector<float>> confident;
    std::vector<std::vector<float>> positive;
    SA_EXPECT_OK(SushiAI::Vision::Detect::keep_confident_detections(
                     reference, 1, SushiAI::Vision::Detect::K_VALIDATION_CONFIDENCE, confident));
    SA_EXPECT_OK(SushiAI::Vision::Detect::keep_confident_detections(reference, 1, 0.0f, positive));

    // 0 is "validation.row_count.one2one" of forward.sacp in MANIFEST.json: no score reaches 0.001.
    ASSERT_EQ(confident.size(), 1u);
    EXPECT_TRUE(confident[0].empty());

    // Every reference score is a sigmoid above zero, so a threshold of zero keeps all 300 rows.
    ASSERT_EQ(positive.size(), 1u);
    EXPECT_TRUE(equal_reference(positive[0], reference));
}

/** @brief Tests that a limit below 1 is refused with the extents it was given. */
TEST(Small_DetectPostprocess, CountingRefusesALimitBelowOne)
{
    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::count_top_detections({1, 3, 2}, 0),
                      StatusCode::InvalidDescription, "positive extents", "a limit of 0");
}

/** @brief Tests that an extent that is not positive is refused with the extents it was given. */
TEST(Small_DetectPostprocess, CountingRefusesAnExtentThatIsNotPositive)
{
    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::count_top_detections({1, 0, 2}, 300),
                      StatusCode::InvalidDescription, "positive extents", "0 anchors");
}
