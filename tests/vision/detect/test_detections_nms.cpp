/**************************************************************************/
/* test_detections_nms.cpp                                                */
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
 * @file test_detections_nms.cpp
 * @brief Tests the suppression postprocess against hand cases and the reference.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Test::equal_reference;
    using SushiAI::Vision::Detect::ClassMode;
    using SushiAI::Vision::Detect::LabelMode;
    using SushiAI::Vision::Detect::NMSConfig;
    using SushiAI::Vision::Detect::PredictionShape;

    /** @brief Holds "suppression.arguments.conf_thres" of forward.sacp in MANIFEST.json. */
    constexpr float K_SUPPRESSION_CONFIDENCE = 1.8e-4f;

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

    /** @brief Returns a suppression config with one threshold pair and the reference's limits. */
    NMSConfig make_nms_config(float confidence, float iou, LabelMode labels, ClassMode classes)
    {
        NMSConfig config;
        config.confidence_threshold = confidence;
        config.iou_threshold = iou;
        config.label_mode = labels;
        config.class_mode = classes;
        return config;
    }

    /** @brief Returns the suppression path's detections of one image of `[A, 4]` boxes. */
    std::vector<float> suppress_one_image(const std::vector<float>& boxes,
                                          const std::vector<float>& scores, std::int64_t classes,
                                          const NMSConfig& config)
    {
        const PredictionShape shape{1, static_cast<std::int64_t>(boxes.size() / 4), classes};
        std::vector<std::vector<float>> detections;
        SA_EXPECT_OK(SushiAI::Vision::Detect::select_nms_detections(boxes, scores, shape, config,
                                                                    detections));
        return detections.front();
    }

    /**
     * @brief Holds the decoded boxes, the decoded scores and the suppressed detections of the
     *        YOLO26 forward reference, and skips every case of its suite when the file is absent.
     */
    class Medium_DetectPostprocessSuppressionReference : public ::testing::Test
    {
    protected:
        /** @brief Opens the reference file and reads the three tensors the cases use. */
        void SetUp() override
        {
            SA_OPEN_REFERENCE_OR_SKIP(forward, "yolo26/forward.sacp");
            SA_READ_REFERENCE_OR_FAIL(box_tensor, forward, "one2many.decoded.boxes");
            SA_READ_REFERENCE_OR_FAIL(score_tensor, forward, "one2many.decoded.scores");
            SA_READ_REFERENCE_OR_FAIL(reference_tensor, forward, "one2many.suppressed.detections");
            boxes = box_tensor.floats();
            scores = score_tensor.floats();
            reference = reference_tensor.floats();
        }

        std::vector<float> boxes;     /**< The decoded boxes of the reference, `[8400, 4]`. */
        std::vector<float> scores;    /**< The decoded scores of the reference, `[8400, 80]`. */
        std::vector<float> reference; /**< The suppressed reference detections, `[300, 6]`. */
    };

} // namespace

/** @brief Tests that the suppression path reproduces the reference rows up to tied scores. */
TEST_F(Medium_DetectPostprocessSuppressionReference, SuppressionReproducesTheReferenceRows)
{
    NMSConfig config;
    config.confidence_threshold = K_SUPPRESSION_CONFIDENCE;
    std::vector<std::vector<float>> detections;
    SA_EXPECT_OK(SushiAI::Vision::Detect::select_nms_detections(
        boxes, scores, PredictionShape{1, 8400, 80}, config, detections));

    // 1542 scores pass and the reference keeps 300 rows: "suppression" in MANIFEST.json.
    ASSERT_EQ(detections.size(), 1u);
    ASSERT_EQ(detections[0].size(), reference.size());
    EXPECT_TRUE(equal_reference(column(detections[0], 4), column(reference, 4)));

    // Rows sharing the last score are interchangeable; see modules/vision/detect/README.md.
    const float last_score = column(reference, 4).back();
    EXPECT_TRUE(equal_reference(sorted_rows_above(detections[0], last_score),
                                sorted_rows_above(reference, last_score)));
}

/** @brief Tests that no reference score passes the validation confidence, as the manifest says. */
TEST_F(Medium_DetectPostprocessSuppressionReference, NoReferenceScorePassesTheValidationConfidence)
{
    std::vector<std::vector<float>> detections;
    SA_EXPECT_OK(SushiAI::Vision::Detect::select_nms_detections(
        boxes, scores, PredictionShape{1, 8400, 80}, NMSConfig{}, detections));

    // 0 is "validation.row_count.one2many" of forward.sacp in MANIFEST.json.
    ASSERT_EQ(detections.size(), 1u);
    EXPECT_TRUE(detections[0].empty());
}

/**
 * @brief Tests that a default configuration passes what the reference validator passes.
 *
 * Boxes 0 and 1 are 10 by 10 squares shifted by 1, an IoU of 81 / 119, 0.68, below the default
 * threshold 0.7, so both survive. Box 2 scores 0.001, which is the default confidence, and is
 * dropped.
 */
TEST(Small_DetectPostprocess, SuppressionDefaultsMatchTheValidator)
{
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f, 6.0f,  6.0f,
                                   10.0f, 10.0f, 45.0f, 45.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.9f, 0.8f, 0.001f};
    const std::vector<float> detections = suppress_one_image(boxes, scores, 1, NMSConfig{});

    const std::vector<float> expected{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0.0f,
                                      1.0f, 1.0f, 11.0f, 11.0f, 0.8f, 0.0f};
    EXPECT_TRUE(equal_reference(detections, expected));
}

/** @brief Tests that the lower-scoring of two overlapping boxes is suppressed. */
TEST(Small_DetectPostprocess, SuppressesTheLowerOfTwoOverlappingBoxes)
{
    const std::vector<float> boxes{0.0f,  0.0f,  10.0f, 10.0f, 1.0f,  1.0f,
                                   11.0f, 11.0f, 20.0f, 20.0f, 30.0f, 30.0f};
    const std::vector<float> scores{0.9f, 0.8f, 0.7f};
    std::vector<std::int64_t> kept;
    SushiAI::Vision::Detect::suppress_overlaps(boxes, scores, 0.5f, kept);

    // Boxes 0 and 1 share 9 * 9 = 81 of 100 + 100 - 81 = 119, which is 0.68 and above 0.5.
    EXPECT_TRUE(equal_reference(kept, std::vector<std::int64_t>{0, 2}));
}

/** @brief Tests that survivors are listed by descending score, equal scores by index. */
TEST(Small_DetectPostprocess, ListsSurvivorsByDescendingScore)
{
    const std::vector<float> boxes{0.0f,  0.0f,  1.0f,  1.0f,  10.0f, 10.0f, 11.0f, 11.0f,
                                   20.0f, 20.0f, 21.0f, 21.0f, 30.0f, 30.0f, 31.0f, 31.0f};
    const std::vector<float> scores{0.1f, 0.9f, 0.5f, 0.5f};
    std::vector<std::int64_t> kept;
    SushiAI::Vision::Detect::suppress_overlaps(boxes, scores, 0.5f, kept);

    EXPECT_TRUE(equal_reference(kept, std::vector<std::int64_t>{1, 2, 3, 0}));
}

/** @brief Tests that an IoU equal to the threshold survives and one above it does not. */
TEST(Small_DetectPostprocess, KeepsABoxWhoseIoUEqualsTheThreshold)
{
    const std::vector<float> boxes{0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 0.0f, 2.0f, 1.0f};
    const std::vector<float> scores{0.9f, 0.8f};
    std::vector<std::int64_t> at_threshold;
    std::vector<std::int64_t> below_iou;
    SushiAI::Vision::Detect::suppress_overlaps(boxes, scores, 0.5f, at_threshold);
    SushiAI::Vision::Detect::suppress_overlaps(boxes, scores, 0.25f, below_iou);

    // The boxes share 2 * 1 = 2 of 4 + 2 - 2 = 4, an IoU of exactly 0.5 in float32.
    EXPECT_TRUE(equal_reference(at_threshold, std::vector<std::int64_t>{0, 1}));
    EXPECT_TRUE(equal_reference(below_iou, std::vector<std::int64_t>{0}));
}

/** @brief Tests that overlapping boxes of one class collapse to the higher score. */
TEST(Medium_DetectPostprocess, OverlappingBoxesOfOneClassCollapse)
{
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f, 5.0f, 5.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.9f, 0.0f, 0.8f, 0.0f};
    const std::vector<float> detections = suppress_one_image(
        boxes, scores, 2,
        make_nms_config(0.25f, 0.5f, LabelMode::MultiLabel, ClassMode::ClassAware));

    // Centre 5 and size 10 are the corners 0 and 10.
    const std::vector<float> expected{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0.0f};
    EXPECT_TRUE(equal_reference(detections, expected));
}

/** @brief Tests that overlapping boxes of different classes both survive a class-aware pass. */
TEST(Medium_DetectPostprocess, OverlappingBoxesOfDifferentClassesBothSurvive)
{
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f, 5.0f, 5.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.9f, 0.0f, 0.0f, 0.8f};
    const std::vector<float> aware = suppress_one_image(
        boxes, scores, 2,
        make_nms_config(0.25f, 0.5f, LabelMode::MultiLabel, ClassMode::ClassAware));
    const std::vector<float> agnostic = suppress_one_image(
        boxes, scores, 2,
        make_nms_config(0.25f, 0.5f, LabelMode::MultiLabel, ClassMode::ClassAgnostic));

    const std::vector<float> expected_aware{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0.0f,
                                            0.0f, 0.0f, 10.0f, 10.0f, 0.8f, 1.0f};
    const std::vector<float> expected_agnostic{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0.0f};
    EXPECT_TRUE(equal_reference(aware, expected_aware));
    EXPECT_TRUE(equal_reference(agnostic, expected_agnostic));
}

/** @brief Tests that a score equal to the confidence threshold is dropped. */
TEST(Medium_DetectPostprocess, DropsAScoreEqualToTheConfidenceThreshold)
{
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f, 25.0f, 25.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.25f, 0.0f, 0.5f, 0.0f};
    const std::vector<float> detections = suppress_one_image(
        boxes, scores, 2,
        make_nms_config(0.25f, 0.5f, LabelMode::MultiLabel, ClassMode::ClassAware));

    const std::vector<float> expected{20.0f, 20.0f, 30.0f, 30.0f, 0.5f, 0.0f};
    EXPECT_TRUE(equal_reference(detections, expected));
}

/** @brief Tests that an anchor contributes every passing class under MultiLabel. */
TEST(Medium_DetectPostprocess, AnAnchorContributesEveryPassingClass)
{
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.6f, 0.1f, 0.7f};
    const std::vector<float> every = suppress_one_image(
        boxes, scores, 3,
        make_nms_config(0.25f, 0.5f, LabelMode::MultiLabel, ClassMode::ClassAware));

    const std::vector<float> expected_every{0.0f, 0.0f, 10.0f, 10.0f, 0.7f, 2.0f,
                                            0.0f, 0.0f, 10.0f, 10.0f, 0.6f, 0.0f};
    EXPECT_TRUE(equal_reference(every, expected_every));
}

/** @brief Tests that an anchor contributes only its best class under BestClass. */
TEST(Medium_DetectPostprocess, AnAnchorContributesItsBestClassUnderBestClass)
{
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.6f, 0.1f, 0.7f};
    const std::vector<float> best = suppress_one_image(
        boxes, scores, 3,
        make_nms_config(0.25f, 0.5f, LabelMode::BestClass, ClassMode::ClassAware));

    const std::vector<float> expected_best{0.0f, 0.0f, 10.0f, 10.0f, 0.7f, 2.0f};
    EXPECT_TRUE(equal_reference(best, expected_best));
}

/** @brief Tests that the detection limit keeps the highest-scoring survivors. */
TEST(Medium_DetectPostprocess, LimitsDetectionsToTheHighestScores)
{
    const std::vector<float> boxes{5.0f,  5.0f,  10.0f, 10.0f, 25.0f, 25.0f,
                                   10.0f, 10.0f, 45.0f, 45.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.5f, 0.9f, 0.7f};
    NMSConfig config = make_nms_config(0.25f, 0.5f, LabelMode::BestClass, ClassMode::ClassAware);
    config.max_detections = 2;
    const std::vector<float> detections = suppress_one_image(boxes, scores, 1, config);

    const std::vector<float> expected{20.0f, 20.0f, 30.0f, 30.0f, 0.9f, 0.0f,
                                      40.0f, 40.0f, 50.0f, 50.0f, 0.7f, 0.0f};
    EXPECT_TRUE(equal_reference(detections, expected));
}

/** @brief Tests that the candidate limit drops the lowest scores before suppression. */
TEST(Medium_DetectPostprocess, LimitsCandidatesBeforeSuppression)
{
    const std::vector<float> boxes{5.0f,  5.0f,  10.0f, 10.0f, 25.0f, 25.0f,
                                   10.0f, 10.0f, 45.0f, 45.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.5f, 0.9f, 0.7f};
    NMSConfig config = make_nms_config(0.25f, 0.5f, LabelMode::BestClass, ClassMode::ClassAware);
    config.max_candidates = 1;
    const std::vector<float> detections = suppress_one_image(boxes, scores, 1, config);

    const std::vector<float> expected{20.0f, 20.0f, 30.0f, 30.0f, 0.9f, 0.0f};
    EXPECT_TRUE(equal_reference(detections, expected));
}

/** @brief Tests that a threshold outside [0, 1] is refused with its value. */
TEST(Small_DetectPostprocess, RejectsAThresholdOutsideTheUnitInterval)
{
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.6f};
    std::vector<std::vector<float>> detections;
    SA_EXPECT_FAILURE(
        SushiAI::Vision::Detect::select_nms_detections(
            boxes, scores, PredictionShape{1, 1, 1},
            make_nms_config(1.5f, 0.5f, LabelMode::BestClass, ClassMode::ClassAware), detections),
        StatusCode::InvalidDescription, "[0, 1]", "1.5");
}

/** @brief Tests that an image whose scores all stay at or below the threshold yields no rows. */
TEST(Small_DetectPostprocess, AnImageBelowTheThresholdYieldsNoRows)
{
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f, 25.0f, 25.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.25f, 0.1f, 0.2f, 0.0f};
    const std::vector<float> detections = suppress_one_image(
        boxes, scores, 2,
        make_nms_config(0.25f, 0.5f, LabelMode::MultiLabel, ClassMode::ClassAware));

    EXPECT_TRUE(detections.empty());
}

/** @brief Tests that each image of a batch is suppressed from its own boxes and scores. */
TEST(Medium_DetectPostprocess, SuppressesEachImageFromItsOwnRows)
{
    const std::vector<float> boxes{5.0f,  5.0f,  10.0f, 10.0f, 5.0f,  5.0f,  10.0f, 10.0f,
                                   25.0f, 25.0f, 10.0f, 10.0f, 45.0f, 45.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.9f, 0.8f, 0.3f, 0.6f};
    std::vector<std::vector<float>> detections;
    SA_EXPECT_OK(SushiAI::Vision::Detect::select_nms_detections(
        boxes, scores, PredictionShape{2, 2, 1},
        make_nms_config(0.25f, 0.5f, LabelMode::BestClass, ClassMode::ClassAware), detections));

    // Image 0 holds one box twice, so 0.8 falls to 0.9; image 1 holds two boxes that are apart.
    const std::vector<float> expected_first{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0.0f};
    const std::vector<float> expected_second{40.0f, 40.0f, 50.0f, 50.0f, 0.6f, 0.0f,
                                             20.0f, 20.0f, 30.0f, 30.0f, 0.3f, 0.0f};
    ASSERT_EQ(detections.size(), 2u);
    EXPECT_TRUE(equal_reference(detections[0], expected_first));
    EXPECT_TRUE(equal_reference(detections[1], expected_second));
}

/** @brief Tests that scores that do not fill the shape are refused, not read as other classes. */
TEST(Small_DetectPostprocess, RejectsScoresThatDoNotFillTheShape)
{
    const std::vector<float> boxes(8, 0.0f);
    const std::vector<float> scores(6, 0.5f);
    std::vector<std::vector<float>> detections;
    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::select_nms_detections(
                          boxes, scores, PredictionShape{1, 2, 2}, NMSConfig{}, detections),
                      StatusCode::InvalidDescription, "[1, 2, 2] scores", "8 and 6 values");
}

/** @brief Tests that a NaN first class score does not hide the finite row maximum. */
TEST(Small_DetectPostprocess, SuppressionKeepsTheFiniteRowMaximumBesideANaN)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> boxes{10.0f, 10.0f, 4.0f, 4.0f};
    const std::vector<float> scores{nan, 0.9f, 0.2f};
    const NMSConfig config =
        make_nms_config(0.25f, 0.45f, LabelMode::BestClass, ClassMode::ClassAware);

    const std::vector<float> detections = suppress_one_image(boxes, scores, 3, config);

    ASSERT_EQ(detections.size(), 6u);
    EXPECT_EQ(detections[4], 0.9f);
    EXPECT_EQ(detections[5], 1.0f);
}

/** @brief Tests that an extent that is not positive is refused with the extents it was given. */
TEST(Small_DetectPostprocess, SuppressionRefusesAnExtentThatIsNotPositive)
{
    std::vector<std::vector<float>> detections;

    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::select_nms_detections(
                          {}, {}, PredictionShape{1, 0, 1}, NMSConfig{}, detections),
                      StatusCode::InvalidDescription, "positive extents", "0 anchors");
}

/** @brief Tests that a detection limit below 1 is refused with the limits it was given. */
TEST(Small_DetectPostprocess, SuppressionRefusesADetectionLimitBelowOne)
{
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.6f};
    NMSConfig config;
    config.max_detections = 0;
    std::vector<std::vector<float>> detections;

    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::select_nms_detections(
                          boxes, scores, PredictionShape{1, 1, 1}, config, detections),
                      StatusCode::InvalidDescription, "must be positive", "got 0 and 30000");
}

/** @brief Tests that a candidate limit below 1 is refused with the limits it was given. */
TEST(Small_DetectPostprocess, SuppressionRefusesACandidateLimitBelowOne)
{
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.6f};
    NMSConfig config;
    config.max_candidates = 0;
    std::vector<std::vector<float>> detections;

    SA_EXPECT_FAILURE(SushiAI::Vision::Detect::select_nms_detections(
                          boxes, scores, PredictionShape{1, 1, 1}, config, detections),
                      StatusCode::InvalidDescription, "must be positive", "got 300 and 0");
}
