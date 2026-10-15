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
 * @file test_detections_nms.cpp
 * @brief Tests the suppression postprocess against hand cases and the reference.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/detections_topk.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/detect/top_k.hpp>

namespace
{
    using SushiAI::Vision::Detect::ClassMode;

    using SushiAI::Vision::Detect::LabelMode;

    using SushiAI::Vision::Detect::NMSConfig;

    using SushiAI::Vision::Detect::PredictionShape;

    using SushiAI::Test::equal_reference;

    using SushiAI::Test::near_reference;

    /** @brief Holds the tolerance of values a postprocess copies from its inputs. */
    constexpr float K_EXACT = 0.0f;

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
        SushiAI::Vision::Detect::select_nms_detections(boxes, scores, shape, config, detections);
        return detections.front();
    }

} // namespace

/** @brief Tests that the suppression path reproduces the reference rows up to tied scores. */
TEST(Medium_DetectPostprocess, SuppressionReproducesTheReferenceRows)
{
    SA_OPEN_REFERENCE_OR_SKIP(forward, "yolo26/forward.sacp");
    SA_READ_REFERENCE_OR_FAIL(box_tensor, forward, "one2many.decoded.boxes");
    SA_READ_REFERENCE_OR_FAIL(score_tensor, forward, "one2many.decoded.scores");
    SA_READ_REFERENCE_OR_FAIL(reference_tensor, forward, "one2many.suppressed.detections");

    const std::vector<float> reference = reference_tensor.floats();
    NMSConfig config;
    config.confidence_threshold = K_SUPPRESSION_CONFIDENCE;
    std::vector<std::vector<float>> detections;
    SushiAI::Vision::Detect::select_nms_detections(
        box_tensor.floats(), score_tensor.floats(), PredictionShape{1, 8400, 80}, config,
        detections);

    // 1542 scores pass and the reference keeps 300 rows: "suppression" in MANIFEST.json.
    ASSERT_EQ(detections.size(), 1u);
    ASSERT_EQ(detections[0].size(), reference.size());
    std::vector<float> score_column;
    std::vector<float> reference_scores;
    for (std::size_t at = 0; at < reference.size(); at += 6)
    {
        score_column.push_back(detections[0][at + 4]);
        reference_scores.push_back(reference[at + 4]);
    }
    EXPECT_TRUE(near_reference(score_column, reference_scores, K_EXACT));

    // Rows sharing the last score are interchangeable; see modules/vision/detect/README.md.
    const float last_score = reference_scores.back();
    EXPECT_TRUE(near_reference(sorted_rows_above(detections[0], last_score),
                               sorted_rows_above(reference, last_score), K_EXACT));
}

/** @brief Tests that no reference score passes the validation confidence, as the manifest says. */
TEST(Medium_DetectPostprocess, NoReferenceScorePassesTheValidationConfidence)
{
    SA_OPEN_REFERENCE_OR_SKIP(forward, "yolo26/forward.sacp");
    SA_READ_REFERENCE_OR_FAIL(box_tensor, forward, "one2many.decoded.boxes");
    SA_READ_REFERENCE_OR_FAIL(score_tensor, forward, "one2many.decoded.scores");

    std::vector<std::vector<float>> detections;
    SushiAI::Vision::Detect::select_nms_detections(
        box_tensor.floats(), score_tensor.floats(), PredictionShape{1, 8400, 80}, NMSConfig{},
        detections);

    // 0 is "validation.row_count.one2many" of forward.sacp in MANIFEST.json.
    ASSERT_EQ(detections.size(), 1u);
    EXPECT_TRUE(detections[0].empty());
}

/** @brief Tests that the suppression settings default to what the reference validator passes. */
TEST(Medium_DetectPostprocess, SuppressionDefaultsMatchTheValidator)
{
    const NMSConfig config;

    EXPECT_NEAR(config.confidence_threshold, 0.001f, K_EXACT);
    EXPECT_NEAR(config.iou_threshold, 0.7f, K_EXACT);
    EXPECT_EQ(config.label_mode, LabelMode::MultiLabel);
    EXPECT_EQ(config.class_mode, ClassMode::ClassAware);
    EXPECT_EQ(config.max_detections, 300);
    EXPECT_EQ(config.max_candidates, 30000);
    EXPECT_NEAR(config.class_offset, 7680.0f, K_EXACT);
}

/** @brief Tests that the lower-scoring of two overlapping boxes is suppressed. */
TEST(Medium_DetectPostprocess, SuppressesTheLowerOfTwoOverlappingBoxes)
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
TEST(Medium_DetectPostprocess, ListsSurvivorsByDescendingScore)
{
    const std::vector<float> boxes{0.0f,  0.0f,  1.0f,  1.0f,  10.0f, 10.0f, 11.0f, 11.0f,
                                   20.0f, 20.0f, 21.0f, 21.0f, 30.0f, 30.0f, 31.0f, 31.0f};
    const std::vector<float> scores{0.1f, 0.9f, 0.5f, 0.5f};
    std::vector<std::int64_t> kept;
    SushiAI::Vision::Detect::suppress_overlaps(boxes, scores, 0.5f, kept);

    EXPECT_TRUE(equal_reference(kept, std::vector<std::int64_t>{1, 2, 3, 0}));
}

/** @brief Tests that an IoU equal to the threshold survives and one above it does not. */
TEST(Medium_DetectPostprocess, KeepsABoxWhoseIoUEqualsTheThreshold)
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
    EXPECT_TRUE(near_reference(detections, expected, K_EXACT));
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
    EXPECT_TRUE(near_reference(aware, expected_aware, K_EXACT));
    EXPECT_TRUE(near_reference(agnostic, expected_agnostic, K_EXACT));
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
    EXPECT_TRUE(near_reference(detections, expected, K_EXACT));
}

/** @brief Tests that an anchor contributes every passing class, or only its best one. */
TEST(Medium_DetectPostprocess, AnAnchorContributesEveryPassingClassOrItsBest)
{
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.6f, 0.1f, 0.7f};
    const std::vector<float> every = suppress_one_image(
        boxes, scores, 3,
        make_nms_config(0.25f, 0.5f, LabelMode::MultiLabel, ClassMode::ClassAware));
    const std::vector<float> best = suppress_one_image(
        boxes, scores, 3,
        make_nms_config(0.25f, 0.5f, LabelMode::BestClass, ClassMode::ClassAware));

    const std::vector<float> expected_every{0.0f, 0.0f, 10.0f, 10.0f, 0.7f, 2.0f,
                                            0.0f, 0.0f, 10.0f, 10.0f, 0.6f, 0.0f};
    const std::vector<float> expected_best{0.0f, 0.0f, 10.0f, 10.0f, 0.7f, 2.0f};
    EXPECT_TRUE(near_reference(every, expected_every, K_EXACT));
    EXPECT_TRUE(near_reference(best, expected_best, K_EXACT));
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
    EXPECT_TRUE(near_reference(detections, expected, K_EXACT));
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
    EXPECT_TRUE(near_reference(detections, expected, K_EXACT));
}

/** @brief Tests that a threshold outside [0, 1] is refused with its value. */
TEST(Medium_DetectPostprocess, RejectsAThresholdOutsideTheUnitInterval)
{
    const std::vector<float> boxes{5.0f, 5.0f, 10.0f, 10.0f};
    const std::vector<float> scores{0.6f};
    SA_EXPECT_ERROR(
        (void)suppress_one_image(
            boxes, scores, 1,
            make_nms_config(1.5f, 0.5f, LabelMode::BestClass, ClassMode::ClassAware)),
        "[0, 1]", "1.5");
}

/** @brief Tests that an image whose scores all stay at or below the threshold yields no rows. */
TEST(Medium_DetectPostprocess, AnImageBelowTheThresholdYieldsNoRows)
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
    SushiAI::Vision::Detect::select_nms_detections(
        boxes, scores, PredictionShape{2, 2, 1},
        make_nms_config(0.25f, 0.5f, LabelMode::BestClass, ClassMode::ClassAware), detections);

    // Image 0 holds one box twice, so 0.8 falls to 0.9; image 1 holds two boxes that are apart.
    const std::vector<float> expected_first{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0.0f};
    const std::vector<float> expected_second{40.0f, 40.0f, 50.0f, 50.0f, 0.6f, 0.0f,
                                             20.0f, 20.0f, 30.0f, 30.0f, 0.3f, 0.0f};
    ASSERT_EQ(detections.size(), 2u);
    EXPECT_TRUE(near_reference(detections[0], expected_first, K_EXACT));
    EXPECT_TRUE(near_reference(detections[1], expected_second, K_EXACT));
}

/** @brief Tests that scores that do not fill the shape are refused, not read as other classes. */
TEST(Medium_DetectPostprocess, RejectsScoresThatDoNotFillTheShape)
{
    const std::vector<float> boxes(8, 0.0f);
    const std::vector<float> scores(6, 0.5f);
    std::vector<std::vector<float>> detections;
    SA_EXPECT_ERROR(
        SushiAI::Vision::Detect::select_nms_detections(
            boxes, scores, PredictionShape{1, 2, 2}, NMSConfig{}, detections),
        "[1, 2, 2] scores", "8 and 6 values");
}
