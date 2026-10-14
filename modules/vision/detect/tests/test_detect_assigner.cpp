/**************************************************************************/
/* test_detect_assigner.cpp                                               */
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
 * @file test_detect_assigner.cpp
 * @brief Tests label assignment and the loss leaves against hand cases and the reference.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/free/anchors.hpp>
#include <SushiAI/vision/detect/free/assigner.hpp>
#include <SushiAI/vision/detect/free/assigner_config.hpp>
#include <SushiAI/vision/detect/free/assignment.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/detect/free/target_leaves.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>

namespace
{
    using SushiAI::Vision::Detect::Free::AssignerConfig;
    using SushiAI::Vision::Detect::Free::AssignerInputs;
    using SushiAI::Vision::Detect::Free::Assignment;
    using SushiAI::Vision::Detect::Free::FeatureLevel;
    using SushiAI::Vision::Detect::GroundTruthBatch;
    using SushiAI::Vision::Detect::Free::TargetLeaves;
    using SushiAI::Vision::Detect::Free::TaskAlignedAssigner;
    using SushiAI::Test::equal_reference;
    using SushiAI::Test::near_reference;

    /** @brief Holds the tolerance of copied values and of multiples of 0.5. */
    constexpr float K_EXACT = 0.0f;

    /** @brief Holds the tolerance of a hand-computed target score, eight float32 steps at 1. */
    constexpr float K_SCORE_TOLERANCE = 1e-6f;

    /** @brief Holds the tolerance of a reference target score; pow and atan may round apart. */
    constexpr float K_REFERENCE_SCORE_TOLERANCE = 1e-5f;

    /** @brief Holds the tolerance of a reference target box in pixels: two float32 steps at 640. */
    constexpr float K_REFERENCE_BOX_TOLERANCE = 1.3e-4f;

    /** @brief Holds the side of the hand-built grid in cells. */
    constexpr std::size_t K_GRID_SIDE = 4;

    /** @brief Holds the anchors of the hand-built grid. */
    constexpr std::size_t K_GRID_ANCHORS = K_GRID_SIDE * K_GRID_SIDE;

    /** @brief Holds the stride of the hand-built grid in pixels. */
    constexpr float K_GRID_STRIDE = 8.0f;

    /** @brief Holds the classes of the hand-built cases. */
    constexpr std::size_t K_CLASSES = 2;

    /** @brief Holds a predicted box no hand-built ground truth touches. */
    const std::vector<float> K_FAR_BOX{100.0f, 100.0f, 101.0f, 101.0f};

    /** @brief Holds one image on a 4 by 4 grid of stride 8: anchor a sits at column a % 4. */
    struct Scene
    {
        std::vector<float> points;           /**< `[16, 2]` pixel centres 4, 12, 20, 28. */
        std::vector<float> scores;           /**< `[16, 2]`. */
        std::vector<float> boxes;            /**< `[16, 4]`. */
        std::vector<std::int64_t> gt_labels; /**< `[M]`. */
        std::vector<float> gt_boxes;         /**< `[M, 4]`. */
        std::vector<std::uint8_t> gt_valid;  /**< `[M]`. */

        /** @brief Builds a scene whose anchors all score @p score and predict the far box. */
        explicit Scene(float score)
            : points(K_GRID_ANCHORS * 2), scores(K_GRID_ANCHORS * K_CLASSES, score)
        {
            for (std::size_t anchor = 0; anchor < K_GRID_ANCHORS; ++anchor)
            {
                points[anchor * 2] =
                    (static_cast<float>(anchor % K_GRID_SIDE) + 0.5f) * K_GRID_STRIDE;
                points[anchor * 2 + 1] =
                    (static_cast<float>(anchor / K_GRID_SIDE) + 0.5f) * K_GRID_STRIDE;
                boxes.insert(boxes.end(), K_FAR_BOX.begin(), K_FAR_BOX.end());
            }
        }

        /** @brief Appends a valid ground-truth box. */
        void add_truth(std::int64_t label, const std::vector<float>& box)
        {
            gt_labels.push_back(label);
            gt_boxes.insert(gt_boxes.end(), box.begin(), box.end());
            gt_valid.push_back(1);
        }

        /** @brief Makes @p anchor predict @p box. */
        void predict(std::size_t anchor, const std::vector<float>& box)
        {
            for (std::size_t column = 0; column < 4; ++column)
                boxes[anchor * 4 + column] = box[column];
        }

        /** @brief Returns the scene as the inputs of one assigner call. */
        [[nodiscard]] AssignerInputs inputs() const
        {
            AssignerInputs inputs;
            inputs.batch = 1;
            inputs.anchors = static_cast<std::int64_t>(K_GRID_ANCHORS);
            inputs.max_boxes = static_cast<std::int64_t>(gt_labels.size());
            inputs.scores = scores;
            inputs.boxes = boxes;
            inputs.anchor_points = points;
            inputs.gt_labels = gt_labels;
            inputs.gt_boxes = gt_boxes;
            inputs.gt_valid = gt_valid;
            return inputs;
        }
    };

    /** @brief Returns the settings of the hand-built cases: metric = score times overlap. */
    AssignerConfig make_hand_config(std::int64_t topk, std::int64_t topk2)
    {
        AssignerConfig config;
        config.topk = topk;
        config.topk2 = topk2;
        config.num_classes = static_cast<std::int64_t>(K_CLASSES);
        config.alpha = 1.0f;
        config.beta = 1.0f;
        config.stride_val = K_GRID_STRIDE;
        return config;
    }

    /** @brief Returns the indices of the anchors @p assignment marks as foreground. */
    std::vector<std::int64_t> foreground_anchors(const Assignment& assignment)
    {
        std::vector<std::int64_t> anchors;
        for (std::size_t row = 0; row < assignment.foreground_mask.size(); ++row)
        {
            if (assignment.foreground_mask[row] != 0)
                anchors.push_back(static_cast<std::int64_t>(row));
        }
        return anchors;
    }

    /** @brief Returns the scene with one box over the whole image and three raised scores. */
    Scene make_crowded_scene()
    {
        const std::vector<float> whole{0.0f, 0.0f, 32.0f, 32.0f};
        Scene scene(0.1f);
        scene.add_truth(1, whole);
        for (std::size_t anchor = 0; anchor < K_GRID_ANCHORS; ++anchor)
            scene.predict(anchor, whole);
        scene.scores[5 * K_CLASSES + 1] = 0.9f;
        scene.scores[10 * K_CLASSES + 1] = 0.6f;
        scene.scores[2 * K_CLASSES + 1] = 0.3f;
        return scene;
    }

    /** @brief Returns the scene with two boxes that hold anchor 5, which predicts @p shared. */
    Scene make_contested_scene(const std::vector<float>& first, const std::vector<float>& second,
                               const std::vector<float>& shared)
    {
        Scene scene(0.5f);
        scene.add_truth(0, first);
        scene.add_truth(1, second);
        for (const std::size_t anchor : {0u, 1u, 4u})
            scene.predict(anchor, first);
        for (const std::size_t anchor : {6u, 9u, 10u})
            scene.predict(anchor, second);
        scene.predict(5, shared);
        return scene;
    }

    /** @brief Returns the four values of @p row of a `[rows, 4]` array. */
    std::vector<float> box_row(const std::vector<float>& values, std::size_t row)
    {
        return std::vector<float>(values.begin() + row * 4, values.begin() + row * 4 + 4);
    }

    /** @brief Returns an assignment of @p batch images of @p anchors anchors with no foreground. */
    Assignment make_idle_assignment(std::int64_t batch, std::int64_t anchors)
    {
        const std::size_t rows = static_cast<std::size_t>(batch * anchors);
        Assignment assignment;
        assignment.batch = batch;
        assignment.anchors = anchors;
        assignment.num_classes = static_cast<std::int64_t>(K_CLASSES);
        assignment.foreground_mask.assign(rows, 0);
        assignment.target_boxes.assign(rows * 4, 0.0f);
        assignment.target_scores.assign(rows * K_CLASSES, 0.0f);
        assignment.target_labels.assign(rows, 0);
        assignment.target_indices.assign(rows, 0);
        return assignment;
    }

    /** @brief Marks @p row of @p assignment as foreground with a pixel box and two class scores. */
    void set_foreground(Assignment& assignment, std::size_t row, const std::vector<float>& box,
                        const std::vector<float>& scores)
    {
        assignment.foreground_mask[row] = 1;
        for (std::size_t column = 0; column < 4; ++column)
            assignment.target_boxes[row * 4 + column] = box[column];
        for (std::size_t column = 0; column < K_CLASSES; ++column)
            assignment.target_scores[row * K_CLASSES + column] = scores[column];
    }

    /** @brief Holds the reference levels MANIFEST.json records for a 640 px input. */
    const std::vector<FeatureLevel> K_REFERENCE_LEVELS{{80, 80, 8.0f}, {40, 40, 16.0f},
                                                       {20, 20, 32.0f}};

    /** @brief Holds the side of the reference input in pixels. */
    constexpr float K_REFERENCE_IMAGE_SIDE = 640.0f;

    /** @brief Holds the row of the reference ground truth that is smaller than a stride. */
    constexpr std::int64_t K_SMALL_BOX_ROW = 3;

    /** @brief Pads the reference's "batch." rows; returns why not when one is missing. */
    std::string pad_reference_truth(SushiAI::Test::ReferenceFile& loss, GroundTruthBatch& padded)
    {
        const SushiAI::Test::ReferenceTensor image = loss.read("batch.batch_idx");
        const SushiAI::Test::ReferenceTensor classes = loss.read("batch.cls");
        const SushiAI::Test::ReferenceTensor boxes = loss.read("batch.bboxes");
        for (const SushiAI::Test::ReferenceTensor* tensor : {&image, &classes, &boxes})
        {
            if (!tensor->found)
                return tensor->reason;
        }

        const std::vector<float> image_values = image.floats();
        const std::vector<float> class_values = classes.floats();
        const std::vector<float> box_values = boxes.floats();
        const std::vector<std::int64_t> image_indices(image_values.begin(), image_values.end());
        const std::vector<std::int64_t> class_indices(class_values.begin(), class_values.end());
        SushiAI::Vision::Detect::pad_ground_truth(
            {image_indices, class_indices, box_values}, 1, K_REFERENCE_IMAGE_SIDE,
            K_REFERENCE_IMAGE_SIDE, padded);
        return {};
    }

    /** @brief Returns how many foreground anchors of @p mask hold box @p row. */
    template <typename Mask>
    std::int64_t count_holders(const std::vector<Mask>& mask,
                               const std::vector<std::int64_t>& indices, std::int64_t row)
    {
        std::int64_t count = 0;
        for (std::size_t anchor = 0; anchor < mask.size(); ++anchor)
        {
            if (mask[anchor] != 0 && indices[anchor] == row)
                ++count;
        }
        return count;
    }

    /**
     * @brief Runs one branch's assigner on the reference inputs and compares every output.
     * @param expected_foreground "foreground_anchor_count" of the branch in MANIFEST.json.
     * @param expected_small_box Foreground rows of the branch's reference that hold box 3.
     */
    void expect_reference_assignment(const std::string& branch, const AssignerConfig& config,
                                     std::int64_t expected_foreground,
                                     std::int64_t expected_small_box)
    {
        const std::string prefix = branch + ".assignment.";
        SA_OPEN_REFERENCE_OR_SKIP(loss, "loss.sacp");
        SA_READ_REFERENCE_OR_FAIL(scores, loss, prefix + "input.scores");
        SA_READ_REFERENCE_OR_FAIL(boxes, loss, prefix + "input.boxes");
        SA_READ_REFERENCE_OR_FAIL(points, loss, "anchors.points");
        SA_READ_REFERENCE_OR_FAIL(mask, loss, prefix + "foreground_mask");
        SA_READ_REFERENCE_OR_FAIL(target_boxes, loss, prefix + "target_boxes");
        SA_READ_REFERENCE_OR_FAIL(target_scores, loss, prefix + "target_scores");
        SA_READ_REFERENCE_OR_FAIL(target_indices, loss, prefix + "target_indices");
        SA_READ_REFERENCE_OR_FAIL(target_labels, loss, prefix + "target_labels");
        GroundTruthBatch truth;
        const std::string missing = pad_reference_truth(loss, truth);
        ASSERT_TRUE(missing.empty()) << missing;

        const std::vector<float> score_values = scores.floats();
        const std::vector<float> box_values = boxes.floats();
        const std::vector<float> point_values = points.floats();
        AssignerInputs inputs;
        inputs.batch = 1;
        inputs.anchors = SushiAI::Vision::Detect::Free::count_anchors(K_REFERENCE_LEVELS);
        inputs.max_boxes = truth.max_boxes;
        inputs.scores = score_values;
        inputs.boxes = box_values;
        inputs.anchor_points = point_values;
        inputs.gt_labels = truth.labels;
        inputs.gt_boxes = truth.boxes;
        inputs.gt_valid = truth.valid;

        TaskAlignedAssigner assigner(config);
        Assignment assignment;
        assigner.assign(inputs, assignment);

        const std::vector<std::int32_t> reference_mask = mask.int32s();
        const std::vector<std::int64_t> reference_indices = target_indices.int64s();

        // Equality of the mask and the indices relies on the fixture having no tied metrics.
        EXPECT_TRUE(equal_reference(assignment.foreground_mask, reference_mask));
        EXPECT_TRUE(equal_reference(assignment.target_indices, reference_indices));
        EXPECT_TRUE(equal_reference(assignment.target_labels, target_labels.int64s()));
        EXPECT_TRUE(near_reference(assignment.target_boxes, target_boxes.floats(),
                                   K_REFERENCE_BOX_TOLERANCE));
        EXPECT_TRUE(near_reference(assignment.target_scores, target_scores.floats(),
                                   K_REFERENCE_SCORE_TOLERANCE));

        const std::int64_t foreground =
            std::accumulate(assignment.foreground_mask.begin(), assignment.foreground_mask.end(),
                            std::int64_t{0});
        const std::int64_t small_box =
            count_holders(assignment.foreground_mask, assignment.target_indices, K_SMALL_BOX_ROW);
        EXPECT_EQ(foreground, expected_foreground);
        EXPECT_EQ(small_box, expected_small_box);
        EXPECT_EQ(small_box, count_holders(reference_mask, reference_indices, K_SMALL_BOX_ROW));
    }
}

/** @brief Tests that the one-to-many preset holds the settings the reference run recorded. */
TEST(Unit_DetectAssigner, OneToManyPresetMatchesTheReferenceSettings)
{
    const std::vector<float> strides{8.0f, 16.0f, 32.0f};
    const AssignerConfig config =
        SushiAI::Vision::Detect::Free::one_to_many_assigner_config(80, strides);

    // The values are "assigner.one2many" of loss.sacp in tests/fixtures/yolo26/MANIFEST.json.
    EXPECT_EQ(config.topk, 10);
    EXPECT_EQ(config.topk2, 10);
    EXPECT_EQ(config.num_classes, 80);
    EXPECT_NEAR(config.alpha, 0.5f, K_EXACT);
    EXPECT_NEAR(config.beta, 6.0f, K_EXACT);
    EXPECT_NEAR(config.stride_val, 16.0f, K_EXACT);
    EXPECT_NEAR(config.eps, 1e-9f, K_EXACT);
}

/** @brief Tests that the one-to-one preset keeps seven candidates and then one. */
TEST(Unit_DetectAssigner, OneToOnePresetMatchesTheReferenceSettings)
{
    const std::vector<float> strides{8.0f, 16.0f, 32.0f};
    const AssignerConfig config =
        SushiAI::Vision::Detect::Free::one_to_one_assigner_config(80, strides);

    // The values are "assigner.one2one" of loss.sacp in tests/fixtures/yolo26/MANIFEST.json.
    EXPECT_EQ(config.topk, 7);
    EXPECT_EQ(config.topk2, 1);
    EXPECT_NEAR(config.alpha, 0.5f, K_EXACT);
    EXPECT_NEAR(config.beta, 6.0f, K_EXACT);
    EXPECT_NEAR(config.stride_val, 16.0f, K_EXACT);
}

/** @brief Tests that stride_val is the second stride, or the only one of a one-level model. */
TEST(Unit_DetectAssigner, PresetTakesStrideValFromTheSecondLevel)
{
    const std::vector<float> two{4.0f, 12.0f};
    const std::vector<float> one{4.0f};
    const std::vector<float> none;
    EXPECT_NEAR(SushiAI::Vision::Detect::Free::one_to_many_assigner_config(3, two).stride_val,
                12.0f,
                K_EXACT);
    EXPECT_NEAR(SushiAI::Vision::Detect::Free::one_to_many_assigner_config(3, one).stride_val,
                4.0f,
                K_EXACT);
    EXPECT_NEAR(SushiAI::Vision::Detect::Free::one_to_many_assigner_config(3, none).stride_val,
                16.0f,
                K_EXACT);
}

/** @brief Tests that the member defaults are the defaults of the reference's constructor. */
TEST(Unit_DetectAssigner, ConfigDefaultsMatchTheReferenceConstructor)
{
    const AssignerConfig config;

    // The values are the arguments of TaskAlignedAssigner.__init__, ultralytics/utils/tal.py:33-39.
    EXPECT_EQ(config.topk, 13);
    EXPECT_EQ(config.topk2, 13);
    EXPECT_EQ(config.num_classes, 80);
    EXPECT_NEAR(config.alpha, 1.0f, K_EXACT);
    EXPECT_NEAR(config.beta, 6.0f, K_EXACT);
    EXPECT_NEAR(config.stride_val, 16.0f, K_EXACT);
    EXPECT_NEAR(config.eps, 1e-9f, K_EXACT);
}

/** @brief Tests that a config with no candidates, a negative stride_val or eps is refused. */
TEST(Unit_DetectAssigner, RejectsAConfigWithoutCandidatesOrWithANegativeMargin)
{
    const AssignerConfig no_candidates = make_hand_config(0, 3);
    AssignerConfig negative_stride = make_hand_config(3, 3);
    negative_stride.stride_val = -1.0f;
    AssignerConfig negative_eps = make_hand_config(3, 3);
    negative_eps.eps = -1.0f;

    SA_EXPECT_ERROR((void)TaskAlignedAssigner(no_candidates), "positive topk", "got 0, 3 and 2");
    SA_EXPECT_ERROR((void)TaskAlignedAssigner(negative_stride), "stride_val", "at least 0", "-1");
    SA_EXPECT_ERROR((void)TaskAlignedAssigner(negative_eps), "eps of at least 0", "-1");
}

/** @brief Tests that the metric is the score to the power alpha times the overlap to beta. */
TEST(Unit_DetectAssigner, TheMetricRaisesTheScoreToAlphaAndTheOverlapToBeta)
{
    const std::vector<float> whole{0.0f, 0.0f, 32.0f, 32.0f};
    const std::vector<float> strides{K_GRID_STRIDE};
    Scene scene(0.0f);
    scene.add_truth(1, whole);
    scene.predict(5, whole);
    scene.predict(10, {0.0f, 0.0f, 24.0f, 24.0f});
    scene.scores[5 * K_CLASSES + 1] = 0.25f;
    scene.scores[10 * K_CLASSES + 1] = 0.81f;

    AssignerConfig one =
        SushiAI::Vision::Detect::Free::one_to_many_assigner_config(K_CLASSES, strides);
    one.topk = 1;
    one.topk2 = 1;
    AssignerConfig two = one;
    two.topk = 2;
    two.topk2 = 2;
    Assignment best;
    Assignment both;
    TaskAlignedAssigner(one).assign(scene.inputs(), best);
    TaskAlignedAssigner(two).assign(scene.inputs(), both);

    // Anchor 10 covers 576 of 1024 px and pays a centre term of 32 / 2048: an overlap of 35 / 64.
    const float overlap = 576.0f / 1024.0f - 32.0f / 2048.0f;
    const float cube = overlap * overlap * overlap;

    // Anchor 5 has sqrt(0.25) * 1 = 0.5 and anchor 10 has sqrt(0.81) * (35 / 64)^6 = 0.024.
    EXPECT_NEAR(one.alpha, 0.5f, K_EXACT);
    EXPECT_NEAR(one.beta, 6.0f, K_EXACT);
    EXPECT_TRUE(equal_reference(foreground_anchors(best), std::vector<std::int64_t>{5}));
    EXPECT_TRUE(equal_reference(foreground_anchors(both), std::vector<std::int64_t>{5, 10}));

    // Each score is the metric over the largest metric, 0.5, times the largest overlap, 1.
    EXPECT_NEAR(both.target_scores[5 * K_CLASSES + 1], 0.5f / 0.5f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(both.target_scores[10 * K_CLASSES + 1], 0.9f * cube * cube / 0.5f,
                K_SCORE_TOLERANCE);
}

/** @brief Tests that target scores are scaled by the largest overlap of their box. */
TEST(Unit_DetectAssigner, NormalisesByTheLargestOverlapOfTheBox)
{
    const std::vector<float> truth{0.0f, 0.0f, 16.0f, 16.0f};
    const std::vector<float> offset{8.0f, 8.0f, 24.0f, 24.0f};
    Scene scene(0.5f);
    scene.add_truth(0, truth);
    scene.predict(0, offset);
    scene.predict(1, offset);
    scene.scores[0 * K_CLASSES] = 0.8f;
    scene.scores[1 * K_CLASSES] = 0.4f;
    TaskAlignedAssigner assigner(make_hand_config(2, 2));
    Assignment assignment;
    assigner.assign(scene.inputs(), assignment);

    // The boxes share 64 of 448 px and the centre term is 128 / 1152: an overlap of 2 / 63.
    const float overlap = 64.0f / 448.0f - 128.0f / 1152.0f;
    EXPECT_TRUE(equal_reference(foreground_anchors(assignment), std::vector<std::int64_t>{0, 1}));

    // The metrics are 0.8 and 0.4 times the overlap, and no anchor of the box overlaps it more.
    EXPECT_NEAR(assignment.target_scores[0 * K_CLASSES], overlap, K_SCORE_TOLERANCE);
    EXPECT_NEAR(assignment.target_scores[1 * K_CLASSES], 0.4f / 0.8f * overlap,
                K_SCORE_TOLERANCE);
}

/** @brief Tests that a prediction apart from its box has a zero overlap, metric and score. */
TEST(Unit_DetectAssigner, APredictionApartFromItsBoxHasNoOverlap)
{
    const std::vector<float> truth{0.0f, 0.0f, 16.0f, 16.0f};
    Scene scene(0.9f);
    scene.add_truth(0, truth);
    scene.predict(4, truth);
    scene.predict(5, truth);
    scene.scores[4 * K_CLASSES] = 0.2f;
    scene.scores[5 * K_CLASSES] = 0.1f;
    Assignment two;
    Assignment three;
    TaskAlignedAssigner(make_hand_config(2, 2)).assign(scene.inputs(), two);
    TaskAlignedAssigner(make_hand_config(3, 3)).assign(scene.inputs(), three);

    // Anchors 0 and 1 score 0.9 but predict the far box, whose complete IoU with the box is -0.84.
    EXPECT_TRUE(equal_reference(foreground_anchors(two), std::vector<std::int64_t>{4, 5}));

    // A third place goes to the first candidate of metric zero, anchor 0, and scores nothing.
    EXPECT_TRUE(equal_reference(foreground_anchors(three), std::vector<std::int64_t>{0, 4, 5}));
    EXPECT_NEAR(three.target_scores[0 * K_CLASSES], 0.0f, K_EXACT);
    EXPECT_NEAR(three.target_scores[4 * K_CLASSES], 0.2f / 0.2f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(three.target_scores[5 * K_CLASSES], 0.1f / 0.2f, K_SCORE_TOLERANCE);
}

/** @brief Tests that an anchor two boxes overlap equally goes to the first of them. */
TEST(Unit_DetectAssigner, AContestedAnchorOfEqualOverlapsGoesToTheFirstBox)
{
    const std::vector<float> truth{0.0f, 0.0f, 16.0f, 16.0f};
    Scene scene(0.5f);
    scene.add_truth(0, truth);
    scene.add_truth(1, truth);
    scene.predict(5, truth);
    TaskAlignedAssigner assigner(make_hand_config(1, 1));
    Assignment assignment;
    assigner.assign(scene.inputs(), assignment);

    // Both boxes keep anchor 5 with metric 0.5 * 1, and its overlap with each of them is 1.
    EXPECT_TRUE(equal_reference(foreground_anchors(assignment), std::vector<std::int64_t>{5}));
    EXPECT_EQ(assignment.target_indices[5], 0);
    EXPECT_EQ(assignment.target_labels[5], 0);
    EXPECT_NEAR(assignment.target_scores[5 * K_CLASSES], 1.0f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(assignment.target_scores[5 * K_CLASSES + 1], 0.0f, K_EXACT);
}

/** @brief Tests that a contested anchor goes to the box of largest overlap that did not keep it. */
TEST(Unit_DetectAssigner, AContestedAnchorCanGoToABoxThatDidNotKeepIt)
{
    const std::vector<float> third{8.0f, 0.0f, 24.0f, 16.0f};
    Scene scene(0.0f);
    scene.add_truth(0, {0.0f, 0.0f, 16.0f, 16.0f});
    scene.add_truth(0, {8.0f, 8.0f, 24.0f, 24.0f});
    scene.add_truth(1, third);
    scene.predict(5, third);
    scene.predict(2, third);
    scene.scores[5 * K_CLASSES] = 0.9f;
    scene.scores[5 * K_CLASSES + 1] = 0.1f;
    scene.scores[2 * K_CLASSES + 1] = 0.9f;
    TaskAlignedAssigner assigner(make_hand_config(1, 1));
    Assignment assignment;
    assigner.assign(scene.inputs(), assignment);

    // Boxes 0 and 1 keep anchor 5 at 0.9 * (1 / 3 - 64 / 832); box 2 keeps anchor 2 at 0.9 * 1.
    EXPECT_TRUE(equal_reference(foreground_anchors(assignment), std::vector<std::int64_t>{2, 5}));

    // Anchor 5 predicts box 2 exactly, so box 2 has its largest overlap and takes it.
    EXPECT_EQ(assignment.target_indices[5], 2);
    EXPECT_EQ(assignment.target_labels[5], 1);
    EXPECT_NEAR(assignment.target_scores[5 * K_CLASSES], 0.0f, K_EXACT);

    // Box 2 holds anchor 5 at metric 0.1 * 1 beside its best, 0.9, and its largest overlap is 1.
    EXPECT_NEAR(assignment.target_scores[5 * K_CLASSES + 1], 0.1f / 0.9f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(assignment.target_scores[2 * K_CLASSES + 1], 0.9f / 0.9f, K_SCORE_TOLERANCE);
}

/** @brief Tests that a batch without ground truth gets no foreground and the idle label. */
TEST(Unit_DetectAssigner, AnImageWithoutGroundTruthGetsNoForeground)
{
    const Scene scene(0.5f);
    TaskAlignedAssigner assigner(make_hand_config(3, 3));
    Assignment assignment;
    assigner.assign(scene.inputs(), assignment);

    EXPECT_TRUE(foreground_anchors(assignment).empty());
    EXPECT_TRUE(equal_reference(assignment.target_labels,
                                std::vector<std::int64_t>(K_GRID_ANCHORS, 2)));
    EXPECT_TRUE(equal_reference(assignment.target_indices,
                                std::vector<std::int64_t>(K_GRID_ANCHORS, 0)));
    EXPECT_TRUE(near_reference(assignment.target_boxes,
                               std::vector<float>(K_GRID_ANCHORS * 4, 0.0f), K_EXACT));
    EXPECT_TRUE(near_reference(assignment.target_scores,
                               std::vector<float>(K_GRID_ANCHORS * K_CLASSES, 0.0f), K_EXACT));
}

/** @brief Tests that a padding row assigns nothing, even over anchors it would contain. */
TEST(Unit_DetectAssigner, APaddingRowAssignsNothing)
{
    Scene scene = make_crowded_scene();
    scene.gt_valid[0] = 0;
    TaskAlignedAssigner assigner(make_hand_config(3, 3));
    Assignment assignment;
    assigner.assign(scene.inputs(), assignment);

    EXPECT_TRUE(foreground_anchors(assignment).empty());
    EXPECT_TRUE(near_reference(assignment.target_scores,
                               std::vector<float>(K_GRID_ANCHORS * K_CLASSES, 0.0f), K_EXACT));
}

/** @brief Tests that a box holding sixteen candidates keeps the topk with the largest metric. */
TEST(Unit_DetectAssigner, KeepsTheTopKCandidatesWhenABoxHoldsMore)
{
    const Scene scene = make_crowded_scene();
    TaskAlignedAssigner assigner(make_hand_config(3, 3));
    Assignment assignment;
    assigner.assign(scene.inputs(), assignment);

    // Every prediction equals the box, so the overlap is 1 and the metric is the class score.
    EXPECT_TRUE(equal_reference(foreground_anchors(assignment),
                                std::vector<std::int64_t>{2, 5, 10}));
    EXPECT_NEAR(assignment.target_scores[5 * K_CLASSES + 1], 0.9f / 0.9f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(assignment.target_scores[10 * K_CLASSES + 1], 0.6f / 0.9f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(assignment.target_scores[2 * K_CLASSES + 1], 0.3f / 0.9f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(assignment.target_scores[5 * K_CLASSES], 0.0f, K_EXACT);
    EXPECT_TRUE(equal_reference(assignment.target_labels,
                                std::vector<std::int64_t>(K_GRID_ANCHORS, 1)));
}

/** @brief Tests that a second top-k of one leaves a box its single best anchor. */
TEST(Unit_DetectAssigner, TheSecondTopKKeepsTheBestAnchorOfABox)
{
    const Scene scene = make_crowded_scene();
    TaskAlignedAssigner assigner(make_hand_config(3, 1));
    Assignment assignment;
    assigner.assign(scene.inputs(), assignment);

    EXPECT_TRUE(equal_reference(foreground_anchors(assignment), std::vector<std::int64_t>{5}));
    EXPECT_NEAR(assignment.target_scores[5 * K_CLASSES + 1], 1.0f, K_SCORE_TOLERANCE);
}

/** @brief Tests that an anchor two boxes select goes to the box its prediction overlaps more. */
TEST(Unit_DetectAssigner, AContestedAnchorGoesToTheHigherOverlap)
{
    const std::vector<float> first{0.0f, 0.0f, 16.0f, 16.0f};
    const std::vector<float> second{8.0f, 8.0f, 24.0f, 24.0f};
    const Scene scene = make_contested_scene(first, second, second);
    TaskAlignedAssigner assigner(make_hand_config(4, 4));
    Assignment assignment;
    assigner.assign(scene.inputs(), assignment);

    // Anchor 5 predicts the second box: overlap 1 with it, 64 / 448 - 128 / 1152 with the first.
    EXPECT_TRUE(equal_reference(foreground_anchors(assignment),
                                std::vector<std::int64_t>{0, 1, 4, 5, 6, 9, 10}));
    EXPECT_TRUE(equal_reference(
        assignment.target_indices,
        std::vector<std::int64_t>{0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0}));
    EXPECT_TRUE(equal_reference(
        assignment.target_labels,
        std::vector<std::int64_t>{0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0}));
    EXPECT_NEAR(assignment.target_scores[5 * K_CLASSES + 1], 1.0f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(assignment.target_scores[5 * K_CLASSES], 0.0f, K_EXACT);
    EXPECT_NEAR(assignment.target_boxes[5 * 4], 8.0f, K_EXACT);
    EXPECT_NEAR(assignment.target_boxes[4 * 4 + 2], 16.0f, K_EXACT);
}

/** @brief Tests that the contested anchor goes to the first box when it overlaps that one more. */
TEST(Unit_DetectAssigner, AContestedAnchorCanGoToTheFirstBox)
{
    const std::vector<float> first{0.0f, 0.0f, 16.0f, 16.0f};
    const std::vector<float> second{8.0f, 8.0f, 24.0f, 24.0f};
    const Scene scene = make_contested_scene(first, second, first);
    TaskAlignedAssigner assigner(make_hand_config(4, 4));
    Assignment assignment;
    assigner.assign(scene.inputs(), assignment);

    EXPECT_TRUE(equal_reference(
        assignment.target_indices,
        std::vector<std::int64_t>{0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0}));
    EXPECT_NEAR(assignment.target_scores[5 * K_CLASSES], 1.0f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(assignment.target_scores[5 * K_CLASSES + 1], 0.0f, K_EXACT);
}

/** @brief Tests that a box smaller than stride_val is enlarged before anchors are tested. */
TEST(Unit_DetectAssigner, ASmallBoxIsEnlargedToStrideVal)
{
    const std::vector<float> small{13.0f, 13.0f, 15.0f, 15.0f};
    Scene scene(0.5f);
    scene.add_truth(0, small);
    for (const std::size_t anchor : {5u, 6u, 9u, 10u})
        scene.predict(anchor, small);

    AssignerConfig enlarging = make_hand_config(4, 4);
    enlarging.stride_val = 16.0f;
    AssignerConfig plain = make_hand_config(4, 4);
    plain.stride_val = 1.0f;
    Assignment enlarged;
    Assignment unchanged;
    TaskAlignedAssigner(enlarging).assign(scene.inputs(), enlarged);
    TaskAlignedAssigner(plain).assign(scene.inputs(), unchanged);

    // Centre 14 with side 16 spans 6 to 22, which holds the centres 12 and 20 on both axes.
    EXPECT_TRUE(equal_reference(foreground_anchors(enlarged),
                                std::vector<std::int64_t>{5, 6, 9, 10}));
    EXPECT_TRUE(foreground_anchors(unchanged).empty());
    EXPECT_NEAR(enlarged.target_boxes[5 * 4], 13.0f, K_EXACT);
}

/** @brief Tests that each image of a batch is assigned from its own predictions and boxes. */
TEST(Unit_DetectAssigner, AssignsEachImageFromItsOwnRows)
{
    const Scene crowded = make_crowded_scene();
    Scene empty(0.5f);
    empty.add_truth(0, {0.0f, 0.0f, 0.0f, 0.0f});
    empty.gt_valid[0] = 0;

    std::vector<float> scores = empty.scores;
    std::vector<float> boxes = empty.boxes;
    std::vector<std::int64_t> labels = empty.gt_labels;
    std::vector<float> truths = empty.gt_boxes;
    std::vector<std::uint8_t> valid = empty.gt_valid;
    scores.insert(scores.end(), crowded.scores.begin(), crowded.scores.end());
    boxes.insert(boxes.end(), crowded.boxes.begin(), crowded.boxes.end());
    labels.insert(labels.end(), crowded.gt_labels.begin(), crowded.gt_labels.end());
    truths.insert(truths.end(), crowded.gt_boxes.begin(), crowded.gt_boxes.end());
    valid.insert(valid.end(), crowded.gt_valid.begin(), crowded.gt_valid.end());

    AssignerInputs inputs = crowded.inputs();
    inputs.batch = 2;
    inputs.scores = scores;
    inputs.boxes = boxes;
    inputs.gt_labels = labels;
    inputs.gt_boxes = truths;
    inputs.gt_valid = valid;
    TaskAlignedAssigner assigner(make_hand_config(3, 3));
    Assignment assignment;
    assigner.assign(inputs, assignment);

    // Image 1 starts at row 16, so its anchors 2, 5 and 10 are rows 18, 21 and 26.
    EXPECT_TRUE(equal_reference(foreground_anchors(assignment),
                                std::vector<std::int64_t>{18, 21, 26}));
    EXPECT_EQ(assignment.target_labels[0], 0);
    EXPECT_EQ(assignment.target_labels[16], 1);
}

/** @brief Tests that fewer anchors than topk, and a class outside the range, are refused. */
TEST(Unit_DetectAssigner, RejectsTooFewAnchorsAndAnUnknownClass)
{
    Scene scene = make_crowded_scene();
    Assignment assignment;
    TaskAlignedAssigner wide(make_hand_config(17, 17));
    SA_EXPECT_ERROR(wide.assign(scene.inputs(), assignment), "16 anchors", "topk 17");

    scene.gt_labels[0] = 2;
    TaskAlignedAssigner assigner(make_hand_config(3, 3));
    SA_EXPECT_ERROR(assigner.assign(scene.inputs(), assignment), "class 2 of 2");
}

/** @brief Tests that label rows are padded per image and converted to pixel corners. */
TEST(Unit_DetectGroundTruth, PadsLabelRowsPerImage)
{
    const std::vector<std::int64_t> images{0, 0, 2};
    const std::vector<std::int64_t> classes{3, 1, 7};
    const std::vector<float> boxes{0.5f, 0.5f, 0.5f, 0.25f, 0.25f, 0.75f,
                                   0.5f, 0.5f, 0.5f, 0.5f,  1.0f,  1.0f};
    GroundTruthBatch padded;
    SushiAI::Vision::Detect::pad_ground_truth({images, classes, boxes}, 3, 64.0f, 32.0f, padded);

    // Row 0: centre (32, 16), half size (16, 4). Row 1: centre (16, 24), half size (16, 8).
    const std::vector<float> expected_boxes{16.0f, 12.0f, 48.0f, 20.0f, 0.0f, 16.0f, 32.0f, 32.0f,
                                            0.0f,  0.0f,  0.0f,  0.0f,  0.0f, 0.0f,  0.0f,  0.0f,
                                            0.0f,  0.0f,  64.0f, 32.0f, 0.0f, 0.0f,  0.0f,  0.0f};
    EXPECT_EQ(padded.max_boxes, 2);
    EXPECT_TRUE(near_reference(padded.boxes, expected_boxes, K_EXACT));
    EXPECT_TRUE(equal_reference(padded.labels, std::vector<std::int64_t>{3, 1, 0, 0, 7, 0}));
    EXPECT_TRUE(equal_reference(padded.valid, std::vector<std::int64_t>{1, 1, 0, 0, 1, 0}));
}

/** @brief Tests that a row whose pixel corners do not sum above zero is padded in as invalid. */
TEST(Unit_DetectGroundTruth, MarksARowWithoutExtentInvalid)
{
    const std::vector<std::int64_t> images{0, 0};
    const std::vector<std::int64_t> classes{4, 5};
    const std::vector<float> boxes{0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f};
    GroundTruthBatch padded;
    SushiAI::Vision::Detect::pad_ground_truth({images, classes, boxes}, 1, 32.0f, 32.0f, padded);

    EXPECT_EQ(padded.max_boxes, 2);
    EXPECT_TRUE(equal_reference(padded.valid, std::vector<std::int64_t>{0, 1}));
    EXPECT_TRUE(equal_reference(padded.labels, std::vector<std::int64_t>{4, 5}));
}

/** @brief Tests that a batch without label rows is padded to no box rows at all. */
TEST(Unit_DetectGroundTruth, PadsNoRowsToABatchWithoutBoxes)
{
    const std::vector<std::int64_t> none;
    const std::vector<float> no_boxes;
    GroundTruthBatch padded;
    SushiAI::Vision::Detect::pad_ground_truth({none, none, no_boxes}, 2, 32.0f, 32.0f, padded);

    EXPECT_EQ(padded.batch, 2);
    EXPECT_EQ(padded.max_boxes, 0);
    EXPECT_TRUE(padded.labels.empty());
    EXPECT_TRUE(padded.boxes.empty());
    EXPECT_TRUE(padded.valid.empty());
}

/** @brief Tests that a label row naming an image outside the batch is refused. */
TEST(Unit_DetectGroundTruth, RejectsARowOutsideTheBatch)
{
    const std::vector<std::int64_t> images{3};
    const std::vector<std::int64_t> classes{0};
    const std::vector<float> boxes{0.5f, 0.5f, 0.5f, 0.5f};
    GroundTruthBatch padded;
    SA_EXPECT_ERROR(
        SushiAI::Vision::Detect::pad_ground_truth(
            {images, classes, boxes}, 3, 32.0f, 32.0f, padded),
        "row 0 names image 3", "batch of 3");
}

/** @brief Tests that the leaves carry the mask, the weights and the scaled targets of a row. */
TEST(Unit_DetectTargetLeaves, BuildsTheLossLeavesOfAnAssignment)
{
    const Scene scene = make_crowded_scene();
    TaskAlignedAssigner assigner(make_hand_config(3, 3));
    Assignment assignment;
    assigner.assign(scene.inputs(), assignment);

    const std::vector<FeatureLevel> levels{{4, 4, K_GRID_STRIDE}};
    std::vector<float> points(K_GRID_ANCHORS * 2);
    std::vector<float> strides(K_GRID_ANCHORS);
    SushiAI::Vision::Detect::Free::make_anchors(levels, points, strides);
    TargetLeaves leaves;
    SushiAI::Vision::Detect::Free::build_target_leaves(
        assignment, {points, strides, 32.0f, 64.0f}, leaves);

    // The three weights are 1, 2/3 and 1/3, so their sum is 2 and anchor 5 holds half of it.
    EXPECT_NEAR(leaves.target_scores_sum[0], 2.0f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(leaves.weight[5], 1.0f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(leaves.normalized_weight[5], 1.0f / 2.0f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(leaves.normalized_weight[10], (2.0f / 3.0f) / 2.0f, K_SCORE_TOLERANCE);

    const std::vector<float> row5_mask(leaves.box_mask.begin() + 20, leaves.box_mask.begin() + 24);
    const std::vector<float> row5_box(leaves.target_boxes.begin() + 20,
                                      leaves.target_boxes.begin() + 24);
    const std::vector<float> row5_distances(leaves.target_distances.begin() + 20,
                                            leaves.target_distances.begin() + 24);
    const std::vector<float> row5_scale(leaves.l1_scale.begin() + 20, leaves.l1_scale.begin() + 24);
    const float across = 8.0f / 64.0f * 0.25f * 0.5f;
    const float down = 8.0f / 32.0f * 0.25f * 0.5f;
    EXPECT_TRUE(near_reference(row5_mask, std::vector<float>(4, 1.0f), K_EXACT));
    EXPECT_TRUE(near_reference(row5_box, std::vector<float>{0.0f, 0.0f, 4.0f, 4.0f}, K_EXACT));
    const std::vector<float> expected_distances{1.5f, 1.5f, 2.5f, 2.5f};
    EXPECT_TRUE(near_reference(row5_distances, expected_distances, K_EXACT));
    const std::vector<float> expected_scale{across, down, across, down};
    EXPECT_TRUE(near_reference(row5_scale, expected_scale, K_SCORE_TOLERANCE));
}

/** @brief Tests that a row reads the point and the stride of its own anchor in a later image. */
TEST(Unit_DetectTargetLeaves, ReadsEachRowsAnchorAcrossImagesAndLevels)
{
    const std::vector<FeatureLevel> levels{{2, 2, 8.0f}, {1, 1, 16.0f}};
    std::vector<float> points(10);
    std::vector<float> strides(5);
    SushiAI::Vision::Detect::Free::make_anchors(levels, points, strides);
    Assignment assignment = make_idle_assignment(2, 5);
    set_foreground(assignment, 1, {2.0f, 0.0f, 18.0f, 20.0f}, {0.5f, 0.0f});
    set_foreground(assignment, 9, {0.0f, 4.0f, 16.0f, 12.0f}, {0.25f, 0.75f});
    TargetLeaves leaves;
    SushiAI::Vision::Detect::Free::build_target_leaves(
        assignment, {points, strides, 32.0f, 64.0f}, leaves);

    // The weights are 0.5 and 1, so their sum is 1.5 and the rows hold 1/3 and 2/3 of it.
    EXPECT_NEAR(leaves.target_scores_sum[0], 1.5f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(leaves.weight[1], 0.5f, K_EXACT);
    EXPECT_NEAR(leaves.weight[9], 1.0f, K_EXACT);
    EXPECT_NEAR(leaves.normalized_weight[1], 0.5f / 1.5f, K_SCORE_TOLERANCE);
    EXPECT_NEAR(leaves.normalized_weight[9], 1.0f / 1.5f, K_SCORE_TOLERANCE);

    // Row 1 is anchor 1 of image 0: point (1.5, 0.5) on stride 8, so the box is its pixels / 8.
    const float across1 = 8.0f / 64.0f * 0.25f * (0.5f / 1.5f);
    const float down1 = 8.0f / 32.0f * 0.25f * (0.5f / 1.5f);
    const std::vector<float> box1{0.25f, 0.0f, 2.25f, 2.5f};
    const std::vector<float> distances1{1.5f - 0.25f, 0.5f - 0.0f, 2.25f - 1.5f, 2.5f - 0.5f};
    const std::vector<float> scale1{across1, down1, across1, down1};
    EXPECT_TRUE(near_reference(box_row(leaves.target_boxes, 1), box1, K_EXACT));
    EXPECT_TRUE(near_reference(box_row(leaves.target_distances, 1), distances1, K_EXACT));
    EXPECT_TRUE(near_reference(box_row(leaves.l1_scale, 1), scale1, K_SCORE_TOLERANCE));

    // Row 9 is anchor 4 of image 1: point (0.5, 0.5) on stride 16, so the box is its pixels / 16.
    const float across9 = 16.0f / 64.0f * 0.25f * (1.0f / 1.5f);
    const float down9 = 16.0f / 32.0f * 0.25f * (1.0f / 1.5f);
    const std::vector<float> box9{0.0f, 0.25f, 1.0f, 0.75f};
    const std::vector<float> distances9{0.5f - 0.0f, 0.5f - 0.25f, 1.0f - 0.5f, 0.75f - 0.5f};
    const std::vector<float> scale9{across9, down9, across9, down9};
    EXPECT_TRUE(near_reference(box_row(leaves.target_boxes, 9), box9, K_EXACT));
    EXPECT_TRUE(near_reference(box_row(leaves.target_distances, 9), distances9, K_EXACT));
    EXPECT_TRUE(near_reference(box_row(leaves.l1_scale, 9), scale9, K_SCORE_TOLERANCE));
    EXPECT_TRUE(near_reference(box_row(leaves.box_mask, 9), std::vector<float>(4, 1.0f), K_EXACT));

    // Row 4 is the same anchor in image 0, which nothing was assigned to.
    EXPECT_TRUE(near_reference(box_row(leaves.box_mask, 4), std::vector<float>(4, 0.0f), K_EXACT));
}

/** @brief Tests that an unassigned row carries the safe box and zero in every weighting leaf. */
TEST(Unit_DetectTargetLeaves, AnUnassignedRowCarriesTheSafeBoxAndNoWeight)
{
    const Scene scene = make_crowded_scene();
    TaskAlignedAssigner assigner(make_hand_config(3, 3));
    Assignment assignment;
    assigner.assign(scene.inputs(), assignment);

    const std::vector<FeatureLevel> levels{{4, 4, K_GRID_STRIDE}};
    std::vector<float> points(K_GRID_ANCHORS * 2);
    std::vector<float> strides(K_GRID_ANCHORS);
    SushiAI::Vision::Detect::Free::make_anchors(levels, points, strides);
    TargetLeaves leaves;
    SushiAI::Vision::Detect::Free::build_target_leaves(
        assignment, {points, strides, 32.0f, 32.0f}, leaves);

    const std::vector<float> safe{0.0f, 0.0f, 1.0f, 1.0f};
    const std::vector<float> zeros(4, 0.0f);
    const std::vector<float> row0_box(leaves.target_boxes.begin(), leaves.target_boxes.begin() + 4);
    const std::vector<float> row0_mask(leaves.box_mask.begin(), leaves.box_mask.begin() + 4);
    const std::vector<float> row0_scale(leaves.l1_scale.begin(), leaves.l1_scale.begin() + 4);
    const std::vector<float> row0_distances(leaves.target_distances.begin(),
                                            leaves.target_distances.begin() + 4);
    const std::vector<float> row5_safe(leaves.safe_boxes.begin() + 20,
                                       leaves.safe_boxes.begin() + 24);
    EXPECT_TRUE(near_reference(row0_box, safe, K_EXACT));
    EXPECT_TRUE(near_reference(row0_mask, zeros, K_EXACT));
    EXPECT_TRUE(near_reference(row0_scale, zeros, K_EXACT));
    EXPECT_TRUE(near_reference(row0_distances, zeros, K_EXACT));
    EXPECT_TRUE(near_reference(row5_safe, safe, K_EXACT));
    EXPECT_NEAR(leaves.weight[0], 0.0f, K_EXACT);
    EXPECT_NEAR(leaves.normalized_weight[0], 0.0f, K_EXACT);
}

/** @brief Tests that the sum of target scores is clamped to 1 when nothing is assigned. */
TEST(Unit_DetectTargetLeaves, TheScoreSumIsAtLeastOne)
{
    const Scene scene(0.5f);
    TaskAlignedAssigner assigner(make_hand_config(3, 3));
    Assignment assignment;
    assigner.assign(scene.inputs(), assignment);

    const std::vector<FeatureLevel> levels{{4, 4, K_GRID_STRIDE}};
    std::vector<float> points(K_GRID_ANCHORS * 2);
    std::vector<float> strides(K_GRID_ANCHORS);
    SushiAI::Vision::Detect::Free::make_anchors(levels, points, strides);
    TargetLeaves leaves;
    SushiAI::Vision::Detect::Free::build_target_leaves(
        assignment, {points, strides, 32.0f, 32.0f}, leaves);

    EXPECT_NEAR(leaves.target_scores_sum[0], 1.0f, K_EXACT);
}

/** @brief Tests that padding the reference labels gives the boxes its assignment targets hold. */
TEST(Unit_DetectGroundTruth, PadsTheReferenceLabelsToItsTargetBoxes)
{
    SA_OPEN_REFERENCE_OR_SKIP(loss, "loss.sacp");
    SA_READ_REFERENCE_OR_FAIL(target_boxes, loss, "one2many.assignment.target_boxes");
    SA_READ_REFERENCE_OR_FAIL(target_indices, loss, "one2many.assignment.target_indices");
    SA_READ_REFERENCE_OR_FAIL(target_labels, loss, "one2many.assignment.target_labels");
    GroundTruthBatch truth;
    const std::string missing = pad_reference_truth(loss, truth);
    ASSERT_TRUE(missing.empty()) << missing;

    const std::vector<std::int64_t> indices = target_indices.int64s();
    std::vector<float> gathered_boxes;
    std::vector<std::int64_t> gathered_labels;
    for (const std::int64_t index : indices)
    {
        const std::size_t row = static_cast<std::size_t>(index);
        gathered_boxes.insert(gathered_boxes.end(), truth.boxes.begin() + row * 4,
                              truth.boxes.begin() + row * 4 + 4);
        gathered_labels.push_back(truth.labels[row]);
    }

    EXPECT_EQ(truth.max_boxes, 4);
    EXPECT_TRUE(equal_reference(truth.valid, std::vector<std::int64_t>(4, 1)));
    EXPECT_TRUE(near_reference(gathered_boxes, target_boxes.floats(), K_REFERENCE_BOX_TOLERANCE));
    EXPECT_TRUE(equal_reference(gathered_labels, target_labels.int64s()));
}

/** @brief Tests that the one-to-many assigner reproduces the reference's assignment outputs. */
TEST(Unit_DetectAssigner, ReproducesTheReferenceOneToManyAssignment)
{
    const std::vector<float> strides{8.0f, 16.0f, 32.0f};

    // 31 is "foreground_anchor_count.one2many" of loss.sacp in MANIFEST.json; 4 of them hold box 3.
    expect_reference_assignment(
        "one2many", SushiAI::Vision::Detect::Free::one_to_many_assigner_config(80, strides), 31, 4);
}

/** @brief Tests that the one-to-one assigner reproduces the reference's assignment outputs. */
TEST(Unit_DetectAssigner, ReproducesTheReferenceOneToOneAssignment)
{
    const std::vector<float> strides{8.0f, 16.0f, 32.0f};

    // 4 is "foreground_anchor_count.one2one" of loss.sacp in MANIFEST.json; 1 of them holds box 3.
    expect_reference_assignment(
        "one2one", SushiAI::Vision::Detect::Free::one_to_one_assigner_config(80, strides), 4, 1);
}
