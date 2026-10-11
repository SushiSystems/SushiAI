/**************************************************************************/
/* test_darknet_training.cpp                                              */
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
 * @file test_darknet_training.cpp
 * @brief Tests the solver and head settings a .cfg yields, the defaults and the refusals.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_training.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using Vision::Detect::Anchor::BoxLoss;
using Vision::Detect::BoxGradient;
using Vision::Detect::Anchor::HeadTrainingOptions;
using Vision::Detect::Anchor::ObjectnessTarget;
using Vision::Importers::DarknetConfig;
using Vision::Importers::DarknetTraining;
using Vision::Importers::parse_darknet_config;
using Vision::Importers::read_darknet_training;

namespace
{
    /** @brief Holds the tolerance of every float field, absolute. */
    constexpr float K_FLOAT_TOLERANCE = 1e-6f;

    /** @brief Holds the tolerance of every double field, absolute. */
    constexpr double K_DOUBLE_TOLERANCE = 1e-12;

    /** @brief Holds the head values a test expects; the thresholds are the match and ignore. */
    struct ExpectedHead
    {
        /** @brief Holds the expected box term. */
        BoxLoss box_loss;

        /** @brief Holds the expected objectness target. */
        ObjectnessTarget objectness;

        /** @brief Holds the expected box weight. */
        float box_weight;

        /** @brief Holds the expected objectness weight. */
        float object_weight;

        /** @brief Holds the expected bound of a box delta; 0 sets none. */
        float max_delta;

        /** @brief Holds the expected match threshold. */
        float match_threshold;

        /** @brief Holds the expected ignore threshold. */
        float ignore_threshold;
    };

    /**
     * @brief Expects @p head to hold every value of @p expected.
     * @param head The head under test.
     * @param expected The values the test states from the cfg.
     */
    void expect_head(const HeadTrainingOptions& head, const ExpectedHead& expected)
    {
        EXPECT_EQ(head.box_loss, expected.box_loss);
        EXPECT_EQ(head.objectness_target, expected.objectness);
        EXPECT_NEAR(head.box_weight, expected.box_weight, K_FLOAT_TOLERANCE);
        EXPECT_NEAR(head.object_weight, expected.object_weight, K_FLOAT_TOLERANCE);
        EXPECT_NEAR(head.max_delta, expected.max_delta, K_FLOAT_TOLERANCE);
        EXPECT_NEAR(head.thresholds.match_threshold, expected.match_threshold, K_FLOAT_TOLERANCE);
        EXPECT_NEAR(head.thresholds.ignore_threshold, expected.ignore_threshold,
                    K_FLOAT_TOLERANCE);
    }

    /**
     * @brief Parses @p text as a .cfg and reads its training settings.
     * @param text A whole .cfg text, its first section [net].
     * @return The settings, or the failure of either step.
     */
    Result<DarknetTraining> read_cfg_text(const std::string& text)
    {
        Result<DarknetConfig> config = parse_darknet_config(text, "test.cfg");
        if (!config.has_value())
            return config.status();

        return read_darknet_training(config.value());
    }
} // namespace

/**
 * @brief Tests the v3-tiny and v3 key set: every training key absent but ignore_thresh and
 *        truth_thresh.
 *
 * The expected values are the SPEC 3 table column: iou_normalizer 0.75 and obj_normalizer 1
 * when absent, iou_thresh 1 when absent, ignore_thresh 0.7, mse.
 */
TEST(Unit_DarknetTraining, ReadsTheV3HeadSettingsWhenTheTrainingKeysAreAbsent)
{
    const std::string text =
        "[net]\n"
        "batch=64\n"
        "[convolutional]\n"
        "filters=255\n"
        "[yolo]\n"
        "mask=3,4,5\n"
        "classes=80\n"
        "ignore_thresh=0.7\n"
        "truth_thresh=1\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetTraining training, read_cfg_text(text));

    ASSERT_EQ(training.heads.size(), 1u);
    expect_head(training.heads[0], ExpectedHead{BoxLoss::SquaredError, ObjectnessTarget::One,
                                                0.75f, 1.0f, 0.0f, 1.0f, 0.7f});
}

/**
 * @brief Tests the v4-tiny key set: CIoU boxes and iou_normalizer 0.07.
 *
 * The objectness and iou_thresh keys are absent, so obj_normalizer is 1 and iou_thresh is 1.
 */
TEST(Unit_DarknetTraining, ReadsTheV4TinyHeadSettings)
{
    const std::string text =
        "[net]\n"
        "batch=64\n"
        "[yolo]\n"
        "mask=3,4,5\n"
        "classes=80\n"
        "iou_loss=ciou\n"
        "iou_normalizer=0.07\n"
        "cls_normalizer=1.0\n"
        "ignore_thresh=0.7\n"
        "truth_thresh=1\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetTraining training, read_cfg_text(text));

    ASSERT_EQ(training.heads.size(), 1u);
    expect_head(training.heads[0], ExpectedHead{BoxLoss::CompleteIoU, ObjectnessTarget::One,
                                                0.07f, 1.0f, 0.0f, 1.0f, 0.7f});
}

/**
 * @brief Tests the v4 key set: CIoU boxes, iou_thresh 0.213 and max_delta 5.
 *
 * The objectness_smooth key is absent, so the objectness target is One.
 */
TEST(Unit_DarknetTraining, ReadsTheV4HeadSettings)
{
    const std::string text =
        "[net]\n"
        "batch=64\n"
        "[yolo]\n"
        "mask=0,1,2\n"
        "classes=80\n"
        "iou_loss=ciou\n"
        "iou_normalizer=0.07\n"
        "cls_normalizer=1.0\n"
        "iou_thresh=0.213\n"
        "ignore_thresh=0.7\n"
        "max_delta=5\n"
        "truth_thresh=1\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetTraining training, read_cfg_text(text));

    ASSERT_EQ(training.heads.size(), 1u);
    expect_head(training.heads[0], ExpectedHead{BoxLoss::CompleteIoU, ObjectnessTarget::One,
                                                0.07f, 1.0f, 5.0f, 0.213f, 0.7f});
}

/**
 * @brief Tests the v7-tiny and v7 key set: objectness_smooth 1 and max_delta 2.
 *
 * cls_normalizer 0.5 is read and reaches no option, as REPORT section 17 states.
 */
TEST(Unit_DarknetTraining, ReadsTheV7HeadSettings)
{
    const std::string text =
        "[net]\n"
        "batch=64\n"
        "[yolo]\n"
        "mask=6,7,8\n"
        "classes=80\n"
        "iou_loss=ciou\n"
        "iou_normalizer=0.05\n"
        "cls_normalizer=0.5\n"
        "obj_normalizer=1.0\n"
        "iou_thresh=0.2\n"
        "ignore_thresh=0.7\n"
        "objectness_smooth=1\n"
        "max_delta=2\n"
        "truth_thresh=1\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetTraining training, read_cfg_text(text));

    ASSERT_EQ(training.heads.size(), 1u);
    expect_head(training.heads[0], ExpectedHead{BoxLoss::CompleteIoU, ObjectnessTarget::BestIoU,
                                                0.05f, 1.0f, 2.0f, 0.2f, 0.7f});
}

/**
 * @brief Tests that an absent [net] solver key keeps the default the header states.
 */
TEST(Unit_DarknetTraining, KeepsTheSolverDefaultsWhenTheKeysAreAbsent)
{
    const std::string text =
        "[net]\n"
        "width=416\n"
        "[yolo]\n"
        "classes=80\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetTraining training, read_cfg_text(text));

    EXPECT_NEAR(training.solver.learning_rate, 0.001, K_DOUBLE_TOLERANCE);
    EXPECT_NEAR(training.solver.momentum, 0.9, K_DOUBLE_TOLERANCE);
    EXPECT_NEAR(training.solver.decay, 0.0005, K_DOUBLE_TOLERANCE);
    EXPECT_EQ(training.solver.burn_in, 0);
    EXPECT_TRUE(training.solver.steps.empty());
    EXPECT_TRUE(training.solver.scales.empty());
    EXPECT_EQ(training.solver.batch, 1);
    EXPECT_EQ(training.solver.subdivisions, 1);
    EXPECT_EQ(training.solver.max_batches, 0);
}

/**
 * @brief Tests that every solver key the file spells is read to the value it states.
 */
TEST(Unit_DarknetTraining, ReadsEverySolverKeyTheFileSpells)
{
    const std::string text =
        "[net]\n"
        "batch=64\n"
        "subdivisions=16\n"
        "momentum=0.949\n"
        "decay=0.0005\n"
        "learning_rate=0.00261\n"
        "burn_in=1000\n"
        "max_batches=500200\n"
        "policy=steps\n"
        "steps=400000,450000\n"
        "scales=0.1,0.1\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetTraining training, read_cfg_text(text));

    EXPECT_EQ(training.solver.batch, 64);
    EXPECT_EQ(training.solver.subdivisions, 16);
    EXPECT_NEAR(training.solver.momentum, 0.949, K_DOUBLE_TOLERANCE);
    EXPECT_NEAR(training.solver.decay, 0.0005, K_DOUBLE_TOLERANCE);
    EXPECT_NEAR(training.solver.learning_rate, 0.00261, K_DOUBLE_TOLERANCE);
    EXPECT_EQ(training.solver.burn_in, 1000);
    EXPECT_EQ(training.solver.max_batches, 500200);
    EXPECT_EQ(training.solver.steps, (std::vector<std::int64_t>{400000, 450000}));
    ASSERT_EQ(training.solver.scales.size(), 2u);
    EXPECT_NEAR(training.solver.scales[0], 0.1, K_DOUBLE_TOLERANCE);
    EXPECT_NEAR(training.solver.scales[1], 0.1, K_DOUBLE_TOLERANCE);
}

/**
 * @brief Tests that heads follow the file's order and that other sections are not heads.
 */
TEST(Unit_DarknetTraining, KeepsTheHeadsInFileOrderAndSkipsOtherSections)
{
    const std::string text =
        "[net]\n"
        "[convolutional]\n"
        "filters=1\n"
        "[yolo]\n"
        "iou_normalizer=0.07\n"
        "[route]\n"
        "layers=-1\n"
        "[yolo]\n"
        "iou_normalizer=0.05\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetTraining training, read_cfg_text(text));

    ASSERT_EQ(training.heads.size(), 2u);
    EXPECT_NEAR(training.heads[0].box_weight, 0.07f, K_FLOAT_TOLERANCE);
    EXPECT_NEAR(training.heads[1].box_weight, 0.05f, K_FLOAT_TOLERANCE);
}

/**
 * @brief Tests that the spelled defaults, mse and objectness_smooth 0, give the same options as
 *        absent keys.
 */
TEST(Unit_DarknetTraining, ReadsMSEAndObjectnessSmoothZeroAsTheDefaults)
{
    const std::string text =
        "[net]\n"
        "[yolo]\n"
        "iou_loss=mse\n"
        "objectness_smooth=0\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetTraining training, read_cfg_text(text));

    ASSERT_EQ(training.heads.size(), 1u);
    expect_head(training.heads[0], ExpectedHead{BoxLoss::SquaredError, ObjectnessTarget::One,
                                                0.75f, 1.0f, 0.0f, 1.0f, 0.7f});
}

/**
 * @brief Tests that a head takes the Darknet-compatible box gradient under either box loss.
 */
TEST(Unit_DarknetTraining, SelectsTheDarknetCompatibleBoxGradientForEveryHead)
{
    const std::string text =
        "[net]\n"
        "[yolo]\n"
        "iou_loss=ciou\n"
        "[yolo]\n"
        "iou_loss=mse\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetTraining training, read_cfg_text(text));

    ASSERT_EQ(training.heads.size(), 2u);
    EXPECT_EQ(training.heads[0].box_gradient, BoxGradient::DarknetCompatible);
    EXPECT_EQ(training.heads[1].box_gradient, BoxGradient::DarknetCompatible);
    EXPECT_EQ(HeadTrainingOptions{}.box_gradient, BoxGradient::Analytic);
}

/**
 * @brief Tests that truth_thresh 1 and label_smooth_eps 0 are the values the reader accepts.
 */
TEST(Unit_DarknetTraining, AcceptsTruthThreshOneAndLabelSmoothZero)
{
    const std::string text =
        "[net]\n"
        "[yolo]\n"
        "truth_thresh=1.0\n"
        "label_smooth_eps=0\n";

    SA_ASSERT_OK_AND_ASSIGN(const DarknetTraining training, read_cfg_text(text));

    EXPECT_EQ(training.heads.size(), 1u);
}

/** @brief Tests that a policy other than steps is refused, naming its line. */
TEST(Unit_DarknetTraining, RefusesAPolicyOtherThanSteps)
{
    const std::string text =
        "[net]\n"
        "policy=exp\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::Unsupported, "line 2", "policy");
}

/** @brief Tests that a truth_thresh other than 1 is refused, naming its line. */
TEST(Unit_DarknetTraining, RefusesATruthThreshOtherThanOne)
{
    const std::string text =
        "[net]\n"
        "[yolo]\n"
        "truth_thresh=0.9\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::Unsupported, "line 3", "truth_thresh");
}

/** @brief Tests that a non-zero label_smooth_eps is refused, naming its line. */
TEST(Unit_DarknetTraining, RefusesANonZeroLabelSmooth)
{
    const std::string text =
        "[net]\n"
        "[yolo]\n"
        "label_smooth_eps=0.1\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::Unsupported, "line 3",
                      "label_smooth_eps");
}

/** @brief Tests that an iou_loss other than mse and ciou is refused, naming its line. */
TEST(Unit_DarknetTraining, RefusesAnIoULossOtherThanMSEAndCIoU)
{
    const std::string text =
        "[net]\n"
        "[yolo]\n"
        "iou_loss=giou\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::Unsupported, "line 3", "iou_loss");
}

/** @brief Tests that an objectness_smooth other than 0 or 1 is invalid, naming its line. */
TEST(Unit_DarknetTraining, RefusesAClsNormalizerThatIsNotANumber)
{
    const std::string text =
        "[net]\n"
        "[yolo]\n"
        "cls_normalizer=half\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::InvalidDescription, "line 3",
                      "cls_normalizer");
}

TEST(Unit_DarknetTraining, RefusesAnObjectnessSmoothOtherThanZeroOrOne)
{
    const std::string text =
        "[net]\n"
        "[yolo]\n"
        "objectness_smooth=2\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::InvalidDescription, "line 3",
                      "objectness_smooth");
}

/** @brief Tests that a head whose iou_thresh is 0 is invalid, naming its section's line. */
TEST(Unit_DarknetTraining, RefusesAHeadWhoseMatchThresholdIsZero)
{
    const std::string text =
        "[net]\n"
        "[yolo]\n"
        "iou_thresh=0\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::InvalidDescription, "line 2",
                      "match threshold");
}

/** @brief Tests that an ignore_thresh above 1 is invalid, naming its section's line. */
TEST(Unit_DarknetTraining, RefusesAHeadWhoseIgnoreThresholdIsAboveOne)
{
    const std::string text =
        "[net]\n"
        "[yolo]\n"
        "ignore_thresh=1.5\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::InvalidDescription, "line 2",
                      "ignore threshold");
}

/** @brief Tests that a learning_rate that is not a number is invalid, naming its line. */
TEST(Unit_DarknetTraining, RefusesALearningRateThatIsNotANumber)
{
    const std::string text =
        "[net]\n"
        "learning_rate=abc\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::InvalidDescription, "line 2",
                      "learning_rate", "abc");
}

/** @brief Tests that a non-finite learning_rate is invalid, naming its line. */
TEST(Unit_DarknetTraining, RefusesANonFiniteLearningRate)
{
    const std::string text =
        "[net]\n"
        "learning_rate=nan\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::InvalidDescription, "line 2", "nan");
}

/** @brief Tests that a batch below 1 is invalid, since the loss divides by it. */
TEST(Unit_DarknetTraining, RefusesABatchBelowOne)
{
    const std::string text =
        "[net]\n"
        "batch=0\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::InvalidDescription, "batch");
}

/** @brief Tests that an empty entry in a list is invalid, naming its line. */
TEST(Unit_DarknetTraining, RefusesAnEmptyEntryInAList)
{
    const std::string text =
        "[net]\n"
        "steps=1,,2\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::InvalidDescription, "line 2", "steps");
}

/** @brief Tests that steps and scales of different lengths are invalid, naming both counts. */
TEST(Unit_DarknetTraining, RefusesStepsAndScalesOfDifferentLengths)
{
    const std::string text =
        "[net]\n"
        "steps=100,200\n"
        "scales=0.1\n";

    SA_EXPECT_FAILURE(read_cfg_text(text), StatusCode::InvalidDescription, "line 1",
                      "'steps' holds 2", "'scales' holds 1");
}
