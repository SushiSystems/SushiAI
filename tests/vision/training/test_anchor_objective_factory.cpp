/**************************************************************************/
/* test_anchor_objective_factory.cpp                                      */
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
 * @file test_anchor_objective_factory.cpp
 * @brief Tests that the anchor objective is built from the task, the section and the shapes.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/training/anchor_objective.hpp>
#include <SushiAI/vision/training/anchor_objective_factory.hpp>
#include <SushiAI/vision/training/best_shape_head_delta_rule.hpp>
#include <SushiAI/vision/training/shape_ratio_head_delta_rule.hpp>

using namespace SushiAI;
using Train::ObjectiveContext;
using Train::OutputShape;
using Vision::Detect::Anchor::HeadCoding;
using Vision::Training::AnchorHead;
using Vision::Training::AnchorObjective;
using Vision::Training::build_anchor_objective;
using Vision::Training::make_anchor_objective;

namespace
{
    /** @brief Holds the images per step. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the classes of the task; a slot then has 5 + 2 entries. */
    constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the channels of a head of two anchors: 2 * (5 + K_CLASSES). */
    constexpr std::int64_t K_HEAD_CHANNELS = 14;

    /** @brief Bounds the error of a float32 a section states as a short binary fraction. */
    constexpr double K_OPTION_TOLERANCE = 0.0;

    /** @brief Holds a task of two heads of two anchors each over K_CLASSES classes. */
    constexpr std::string_view K_TASK = R"({
        "type": "anchor_detection",
        "classes": 2,
        "anchors": [[0.25, 0.25], [0.5, 0.25], [0.5, 0.75], [0.75, 0.5]],
        "heads": [
            {"output": "head.0", "anchors": [0, 1], "coding": "logit", "scale_xy": 1.0},
            {"output": "head.1", "anchors": [2, 3], "coding": "activated", "scale_xy": 1.5}
        ],
        "suppression": {"criterion": "iou"}
    })";

    /** @brief Parses @p text as a section of a model or a training file. */
    Config::Value section_from(std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    }

    /** @brief Returns the context of an [8, 8, 3] model that publishes both heads and a stem. */
    ObjectiveContext make_context()
    {
        ObjectiveContext given;
        given.batch = K_BATCH;
        given.classes = K_CLASSES;
        given.input_sample = Shape{8, 6, 3};
        given.task = section_from(K_TASK);
        given.outputs = {OutputShape{"head.0", Shape{K_BATCH, 4, 3, K_HEAD_CHANNELS}},
                         OutputShape{"stem", Shape{K_BATCH, 4, 3, 8}},
                         OutputShape{"head.1", Shape{K_BATCH, 2, 1, K_HEAD_CHANNELS}}};
        return given;
    }
} // namespace

/** @brief Tests that a task without an assignment member trains with Darknet's rule. */
TEST(Small_AnchorObjectiveFactory, TheDefaultAssignmentPicksTheDarknetRule)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            build_anchor_objective(make_context(), section_from("{}")));

    EXPECT_NE(dynamic_cast<const Vision::Training::BestShapeHeadDeltaRule*>(&objective.rule()),
              nullptr);
}

/** @brief Tests that a shape_ratio assignment over heads of logits picks the shape-ratio rule. */
TEST(Small_AnchorObjectiveFactory, TheShapeRatioAssignmentPicksTheShapeRatioRule)
{
    ObjectiveContext given = make_context();
    given.task = section_from(R"({
        "type": "anchor_detection",
        "classes": 2,
        "anchors": [[0.25, 0.25], [0.5, 0.25], [0.5, 0.75], [0.75, 0.5]],
        "heads": [
            {"output": "head.0", "anchors": [0, 1], "coding": "logit", "scale_xy": 2.0,
             "size": "squared_sigmoid"},
            {"output": "head.1", "anchors": [2, 3], "coding": "logit", "scale_xy": 2.0,
             "size": "squared_sigmoid"}
        ],
        "suppression": {"criterion": "iou"},
        "assignment": {"rule": "shape_ratio"}
    })");

    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            build_anchor_objective(given, section_from("{}")));

    EXPECT_NE(
        dynamic_cast<const Vision::Training::ShapeRatioHeadDeltaRule*>(&objective.rule()),
        nullptr);
}

/** @brief Tests that the shape-ratio rule refuses a head that is not of logits. */
TEST(Small_AnchorObjectiveFactory, TheShapeRatioAssignmentRefusesAnActivatedHead)
{
    ObjectiveContext given = make_context();
    given.task = section_from(R"({
        "type": "anchor_detection",
        "classes": 2,
        "anchors": [[0.25, 0.25], [0.5, 0.25], [0.5, 0.75], [0.75, 0.5]],
        "heads": [
            {"output": "head.0", "anchors": [0, 1], "coding": "logit", "scale_xy": 1.0,
             "size": "squared_sigmoid"},
            {"output": "head.1", "anchors": [2, 3], "coding": "activated", "scale_xy": 1.5}
        ],
        "suppression": {"criterion": "iou"},
        "assignment": {"rule": "shape_ratio"}
    })");

    SA_EXPECT_FAILURE(build_anchor_objective(given, section_from("{}")),
                      StatusCode::InvalidDescription, "head.1");
}

/** @brief Tests that each head takes its extents from the output the task names. */
TEST(Small_AnchorObjectiveFactory, EachHeadTakesItsExtentsFromItsPublishedOutput)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            build_anchor_objective(make_context(), section_from("{}")));

    EXPECT_EQ(objective.network().height, 8);
    EXPECT_EQ(objective.network().width, 6);
    ASSERT_EQ(objective.layout().anchors.size(), 4u);
    ASSERT_EQ(objective.heads().size(), 2u);

    const AnchorHead& fine = objective.heads()[0];
    EXPECT_EQ(fine.output, "head.0");
    EXPECT_EQ(fine.shape.batch, K_BATCH);
    EXPECT_EQ(fine.shape.height, 4);
    EXPECT_EQ(fine.shape.width, 3);
    EXPECT_EQ(fine.options.num_classes, K_CLASSES);
    EXPECT_EQ(fine.options.coding, HeadCoding::Logit);

    const AnchorHead& coarse = objective.heads()[1];
    EXPECT_EQ(coarse.output, "head.1");
    EXPECT_EQ(coarse.shape.height, 2);
    EXPECT_EQ(coarse.shape.width, 1);
    EXPECT_EQ(coarse.options.coding, HeadCoding::Activated);
    EXPECT_NEAR(coarse.options.scale_xy, 1.5, K_OPTION_TOLERANCE);
}

/** @brief Tests that the section's options reach every head, defaults for keys left out. */
TEST(Small_AnchorObjectiveFactory, TheSectionSetsTheTrainingOptionsOfEveryHead)
{
    const Vision::Detect::Anchor::HeadTrainingOptions defaults;

    SA_ASSERT_OK_AND_ASSIGN(
        const AnchorObjective objective,
        build_anchor_objective(make_context(),
                               section_from(R"({"type": "anchor", "box_weight": 0.5})")));

    for (const AnchorHead& head : objective.heads())
    {
        EXPECT_NEAR(head.training.box_weight, 0.5, K_OPTION_TOLERANCE);
        EXPECT_NEAR(head.training.object_weight, defaults.object_weight, K_OPTION_TOLERANCE);
        EXPECT_EQ(head.training.box_loss, defaults.box_loss);
    }
}

/** @brief Tests that data which states no class count is accepted and another count is not. */
TEST(Small_AnchorObjectiveFactory, TheClassCountOfTheDataMustBeTheTasks)
{
    ObjectiveContext given = make_context();
    given.classes = 0;
    SA_EXPECT_OK(build_anchor_objective(given, section_from("{}")));

    given.classes = 80;
    SA_EXPECT_FAILURE(build_anchor_objective(given, section_from("{}")),
                      StatusCode::Incompatible, "80 classes", "task states 2");
}

/** @brief Tests that a model file without a task section is refused. */
TEST(Small_AnchorObjectiveFactory, AModelWithoutATaskIsRefused)
{
    ObjectiveContext given = make_context();
    given.task = Config::Value();

    SA_EXPECT_FAILURE(build_anchor_objective(given, section_from("{}")),
                      StatusCode::InvalidDescription, "'task'", "'anchor'");
}

/** @brief Tests that a head the model does not publish is refused with what it publishes. */
TEST(Small_AnchorObjectiveFactory, AHeadTheModelDoesNotPublishIsRefused)
{
    ObjectiveContext given = make_context();
    given.outputs.pop_back();

    SA_EXPECT_FAILURE(build_anchor_objective(given, section_from("{}")),
                      StatusCode::InvalidDescription, "head 1", "'head.1'", "'head.0'",
                      "'stem'");
}

/** @brief Tests that an output of other channels, another batch or another rank is refused. */
TEST(Small_AnchorObjectiveFactory, AnOutputOfOtherExtentsIsRefused)
{
    ObjectiveContext channels = make_context();
    channels.outputs[0].shape = Shape{K_BATCH, 4, 3, K_HEAD_CHANNELS + 1};
    SA_EXPECT_FAILURE(build_anchor_objective(channels, section_from("{}")),
                      StatusCode::Incompatible, "'head.0'", "[2, 4, 3, 15]", "[2, H, W, 14]");

    ObjectiveContext batch = make_context();
    batch.outputs[2].shape = Shape{K_BATCH + 1, 2, 1, K_HEAD_CHANNELS};
    SA_EXPECT_FAILURE(build_anchor_objective(batch, section_from("{}")),
                      StatusCode::Incompatible, "'head.1'", "[2, H, W, 14]");

    ObjectiveContext rank = make_context();
    rank.outputs[0].shape = Shape{K_BATCH, 12 * K_HEAD_CHANNELS};
    SA_EXPECT_FAILURE(build_anchor_objective(rank, section_from("{}")),
                      StatusCode::Incompatible, "'head.0'", "[2, H, W, 14]");
}

/** @brief Tests that an input sample that is not [H, W, C] is refused. */
TEST(Small_AnchorObjectiveFactory, AnInputOfAnotherRankIsRefused)
{
    ObjectiveContext given = make_context();
    given.input_sample = Shape{192};

    SA_EXPECT_FAILURE(build_anchor_objective(given, section_from("{}")),
                      StatusCode::InvalidDescription, "[192]", "[H, W, C]");
}

/** @brief Tests that a key the section does not have is refused by the section's reader. */
TEST(Small_AnchorObjectiveFactory, AKeyTheSectionDoesNotHaveIsRefused)
{
    SA_EXPECT_FAILURE(build_anchor_objective(make_context(),
                                             section_from(R"({"label_smoothing": 0.1})")),
                      StatusCode::InvalidDescription, "objective", "'label_smoothing'");
}

/** @brief Tests that the registry form returns the objective and prefixes a refusal. */
TEST(Small_AnchorObjectiveFactory, TheRegistryFormPrefixesARefusalWithItsContext)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        make_anchor_objective(make_context(), section_from("{}"), "the objective block"));
    ASSERT_NE(objective, nullptr);
    EXPECT_EQ(objective->name(), "anchor_delta");

    ObjectiveContext given = make_context();
    given.task = Config::Value();
    SA_EXPECT_FAILURE(make_anchor_objective(given, section_from("{}"), "the objective block"),
                      StatusCode::InvalidDescription, "the objective block", "'task'");
}
