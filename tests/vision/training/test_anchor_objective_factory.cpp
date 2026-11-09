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
 * @file test_anchor_objective_factory.cpp
 * @brief Tests that the anchor objective is built from the task, the section and the shapes.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <array>
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
    using SushiAI::Test::require_value;
    /** @brief Holds the images per step. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the classes of the task; a slot then has 5 + 2 entries. */
    constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the channels of a head of two anchors: 2 * (5 + K_CLASSES). */
    constexpr std::int64_t K_HEAD_CHANNELS = 14;

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
        return require_value(Config::parse_json(text, "<test>"));
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

    /** @brief Names an output the factory refuses and what the refusal names; "" matches any. */
    struct RefusedOutput
    {
        ObjectiveContext given;                 /**< What the factory is given. */
        std::array<const char*, 3> needles;     /**< What the message names. */
    };

    /** @brief Returns the context whose output @p index has the extents @p shape. */
    ObjectiveContext with_output(std::size_t index, Shape shape)
    {
        ObjectiveContext given = make_context();
        given.outputs[index].shape = shape;
        return given;
    }

    /** @brief Provides the outputs of other channels, another batch and another rank. */
    class Small_AnchorObjectiveFactoryOutputs : public ::testing::TestWithParam<RefusedOutput>
    {

    };
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

/** @brief Tests that the network is the input sample's height and width. */
TEST(Small_AnchorObjectiveFactory, TheNetworkSizeIsTheInputsHeightAndWidth)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            build_anchor_objective(make_context(), section_from("{}")));

    EXPECT_EQ(objective.network().height, 8);
    EXPECT_EQ(objective.network().width, 6);
    EXPECT_EQ(objective.layout().anchors.size(), 4u);
}

/** @brief Tests that each head takes its extents from the output the task names. */
TEST(Small_AnchorObjectiveFactory, EachHeadTakesItsExtentsFromItsPublishedOutput)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            build_anchor_objective(make_context(), section_from("{}")));
    ASSERT_EQ(objective.heads().size(), 2u);

    const AnchorHead& fine = objective.heads()[0];
    EXPECT_EQ(fine.output, "head.0");
    EXPECT_EQ(fine.shape.batch, K_BATCH);
    EXPECT_EQ(fine.shape.height, 4);
    EXPECT_EQ(fine.shape.width, 3);

    const AnchorHead& coarse = objective.heads()[1];
    EXPECT_EQ(coarse.output, "head.1");
    EXPECT_EQ(coarse.shape.height, 2);
    EXPECT_EQ(coarse.shape.width, 1);
}

/** @brief Tests that each head takes its class count, coding and stretch from the task. */
TEST(Small_AnchorObjectiveFactory, EachHeadTakesItsCodingAndScaleFromTheTask)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            build_anchor_objective(make_context(), section_from("{}")));
    ASSERT_EQ(objective.heads().size(), 2u);

    const AnchorHead& fine = objective.heads()[0];
    EXPECT_EQ(fine.options.num_classes, K_CLASSES);
    EXPECT_EQ(fine.options.coding, HeadCoding::Logit);

    const AnchorHead& coarse = objective.heads()[1];
    EXPECT_EQ(coarse.options.coding, HeadCoding::Activated);
    EXPECT_EQ(coarse.options.scale_xy, 1.5);
}

/** @brief Tests that the section's options reach every head, defaults for keys left out. */
TEST(Small_AnchorObjectiveFactory, TheSectionSetsTheTrainingOptionsOfEveryHead)
{
    // The objective of an empty section holds the defaults the keys left out must keep.
    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective baseline,
                            build_anchor_objective(make_context(), section_from("{}")));
    SA_ASSERT_OK_AND_ASSIGN(
        const AnchorObjective objective,
        build_anchor_objective(make_context(),
                               section_from(R"({"type": "anchor", "box_weight": 0.5})")));
    ASSERT_EQ(objective.heads().size(), 2u);

    EXPECT_EQ(objective.heads()[0].training.box_weight, 0.5);
    EXPECT_EQ(objective.heads()[1].training.box_weight, 0.5);
    EXPECT_EQ(objective.heads()[0].training.object_weight,
              baseline.heads()[0].training.object_weight);
    EXPECT_EQ(objective.heads()[1].training.object_weight,
              baseline.heads()[1].training.object_weight);
    EXPECT_EQ(objective.heads()[0].training.box_loss, baseline.heads()[0].training.box_loss);
    EXPECT_EQ(objective.heads()[1].training.box_loss, baseline.heads()[1].training.box_loss);
}

/** @brief Tests that data which states no class count is accepted. */
TEST(Small_AnchorObjectiveFactory, DataThatStatesNoClassCountIsAccepted)
{
    ObjectiveContext given = make_context();
    given.classes = 0;

    SA_EXPECT_OK(build_anchor_objective(given, section_from("{}")));
}

/** @brief Tests that data of another class count than the task's is refused. */
TEST(Small_AnchorObjectiveFactory, ADataClassCountOtherThanTheTasksIsRefused)
{
    ObjectiveContext given = make_context();
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
TEST_P(Small_AnchorObjectiveFactoryOutputs, AnOutputOfOtherExtentsIsRefused)
{
    const RefusedOutput& row = GetParam();

    SA_EXPECT_FAILURE(build_anchor_objective(row.given, section_from("{}")),
                      StatusCode::Incompatible, row.needles[0], row.needles[1], row.needles[2]);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_AnchorObjectiveFactoryOutputs,
    ::testing::Values(
        RefusedOutput{with_output(0, Shape{K_BATCH, 4, 3, K_HEAD_CHANNELS + 1}),
                      {"'head.0'", "[2, 4, 3, 15]", "[2, H, W, 14]"}},
        RefusedOutput{with_output(2, Shape{K_BATCH + 1, 2, 1, K_HEAD_CHANNELS}),
                      {"'head.1'", "[2, H, W, 14]", ""}},
        RefusedOutput{with_output(0, Shape{K_BATCH, 12 * K_HEAD_CHANNELS}),
                      {"'head.0'", "[2, H, W, 14]", ""}}));

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

/** @brief Tests that the registry form returns an objective named anchor_delta. */
TEST(Small_AnchorObjectiveFactory, TheRegistryFormReturnsTheObjective)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        make_anchor_objective(make_context(), section_from("{}"), "the objective block"));
    ASSERT_NE(objective, nullptr);

    EXPECT_EQ(objective->name(), "anchor_delta");
}

/** @brief Tests that the registry form prefixes a refusal with its context. */
TEST(Small_AnchorObjectiveFactory, TheRegistryFormPrefixesARefusalWithItsContext)
{
    ObjectiveContext given = make_context();
    given.task = Config::Value();

    SA_EXPECT_FAILURE(make_anchor_objective(given, section_from("{}"), "the objective block"),
                      StatusCode::InvalidDescription, "the objective block", "'task'");
}
