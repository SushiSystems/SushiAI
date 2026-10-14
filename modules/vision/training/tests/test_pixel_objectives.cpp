/**************************************************************************/
/* test_pixel_objectives.cpp                                              */
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
 * @file test_pixel_objectives.cpp
 * @brief Tests the leaves, the graphs and the factories of the two per-pixel objectives.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/vision/training/dice_objective.hpp>
#include <SushiAI/vision/training/pixel_cross_entropy_objective.hpp>
#include <SushiAI/vision/training/pixel_extents.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using Vision::Training::DiceObjective;
using Vision::Training::make_dice_objective;
using Vision::Training::make_pixel_cross_entropy_objective;
using Vision::Training::PixelCrossEntropyObjective;
using Vision::Training::PixelExtents;

namespace
{
    /** @brief Holds the extents of the logits the cases score: N, H, W and C. */
    constexpr PixelExtents K_EXTENTS{2, 4, 4, 3};

    /** @brief Holds the channels of the input the test model reads. */
    constexpr std::int64_t K_INPUT_CHANNELS = 5;

    /** @brief Holds the pixels of a batch, N * H * W. */
    constexpr std::int64_t K_PIXELS = 32;

    /** @brief Names the objective block in a diagnostic. */
    constexpr std::string_view K_CONTEXT = "the objective block";

    /** @brief Returns a biased 1x1 convolution from K_INPUT_CHANNELS to C channels. */
    NN::Conv2d make_model()
    {
        NN::Conv2d model;
        model.in_channels = K_INPUT_CHANNELS;
        model.out_channels = K_EXTENTS.classes;

        return model;
    }

    /** @brief Returns the shape of the batch the test model reads. */
    Shape input_shape()
    {
        return Shape{K_EXTENTS.batch, K_EXTENTS.height, K_EXTENTS.width, K_INPUT_CHANNELS};
    }

    /** @brief Counts the nodes of @p graph whose operation is @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                ++count;
        }

        return count;
    }

    /** @brief Returns what a factory is given for the test model over @p classes classes. */
    Train::ObjectiveContext make_context(std::int64_t classes)
    {
        Train::ObjectiveContext given;
        given.batch = K_EXTENTS.batch;
        given.classes = classes;
        given.input_sample = Shape{K_EXTENTS.height, K_EXTENTS.width, K_INPUT_CHANNELS};
        given.outputs.push_back(Train::OutputShape{
            "output",
            Shape{K_EXTENTS.batch, K_EXTENTS.height, K_EXTENTS.width, K_EXTENTS.classes}});

        return given;
    }

    /** @brief Parses @p text as the keys of an objective block. */
    Config::Value keys(std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    }
} // namespace

/** @brief Tests that make() keeps what it was given and refuses an extent below 1. */
TEST(Unit_VisionTrainingPixelObjectives, MakeKeepsItsExtentsAndRefusesNone)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PixelCrossEntropyObjective> objective,
                            PixelCrossEntropyObjective::make(K_EXTENTS, 7));

    EXPECT_EQ(objective->extents().pixels(), K_PIXELS);
    EXPECT_EQ(objective->extents().classes, 3);
    EXPECT_EQ(objective->ignore_index(), 7);
    EXPECT_EQ(objective->name(), "pixel_cross_entropy");

    SA_EXPECT_FAILURE(PixelCrossEntropyObjective::make(PixelExtents{2, 0, 4, 3}),
                      StatusCode::InvalidDescription, "pixel_cross_entropy", "at least 1");
    SA_EXPECT_FAILURE(DiceObjective::make(PixelExtents{2, 4, 4, 0}),
                      StatusCode::InvalidDescription, "dice", "at least 1");
}

/** @brief Tests the ignore index that names a class and the element type that is no real. */
TEST(Unit_VisionTrainingPixelObjectives, MakeRefusesAnIgnoreIndexThatIsAClass)
{
    SA_EXPECT_FAILURE(PixelCrossEntropyObjective::make(K_EXTENTS, 2),
                      StatusCode::InvalidDescription, "ignore index 2", "3 classes");
    SA_EXPECT_OK(PixelCrossEntropyObjective::make(K_EXTENTS, 3));
    SA_EXPECT_OK(PixelCrossEntropyObjective::make(K_EXTENTS, -1));
    SA_EXPECT_FAILURE(DiceObjective::make(K_EXTENTS, 0), StatusCode::InvalidDescription,
                      "ignore index 0");
    SA_EXPECT_FAILURE(PixelCrossEntropyObjective::make(K_EXTENTS, 255, DType::Int32),
                      StatusCode::InvalidDescription, "float32");
}

/** @brief Tests the largest batch of 512 by 512 pixels over 150 classes INT32 indexes: 54. */
TEST(Unit_VisionTrainingPixelObjectives, MakeRefusesWhatInt32CannotIndex)
{
    // 54 * 512 * 512 * 150 is 2 123 366 400; one image more is 2 162 688 000.
    SA_EXPECT_OK(PixelCrossEntropyObjective::make(PixelExtents{54, 512, 512, 150}));
    SA_EXPECT_FAILURE(PixelCrossEntropyObjective::make(PixelExtents{55, 512, 512, 150}),
                      StatusCode::InvalidDescription, "55 * 512 * 512 * 150", "2147483647");

    // The Dice objective gathers one probability per pixel, so the same bound holds for it.
    SA_EXPECT_OK(DiceObjective::make(PixelExtents{54, 512, 512, 150}));
    SA_EXPECT_FAILURE(DiceObjective::make(PixelExtents{55, 512, 512, 150}),
                      StatusCode::InvalidDescription, "55 * 512 * 512 * 150", "2147483647");
}

/** @brief Tests the two leaves of the cross entropy and the mask role it reads. */
TEST(Unit_VisionTrainingPixelObjectives, TheCrossEntropyDeclaresIndicesAndWeights)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PixelCrossEntropyObjective> objective,
                            PixelCrossEntropyObjective::make(K_EXTENTS));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;

    objective->declare(builder, leaves);

    ASSERT_EQ(leaves.all().size(), 2u);
    EXPECT_EQ(graph.value(leaves.at("targets")).shape, (Shape{K_PIXELS}));
    EXPECT_EQ(graph.value(leaves.at("targets")).dtype, DType::Int32);
    EXPECT_EQ(graph.value(leaves.at("weights")).shape, (Shape{K_PIXELS, 1}));
    EXPECT_EQ(graph.value(leaves.at("weights")).dtype, DType::Float32);

    ASSERT_EQ(leaves.required().size(), 1u);
    const Data::TargetDescription& mask = leaves.required().front();
    EXPECT_EQ(mask.role, Data::K_ROLE_MASK);
    EXPECT_EQ(mask.sample_shape, (Shape{4, 4}));
    EXPECT_EQ(mask.dtype, DType::Int32);
    EXPECT_EQ(mask.classes, 0);
}

/** @brief Tests the five leaves of the Dice objective, none of them [R, C]. */
TEST(Unit_VisionTrainingPixelObjectives, TheDiceDeclaresTwoIndexLeavesAndNoOneHot)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DiceObjective> objective,
                            DiceObjective::make(K_EXTENTS));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;

    objective->declare(builder, leaves);

    ASSERT_EQ(leaves.all().size(), 5u);
    EXPECT_FALSE(leaves.has("one_hot"));
    EXPECT_EQ(graph.value(leaves.at("score_indices")).shape, (Shape{K_PIXELS}));
    EXPECT_EQ(graph.value(leaves.at("score_indices")).dtype, DType::Int32);
    EXPECT_EQ(graph.value(leaves.at("class_indices")).shape, (Shape{K_PIXELS}));
    EXPECT_EQ(graph.value(leaves.at("class_indices")).dtype, DType::Int32);
    for (const auto& leaf : leaves.all())
        EXPECT_LE(graph.value(leaf.second).shape.num_elements(), K_PIXELS) << leaf.first;
    EXPECT_EQ(graph.value(leaves.at("kept")).shape, (Shape{K_PIXELS, 1}));
    EXPECT_EQ(graph.value(leaves.at("class_pixels")).shape, (Shape{3}));
    EXPECT_EQ(graph.value(leaves.at("class_weights")).shape, (Shape{3}));
    ASSERT_EQ(leaves.required().size(), 1u);
    EXPECT_EQ(leaves.required().front().role, Data::K_ROLE_MASK);
}

/** @brief Tests that the cross entropy is one element built from one gather and no one-hot. */
TEST(Unit_VisionTrainingPixelObjectives, TheCrossEntropyGathersOneLogProbabilityPerPixel)
{
    NN::Conv2d model = make_model();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PixelCrossEntropyObjective> objective,
                            PixelCrossEntropyObjective::make(K_EXTENTS));

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, *objective, input_shape()));

    EXPECT_EQ(built.graph.value(built.spec.loss).shape.num_elements(), 1);
    EXPECT_EQ(built.graph.value(built.spec.targets).dtype, DType::Int32);
    EXPECT_EQ(count_nodes(built.graph, Ops::K_NLL_LOSS.id), 0u);
    EXPECT_EQ(count_nodes(built.graph, Ops::K_LOG_SOFTMAX.id), 1u);
    EXPECT_EQ(count_nodes(built.graph, Ops::K_GATHER_ROWS.id), 1u);
    ASSERT_EQ(built.parameters.size(), 2u);
    EXPECT_NE(built.parameters[0].gradient, Graph::K_NO_VALUE);
    EXPECT_NE(built.parameters[1].gradient, Graph::K_NO_VALUE);
}

/** @brief Tests that the Dice loss is one element whose gradient crosses the scatter-add. */
TEST(Unit_VisionTrainingPixelObjectives, TheDiceLossIsDifferentiated)
{
    NN::Conv2d model = make_model();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<DiceObjective> objective,
                            DiceObjective::make(K_EXTENTS));

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, *objective, input_shape()));

    EXPECT_EQ(built.graph.value(built.spec.loss).shape.num_elements(), 1);
    EXPECT_EQ(built.spec.targets, Graph::K_NO_VALUE);

    // The forward pass emits one of each; each backward rule appends the other operation.
    EXPECT_EQ(count_nodes(built.graph, Ops::K_GATHER_ROWS.id), 2u);
    EXPECT_EQ(count_nodes(built.graph, Ops::K_SCATTER_ADD_ROWS.id), 2u);
    ASSERT_EQ(built.parameters.size(), 2u);
    EXPECT_NE(built.parameters[0].gradient, Graph::K_NO_VALUE);
    EXPECT_NE(built.parameters[1].gradient, Graph::K_NO_VALUE);
}

/** @brief Tests that both factories read N, H, W and C from the model's one output. */
TEST(Unit_VisionTrainingPixelObjectives, TheFactoriesReadTheModelsOutput)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> cross_entropy,
        make_pixel_cross_entropy_objective(make_context(3), keys("{}"), K_CONTEXT));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> dice,
        make_dice_objective(make_context(0), keys(R"({"ignore_index": 9})"), K_CONTEXT));

    EXPECT_EQ(cross_entropy->name(), "pixel_cross_entropy");
    const auto* const made = dynamic_cast<const PixelCrossEntropyObjective*>(cross_entropy.get());
    ASSERT_NE(made, nullptr);
    EXPECT_EQ(made->extents().pixels(), K_PIXELS);
    EXPECT_EQ(made->ignore_index(), 255);

    const auto* const made_dice = dynamic_cast<const DiceObjective*>(dice.get());
    ASSERT_NE(made_dice, nullptr);
    EXPECT_EQ(made_dice->ignore_index(), 9);
}

/** @brief Tests the keys a factory refuses: an unknown one, a real and an index of a class. */
TEST(Unit_VisionTrainingPixelObjectives, TheFactoriesRefuseABadSection)
{
    SA_EXPECT_FAILURE(
        make_pixel_cross_entropy_objective(make_context(3), keys(R"({"weight": 1})"), K_CONTEXT),
        StatusCode::InvalidDescription, "the objective block", "'weight'", "'ignore_index'");
    SA_EXPECT_FAILURE(
        make_dice_objective(make_context(3), keys(R"({"ignore_index": 0.5})"), K_CONTEXT),
        StatusCode::InvalidDescription, "'ignore_index'", "whole number");
    SA_EXPECT_FAILURE(
        make_pixel_cross_entropy_objective(make_context(3), keys(R"({"ignore_index": 1})"),
                                           K_CONTEXT),
        StatusCode::InvalidDescription, "ignore index 1");
}

/** @brief Tests the models a factory refuses, and data of more classes than the model scores. */
TEST(Unit_VisionTrainingPixelObjectives, TheFactoriesRefuseAModelThatDoesNotFit)
{
    Train::ObjectiveContext two_outputs = make_context(3);
    two_outputs.outputs.push_back(Train::OutputShape{"second", Shape{2, 4, 4, 3}});
    Train::ObjectiveContext flat = make_context(3);
    flat.outputs.front().shape = Shape{2, 3};
    Train::ObjectiveContext half_size = make_context(3);
    half_size.outputs.front().shape = Shape{2, 2, 2, 3};

    SA_EXPECT_FAILURE(make_pixel_cross_entropy_objective(two_outputs, keys("{}"), K_CONTEXT),
                      StatusCode::InvalidDescription, "one [N, H, W, classes] output",
                      "publishes 2");
    SA_EXPECT_FAILURE(make_dice_objective(flat, keys("{}"), K_CONTEXT),
                      StatusCode::InvalidDescription, "'output'", "[2, 3]");
    SA_EXPECT_FAILURE(make_pixel_cross_entropy_objective(half_size, keys("{}"), K_CONTEXT),
                      StatusCode::InvalidDescription, "[2, 2, 2, 3]", "[4, 4, 5]",
                      "input size");
    SA_EXPECT_FAILURE(make_dice_objective(make_context(4), keys("{}"), K_CONTEXT),
                      StatusCode::InvalidDescription, "holds 4 classes", "scores 3");
}
