/**************************************************************************/
/* test_anchor_objective.cpp                                              */
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
 * @file test_anchor_objective.cpp
 * @brief Tests the leaves, the root, the summed cost and the refusals of AnchorObjective.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/training/anchor_objective.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::ValueID;
using Vision::Detect::Anchor::AnchorLayout;
using Vision::Detect::Anchor::HeadCost;
using Vision::Detect::Anchor::HeadDelta;
using Vision::Detect::Anchor::NetworkSize;
using Vision::Training::AnchorHead;
using Vision::Training::AnchorObjective;

namespace
{
    /** @brief Bounds the error of a sum of a few small integers and halves, held in double. */
    constexpr double K_COST_TOLERANCE = 1e-12;

    /** @brief Bounds the error allowed on the factor a SCALE node stores. */
    constexpr double K_ATTRIBUTE_TOLERANCE = 1e-12;

    /** @brief Holds the batch every head of these tests is declared with. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the classes of every head; a slot then has 5 + 2 entries. */
    constexpr std::int64_t K_CLASSES = 2;

    /** @brief Returns a 64 by 64 pixel network input. */
    NetworkSize network()
    {
        return NetworkSize{64, 64};
    }

    /** @brief Returns four anchors, two for each of two heads. */
    AnchorLayout two_head_layout()
    {
        AnchorLayout layout;
        layout.anchors = {{0.2f, 0.2f}, {0.4f, 0.3f}, {0.5f, 0.6f}, {0.9f, 0.8f}};
        layout.head_anchors = {{0, 1}, {2, 3}};
        return layout;
    }

    /** @brief Returns a head published as @p output on a @p side by @p side grid. */
    AnchorHead make_head(std::string output, std::int64_t side)
    {
        AnchorHead head;
        head.output = std::move(output);
        head.shape = {K_BATCH, side, side};
        head.options.num_classes = K_CLASSES;
        return head;
    }

    /** @brief Returns "head.0" on a 4 by 4 grid and "head.1" on a 2 by 2 grid. */
    std::vector<AnchorHead> two_heads()
    {
        return {make_head("head.0", 4), make_head("head.1", 2)};
    }

    /** @brief Returns the operation names of nodes [from, end). */
    std::vector<std::string> op_names(const Graph::Graph& graph, std::size_t from)
    {
        std::vector<std::string> names;
        for (std::size_t i = from; i < graph.node_count(); ++i)
            names.emplace_back(Ops::name_of(graph.nodes()[i].op));
        return names;
    }
} // namespace

/** @brief Tests that declare adds one float32 leaf per head, in head order and head shape. */
TEST(Small_AnchorObjective, DeclaresOneLeafPerHeadInTheShapeOfItsOutput)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            AnchorObjective::make(network(), two_head_layout(), two_heads()));

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective.declare(builder, leaves);

    ASSERT_EQ(leaves.all().size(), 2u);
    EXPECT_EQ(leaves.all()[0].first, "delta.head.0");
    EXPECT_EQ(leaves.all()[1].first, "delta.head.1");
    EXPECT_EQ(objective.leaf_role(1), "delta.head.1");

    // Two anchors of 5 + 2 entries give 14 channels.
    const Graph::Value& fine = graph.value(leaves.at("delta.head.0"));
    const Graph::Value& coarse = graph.value(leaves.at("delta.head.1"));
    EXPECT_EQ(fine.shape, (Shape{2, 4, 4, 14}));
    EXPECT_EQ(coarse.shape, (Shape{2, 2, 2, 14}));
    EXPECT_EQ(fine.dtype, DType::Float32);
    EXPECT_EQ(coarse.dtype, DType::Float32);
    EXPECT_FALSE(fine.requires_grad);
    EXPECT_FALSE(coarse.requires_grad);
}

/** @brief Tests that the root adds one delta loss per head, each scaled by -1 over the batch. */
TEST(Small_AnchorObjective, TheRootSumsOneDeltaLossPerHeadScaledByTheBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            AnchorObjective::make(network(), two_head_layout(), two_heads()));

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective.declare(builder, leaves);

    const ValueID fine = builder.parameter(Shape{2, 4, 4, 14}, "fine");
    const ValueID coarse = builder.parameter(Shape{2, 2, 2, 14}, "coarse");
    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("head.1", coarse));
    SA_EXPECT_OK(outputs.add("head.0", fine));
    const std::size_t first = graph.node_count();

    SA_ASSERT_OK_AND_ASSIGN(const ValueID root,
                            objective.loss(builder, graph, outputs, leaves));

    ASSERT_EQ(op_names(graph, first), (std::vector<std::string>{
        "sushiai.mul", "sushiai.reshape", "sushiai.sum_rows", "sushiai.scale",
        "sushiai.mul", "sushiai.reshape", "sushiai.sum_rows", "sushiai.scale",
        "sushiai.add",
    }));

    const Graph::Node& fine_product = graph.nodes()[first];
    const Graph::Node& fine_scale = graph.nodes()[first + 3];
    const Graph::Node& coarse_product = graph.nodes()[first + 4];
    const Graph::Node& coarse_scale = graph.nodes()[first + 7];
    const Graph::Node& sum = graph.nodes()[first + 8];

    // Each head meets the leaf of its own name, whatever order the model published them in.
    EXPECT_EQ(fine_product.inputs, (std::vector<ValueID>{leaves.at("delta.head.0"), fine}));
    EXPECT_EQ(coarse_product.inputs, (std::vector<ValueID>{leaves.at("delta.head.1"), coarse}));
    EXPECT_NEAR(fine_scale.attributes.get(0), -1.0 / 2.0, K_ATTRIBUTE_TOLERANCE);
    EXPECT_NEAR(coarse_scale.attributes.get(0), -1.0 / 2.0, K_ATTRIBUTE_TOLERANCE);
    EXPECT_EQ(sum.inputs,
              (std::vector<ValueID>{fine_scale.outputs.at(0), coarse_scale.outputs.at(0)}));
    EXPECT_EQ(sum.outputs, (std::vector<ValueID>{root}));
    EXPECT_EQ(graph.value(root).shape, (Shape{1}));
}

/** @brief Tests that a model that does not publish a head is refused by the head's name. */
TEST(Small_AnchorObjective, AModelWithoutAHeadIsRefusedNamingTheHead)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            AnchorObjective::make(network(), two_head_layout(), two_heads()));

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective.declare(builder, leaves);

    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("head.0", builder.parameter(Shape{2, 4, 4, 14}, "fine")));

    SA_EXPECT_FAILURE(objective.loss(builder, graph, outputs, leaves),
                      StatusCode::InvalidDescription, "head.1");
}

/** @brief Tests that a head whose shape differs from the declared one is refused, naming both. */
TEST(Small_AnchorObjective, AHeadOfAnotherShapeIsRefusedNamingBothShapes)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            AnchorObjective::make(network(), two_head_layout(), two_heads()));

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective.declare(builder, leaves);

    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("head.0", builder.parameter(Shape{2, 4, 4, 21}, "fine")));
    SA_EXPECT_OK(outputs.add("head.1", builder.parameter(Shape{2, 2, 2, 14}, "coarse")));

    SA_EXPECT_FAILURE(objective.loss(builder, graph, outputs, leaves),
                      StatusCode::InvalidDescription, "[2, 4, 4, 14]", "[2, 4, 4, 21]");
}

/** @brief Tests that the cost is the field-by-field sum of the two heads' costs, to 1e-12. */
TEST(Small_AnchorObjective, TheCostIsTheSumOverTheHeads)
{
    // One anchor and one class per head: a slot is x, y, w, h, objectness, class.
    AnchorLayout layout;
    layout.anchors = {{0.2f, 0.2f}, {0.6f, 0.6f}};
    layout.head_anchors = {{0}, {1}};

    std::vector<AnchorHead> heads = {make_head("head.0", 1), make_head("head.1", 1)};
    heads[0].options.num_classes = 1;
    heads[1].options.num_classes = 1;
    heads[1].training.object_weight = 0.5f;

    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            AnchorObjective::make(network(), layout, heads));

    std::vector<HeadDelta> deltas(2);
    deltas[0].delta.assign(12, 0.0f);
    deltas[0].delta[0] = 1.0f;
    deltas[0].delta[4] = 2.0f;
    deltas[0].delta[11] = 3.0f;
    deltas[1].delta.assign(12, 0.0f);
    deltas[1].delta[4] = 1.0f;
    deltas[1].delta[8] = 2.0f;

    const HeadCost cost = objective.measure_cost(
        SushiRuntime::span<const HeadDelta>(deltas.data(), deltas.size()));

    // Head 0 holds the squares 1, 4, 9 and head 1 the squares 1, 4; the batch of 2 divides.
    EXPECT_NEAR(cost.total, 7.0 + 2.5, K_COST_TOLERANCE);
    EXPECT_NEAR(cost.non_box, 6.5 + 0.25, K_COST_TOLERANCE);
    EXPECT_NEAR(cost.box, 0.5 + 2.25, K_COST_TOLERANCE);
    EXPECT_NEAR(cost.stored, 14.0 + 5.0, K_COST_TOLERANCE);
}

/** @brief Tests that the name is "anchor_delta" and that no batch alone feeds the objective. */
TEST(Small_AnchorObjective, NamesItselfAndIsFedByItsOwnStep)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            AnchorObjective::make(network(), two_head_layout(), two_heads()));

    EXPECT_FALSE(objective.is_fed_by_batch());
    EXPECT_EQ(objective.name(), "anchor_delta");
    EXPECT_EQ(objective.heads().size(), 2u);
    EXPECT_EQ(objective.layout().anchors.size(), 4u);
    EXPECT_EQ(objective.network().width, 64);
}

/** @brief Tests that a head count other than the layout's is InvalidDescription. */
TEST(Small_AnchorObjective, AHeadCountOtherThanTheLayoutsIsRefused)
{
    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), {}),
                      StatusCode::InvalidDescription, "0 heads", "holds 2");
    SA_EXPECT_FAILURE(
        AnchorObjective::make(network(), two_head_layout(), {make_head("head.0", 4)}),
        StatusCode::InvalidDescription, "1 heads", "holds 2");
}

/** @brief Tests that heads declared with different batches are InvalidDescription. */
TEST(Small_AnchorObjective, HeadsOfDifferentBatchesAreRefused)
{
    std::vector<AnchorHead> heads = two_heads();
    heads[1].shape.batch = 3;

    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), heads),
                      StatusCode::InvalidDescription, "head.1", "batch of 3", "head.0");
}

/** @brief Tests that an empty and a repeated output name are InvalidDescription. */
TEST(Small_AnchorObjective, AnEmptyOrARepeatedOutputNameIsRefused)
{
    std::vector<AnchorHead> unnamed = two_heads();
    unnamed[1].output.clear();
    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), unnamed),
                      StatusCode::InvalidDescription, "head 1", "names no model output");

    std::vector<AnchorHead> repeated = two_heads();
    repeated[1].output = "head.0";
    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), repeated),
                      StatusCode::InvalidDescription, "both read", "head.0");
}

/** @brief Tests that an extent or a network side below 1 is InvalidDescription. */
TEST(Small_AnchorObjective, AnExtentOrANetworkSideBelowOneIsRefused)
{
    std::vector<AnchorHead> flat = two_heads();
    flat[0].shape.height = 0;
    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), flat),
                      StatusCode::InvalidDescription, "head.0", "[2, 0, 4]");

    SA_EXPECT_FAILURE(AnchorObjective::make(NetworkSize{0, 64}, two_head_layout(), two_heads()),
                      StatusCode::InvalidDescription, "0 by 64");
}

/** @brief Tests that options the head and the training validators refuse are passed on. */
TEST(Small_AnchorObjective, OptionsTheValidatorsRefuseAreRefused)
{
    std::vector<AnchorHead> classless = two_heads();
    classless[0].options.num_classes = 0;
    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), classless),
                      StatusCode::InvalidDescription, "at least one class");

    std::vector<AnchorHead> weightless = two_heads();
    weightless[1].training.box_weight = -1.0f;
    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), weightless),
                      StatusCode::InvalidDescription);
}
