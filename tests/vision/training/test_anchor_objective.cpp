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
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
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


    /** @brief Names a head list the layout does not take and what the refusal names. */
    struct RefusedHeadCount
    {
        std::vector<AnchorHead> heads; /**< The heads given. */
        const char* count;             /**< The count the message states. */
    };

    /** @brief Provides the head counts other than the layout's. */
    class Small_AnchorObjectiveHeadCount : public ::testing::TestWithParam<RefusedHeadCount>
    {

    };

    /** @brief Names a head list that the validators refuse and what the refusal names. */
    struct RefusedOptions
    {
        std::vector<AnchorHead> heads; /**< The heads given. */
        const char* needle;            /**< What the message names. */
    };

    /** @brief Returns the heads of the layout with @p change applied to head @p index. */
    std::vector<AnchorHead> heads_with(std::size_t index, void (*change)(AnchorHead&))
    {
        std::vector<AnchorHead> heads = two_heads();
        change(heads[index]);
        return heads;
    }

    /** @brief Provides the options the validators refuse. */
    class Small_AnchorObjectiveOptions : public ::testing::TestWithParam<RefusedOptions>
    {

    };
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

/**
 * @brief Tests that the root is minus one over the batch times the sum of delta times head, over
 *        every head, each head meeting the leaf of its own name.
 */
TEST(Small_AnchorObjective, TheRootIsTheDeltaLossSummedOverTheHeads)
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

    SA_ASSERT_OK_AND_ASSIGN(const ValueID root,
                            objective.loss(builder, graph, outputs, leaves));

    EXPECT_EQ(graph.value(root).shape, (Shape{1}));

    // -(448 * 1 * 2 + 112 * 2 * 3) / 2, every value exact in double.
    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(leaves.at("delta.head.0"), std::vector<double>(448, 1.0)));
    SA_EXPECT_OK(evaluator.write(leaves.at("delta.head.1"), std::vector<double>(112, 2.0)));
    SA_EXPECT_OK(evaluator.write(fine, std::vector<double>(448, 2.0)));
    SA_EXPECT_OK(evaluator.write(coarse, std::vector<double>(112, 3.0)));
    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> value, evaluator.read(root));
    EXPECT_EQ(value, (std::vector<double>{-784.0}));
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

/** @brief Tests that the cost is the field-by-field sum of the two heads' costs. */
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

    // Head 0 holds the squares 1, 4, 9 and head 1 the squares 1, 4; the batch of 2 divides; all exact.
    EXPECT_EQ(cost.total, 7.0 + 2.5);
    EXPECT_EQ(cost.non_box, 6.5 + 0.25);
    EXPECT_EQ(cost.box, 0.5 + 2.25);
    EXPECT_EQ(cost.stored, 14.0 + 5.0);
}

/** @brief Tests that no batch alone feeds the objective: its own step does. */
TEST(Small_AnchorObjective, NoBatchAloneFeedsIt)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorObjective objective,
                            AnchorObjective::make(network(), two_head_layout(), two_heads()));

    EXPECT_FALSE(objective.is_fed_by_batch());
}

/** @brief Tests that a head count other than the layout's is InvalidDescription. */
TEST_P(Small_AnchorObjectiveHeadCount, AHeadCountOtherThanTheLayoutsIsRefused)
{
    const RefusedHeadCount& row = GetParam();

    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), row.heads),
                      StatusCode::InvalidDescription, row.count, "holds 2");
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_AnchorObjectiveHeadCount,
                         ::testing::Values(RefusedHeadCount{{}, "0 heads"},
                                           RefusedHeadCount{{make_head("head.0", 4)},
                                                            "1 heads"}));

/** @brief Tests that heads declared with different batches are InvalidDescription. */
TEST(Small_AnchorObjective, HeadsOfDifferentBatchesAreRefused)
{
    std::vector<AnchorHead> heads = two_heads();
    heads[1].shape.batch = 3;

    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), heads),
                      StatusCode::InvalidDescription, "head.1", "batch of 3", "head.0");
}

/** @brief Tests that an empty output name is InvalidDescription, naming the head. */
TEST(Small_AnchorObjective, AnEmptyOutputNameIsRefused)
{
    std::vector<AnchorHead> unnamed = two_heads();
    unnamed[1].output.clear();

    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), unnamed),
                      StatusCode::InvalidDescription, "head 1", "names no model output");
}

/** @brief Tests that two heads reading one output are InvalidDescription, naming the output. */
TEST(Small_AnchorObjective, ARepeatedOutputNameIsRefused)
{
    std::vector<AnchorHead> repeated = two_heads();
    repeated[1].output = "head.0";

    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), repeated),
                      StatusCode::InvalidDescription, "both read", "head.0");
}

/** @brief Tests that an extent below 1 is InvalidDescription, naming the head and its shape. */
TEST(Small_AnchorObjective, AnExtentBelowOneIsRefused)
{
    std::vector<AnchorHead> flat = two_heads();
    flat[0].shape.height = 0;

    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), flat),
                      StatusCode::InvalidDescription, "head.0", "[2, 0, 4]");
}

/** @brief Tests that a network side below 1 is InvalidDescription, naming both sides. */
TEST(Small_AnchorObjective, ANetworkSideBelowOneIsRefused)
{
    SA_EXPECT_FAILURE(AnchorObjective::make(NetworkSize{0, 64}, two_head_layout(), two_heads()),
                      StatusCode::InvalidDescription, "0 by 64");
}

/** @brief Tests that options the head and the training validators refuse are passed on. */
TEST_P(Small_AnchorObjectiveOptions, OptionsTheValidatorsRefuseAreRefused)
{
    const RefusedOptions& row = GetParam();

    SA_EXPECT_FAILURE(AnchorObjective::make(network(), two_head_layout(), row.heads),
                      StatusCode::InvalidDescription, row.needle);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_AnchorObjectiveOptions,
    ::testing::Values(
        RefusedOptions{heads_with(0, [](AnchorHead& head) { head.options.num_classes = 0; }),
                       "at least one class"},
        RefusedOptions{heads_with(1, [](AnchorHead& head) { head.training.box_weight = -1.0f; }),
                       "box weight of an anchor head"}));
