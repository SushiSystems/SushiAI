/**************************************************************************/
/* test_diagnostics.cpp                                                   */
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
 * @file test_diagnostics.cpp
 * @brief Tests that each host-lane precondition raises a message carrying the values that disagree.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/loss.hpp>
#include "../../common/test_throws.hpp"

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::ValueId;

namespace
{
    /** @brief Differentiates @p loss with respect to @p wrt, for a one-line assertion. */
    void differentiate_wrt(Graph::Graph& graph, ValueId loss, ValueId wrt)
    {
        const std::vector<ValueId> targets = {wrt};
        (void)Autograd::differentiate(graph, loss,
                                      SushiRuntime::span<const ValueId>(targets.data(), targets.size()));
    }
}

TEST(Unit_Diagnostics, ShapeRankBeyondTheMaximumNamesTheRankAndTheLimit)
{
    // The limit in the message is SushiBLAS's MAX_RANK; see tests/functional/unit/README.md.
    SA_EXPECT_ERROR((void)Shape({1, 1, 1, 1, 1, 1, 1}),
                    "Shape rank 7", "exceeds the maximum of 6");
}

TEST(Unit_Diagnostics, ShapeNonPositiveExtentNamesTheOffendingExtent)
{
    // Shape itself refuses a zero-element tensor; see tests/functional/unit/README.md.
    SA_EXPECT_ERROR((void)Shape({4, 0}), "Shape extents must be positive", "got 0");
    SA_EXPECT_ERROR((void)Shape({-3}), "Shape extents must be positive", "got -3");
}

TEST(Unit_Diagnostics, ShapeDimensionOutOfRangeNamesTheIndexAndTheRank)
{
    const Shape shape{4, 3};

    SA_EXPECT_ERROR((void)shape.dim(2), "Dimension 2", "rank-2 shape");
    SA_EXPECT_ERROR((void)shape.dim(-3), "Dimension -3", "rank-2 shape");
}

TEST(Unit_Diagnostics, MatmulContractionMismatchNamesBothInnerExtents)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{4, 5}, "a");
    const ValueId b = builder.input(Shape{7, 3}, "b");

    SA_EXPECT_ERROR((void)builder.matmul(a, b),
                    "MATMUL contracts 5 against 7", "[4, 5]", "[7, 3]");
}

TEST(Unit_Diagnostics, MatmulRejectsARank1OperandAndPrintsWhatItGot)
{
    // The message shows both operand shapes; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId vector = builder.input(Shape{5}, "v");
    const ValueId matrix = builder.input(Shape{5, 3}, "m");

    SA_EXPECT_ERROR((void)builder.matmul(vector, matrix),
                    "MATMUL needs two rank-2 operands", "[5]", "[5, 3]");
    SA_EXPECT_ERROR((void)builder.matmul(matrix, vector),
                    "MATMUL needs two rank-2 operands", "[5, 3]", "[5]");
}

TEST(Unit_Diagnostics, MatmulNamesBothDtypesWhenTheyDisagree)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{4, 5}, "a", DType::FLOAT32);
    const ValueId b = builder.input(Shape{5, 3}, "b", DType::FLOAT64);

    SA_EXPECT_ERROR((void)builder.matmul(a, b),
                    "MATMUL needs operands of one element type", "float32", "float64");
}

TEST(Unit_Diagnostics, BiasAddNamesTheAxisThatDoesNotLineUp)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{4, 3}, "x");
    const ValueId wrong_length = builder.input(Shape{5}, "b5");
    const ValueId rank_two_bias = builder.input(Shape{1, 3}, "b13");
    const ValueId vector = builder.input(Shape{3}, "v");

    SA_EXPECT_ERROR((void)builder.bias_add(x, wrong_length),
                    "BIAS_ADD bias has 5 entries", "[4, 3]", "3 columns");
    SA_EXPECT_ERROR((void)builder.bias_add(x, rank_two_bias),
                    "BIAS_ADD needs a rank-1 bias", "[1, 3]");
    SA_EXPECT_ERROR((void)builder.bias_add(vector, vector),
                    "BIAS_ADD needs a rank-2 left operand", "[3]");
}

TEST(Unit_Diagnostics, SumRowsAndLogSoftmaxRejectARank1Operand)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId vector = builder.input(Shape{6}, "v");

    SA_EXPECT_ERROR((void)builder.sum_rows(vector), "SUM_ROWS needs a rank-2 operand", "[6]");
    SA_EXPECT_ERROR((void)builder.log_softmax(vector), "LOG_SOFTMAX needs a rank-2 operand", "[6]");
}

TEST(Unit_Diagnostics, ElementwiseOperationsNameBothShapesAndBothDtypes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{4, 3}, "a");
    const ValueId b = builder.input(Shape{4, 2}, "b");
    const ValueId wide = builder.input(Shape{4, 3}, "wide", DType::FLOAT64);

    SA_EXPECT_ERROR((void)builder.add(a, b), "ADD needs equally shaped operands",
                    "[4, 3]", "[4, 2]");
    SA_EXPECT_ERROR((void)builder.sub(a, b), "SUB needs equally shaped operands");
    SA_EXPECT_ERROR((void)builder.mul(a, b), "MUL needs equally shaped operands");
    SA_EXPECT_ERROR((void)builder.mul(a, wide), "MUL needs operands of one element type",
                    "float32", "float64");
    SA_EXPECT_ERROR((void)builder.accumulate(a, b), "ACCUMULATE needs equally shaped operands");
}

TEST(Unit_Diagnostics, NllLossNamesTheShapeItCannotServe)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId log_probs = builder.input(Shape{4, 3}, "logp");
    const ValueId mismatched = builder.input(Shape{4, 5}, "targets");
    const ValueId vector = builder.input(Shape{4}, "v");

    SA_EXPECT_ERROR((void)builder.nll_loss(log_probs, mismatched),
                    "NLL_LOSS needs equally shaped operands", "[4, 3]", "[4, 5]");
    SA_EXPECT_ERROR((void)builder.nll_loss(vector, vector),
                    "NLL_LOSS needs rank-2 operands", "[4]");
}

TEST(Unit_Diagnostics, ActivationBackwardRefusesAnOperationThatIsNotOne)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId dy = builder.input(Shape{4, 3}, "dy");

    SA_EXPECT_ERROR((void)builder.activation_backward(Ops::RELU.id, dy, dy),
                    "sushiai.relu", "is not an activation gradient");
}

TEST(Unit_Diagnostics, ASecondProducerNamesBothOperationsAndTheRule)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{4, 3}, "x");
    const ValueId out = graph.add_value(Shape{4, 3}, DType::FLOAT32,
                                        Graph::Lifetime::ACTIVATION, "out");
    graph.add_node(Ops::RELU.id, {x}, {out});

    // The message names both colliding operations; see tests/functional/unit/README.md.
    SA_EXPECT_ERROR((void)graph.add_node(Ops::TANH.id, {x}, {out}),
                    "sushiai.tanh", "sushiai.relu", "single-assignment");
}

TEST(Unit_Diagnostics, AnUnknownOperandNamesTheOperationThatAskedForIt)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{4, 3}, "x");
    const ValueId out = graph.add_value(Shape{4, 3}, DType::FLOAT32,
                                        Graph::Lifetime::ACTIVATION, "out");

    SA_EXPECT_ERROR((void)graph.add_node(Ops::RELU.id, {9999u}, {out}),
                    "sushiai.relu", "reads value 9999", "does not exist");
    SA_EXPECT_ERROR((void)graph.add_node(Ops::RELU.id, {x}, {9999u}),
                    "sushiai.relu", "writes value 9999", "does not exist");
}

TEST(Unit_Diagnostics, LinearNamesItsOwnDimensionsWhenTheInputDoesNotFit)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{8, 5}, "x");

    NN::Linear layer{6, 4};
    SA_EXPECT_ERROR((void)layer.forward(builder, x),
                    "Linear(6, 4)", "[8, 5]", "inner extent must be 6");

    NN::Linear rank_checker{5, 4};
    const ValueId vector = builder.input(Shape{5}, "v");
    SA_EXPECT_ERROR((void)rank_checker.forward(builder, vector),
                    "Linear needs a rank-2 input", "[5]");

    NN::Linear empty{0, 4};
    SA_EXPECT_ERROR((void)empty.forward(builder, x),
                    "Linear needs positive feature counts", "in=0", "out=4");
}

TEST(Unit_Diagnostics, ANonScalarRootNamesTheValueAndItsShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId y = builder.relu(x);

    SA_EXPECT_ERROR(differentiate_wrt(graph, y, x),
                    "needs a scalar to differentiate", "[4, 3]");
}

TEST(Unit_Diagnostics, ALossThatNoRuleCanSeedIsRejectedRatherThanReturningNothing)
{
    // A scalar loss no rule can seed is an error; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{1, 4}, "x");
    const ValueId w = builder.input(Shape{4, 1}, "w");
    const ValueId scalar = builder.matmul(x, w);

    SA_EXPECT_ERROR(differentiate_wrt(graph, scalar, x),
                    "sushiai.matmul", "NLL_LOSS", "CrossEntropyLoss");
}

TEST(Unit_Diagnostics, ALossNoOperationProducesIsRejected)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    // A loss with no producer has no operation to name; see tests/functional/unit/README.md.
    const ValueId orphan = graph.add_value(Shape{1}, DType::FLOAT32,
                                           Graph::Lifetime::ACTIVATION, "orphan");
    SA_EXPECT_ERROR(differentiate_wrt(graph, orphan, orphan), "nothing produces it", "NLL_LOSS");

    // A leaf does have a producing node, and it is named.
    const ValueId leaf = builder.parameter(Shape{1}, "leaf");
    SA_EXPECT_ERROR(differentiate_wrt(graph, leaf, leaf), "sushiai.leaf", "NLL_LOSS");
}

TEST(Unit_Diagnostics, AnOperationWithNoBackwardRuleNamesItselfAndItsNode)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");

    const ValueId mystery = graph.add_value(Shape{4, 3}, DType::FLOAT32,
                                            Graph::Lifetime::ACTIVATION, "mystery");
    graph.add_node(OpID(0x1234'5678u), {x}, {mystery});

    const ValueId loss = builder.nll_loss(mystery, targets);

    // A foreign identity prints as sushiai.unknown; see tests/functional/unit/README.md.
    SA_EXPECT_ERROR(differentiate_wrt(graph, loss, x),
                    "No backward rule for", "sushiai.unknown", "node 2");
}

TEST(Unit_Diagnostics, LogSoftmaxOutsideTheFusedPairExplainsTheRestriction)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{4, 3}, "x");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");
    const ValueId detoured = builder.relu(builder.log_softmax(x));
    const ValueId loss = builder.nll_loss(detoured, targets);

    // The message tells the caller to route through NLL_LOSS; see tests/functional/unit/README.md.
    SA_EXPECT_ERROR(differentiate_wrt(graph, loss, x),
                    "LOG_SOFTMAX", "row-wise reduction", "CrossEntropyLoss");
}

TEST(Unit_Diagnostics, AGradientContributedToTheWrongShapeNamesBothShapes)
{
    // A SUB with a mis-shaped result provokes the error; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.parameter(Shape{2, 6}, "x");
    const ValueId other = builder.input(Shape{4, 3}, "other");
    const ValueId targets = builder.input(Shape{4, 3}, "targets");

    const ValueId difference = graph.add_value(Shape{4, 3}, DType::FLOAT32,
                                               Graph::Lifetime::ACTIVATION, "bad_sub");
    graph.add_node(Ops::SUB.id, {x, other}, {difference});

    const ValueId loss = builder.nll_loss(difference, targets);

    SA_EXPECT_ERROR(differentiate_wrt(graph, loss, x),
                    "gradient of shape [4, 3]", "shape [2, 6]");
}
