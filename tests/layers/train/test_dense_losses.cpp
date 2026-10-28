/**************************************************************************/
/* test_dense_losses.cpp                                                  */
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
 * @file test_dense_losses.cpp
 * @brief Tests the loss graphs that compare an output with a target element by element.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/train/dense_losses.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Returns the nodes of @p graph that are no leaf, in emission order. */
    std::vector<const Graph::Node*> emitted_nodes(const Graph::Graph& graph)
    {
        std::vector<const Graph::Node*> emitted;
        emitted.reserve(graph.node_count());
        for (const Graph::Node& node : graph.nodes())
            if (node.op != Ops::K_LEAF.id)
                emitted.push_back(&node);

        return emitted;
    }

    /** @brief Returns the operation of every node of @p graph that is no leaf, in emission order. */
    std::vector<OpID> operations_of(const Graph::Graph& graph)
    {
        std::vector<OpID> operations;
        for (const Graph::Node* const node : emitted_nodes(graph))
            operations.push_back(node->op);

        return operations;
    }

    /** @brief Returns the value the node that produced @p id reads as operand @p operand. */
    Graph::ValueID operand_of(const Graph::Graph& graph, Graph::ValueID id, std::size_t operand)
    {
        return graph.node(graph.value(id).producer).inputs.at(operand);
    }
} // namespace

/** @brief Tests that the mean is a reshape to one column, a column sum and a scale. */
TEST(Small_DenseLosses, TheMeanIsOneColumnSummedAndScaled)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID values = builder.input(Shape{4, 3}, "values");

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID mean,
                            Train::emit_mean_of_elements(builder, graph, values));

    EXPECT_EQ(operations_of(graph), (std::vector<OpID>{Ops::K_RESHAPE.id, Ops::K_SUM_ROWS.id,
                                                       Ops::K_SCALE.id}));
    EXPECT_EQ(graph.value(mean).shape.num_elements(), 1);
    EXPECT_EQ(graph.value(operand_of(graph, operand_of(graph, mean, 0), 0)).shape,
              (Shape{12, 1}));
}

/** @brief Tests that the logistic loss is built from the logit in its stable form. */
TEST(Small_DenseLosses, TheLogisticLossReadsTheLogitInItsStableForm)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID logits = builder.input(Shape{4, 3}, "logits");
    const Graph::ValueID targets = builder.input(Shape{4, 3}, "targets");

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            Train::emit_logistic_loss(builder, graph, logits, targets));

    // max(x, 0), x * y, then log(1 + exp(-|x|)); see tests/layers/train/README.md.
    EXPECT_EQ(operations_of(graph),
              (std::vector<OpID>{Ops::K_RELU.id, Ops::K_MUL.id, Ops::K_ABS.id, Ops::K_SCALE.id,
                                 Ops::K_SOFTPLUS.id, Ops::K_SUB.id, Ops::K_ADD.id,
                                 Ops::K_RESHAPE.id, Ops::K_SUM_ROWS.id, Ops::K_SCALE.id}));
    EXPECT_EQ(graph.value(loss).shape.num_elements(), 1);
    EXPECT_EQ(graph.value(loss).dtype, DType::Float32);

    // The exponential is taken of minus the magnitude, never of the logit itself.
    const Graph::Node& softplus = *emitted_nodes(graph).at(4);
    const Graph::Node& negated = graph.node(graph.value(softplus.inputs.at(0)).producer);
    EXPECT_EQ(negated.op, Ops::K_SCALE.id);
    EXPECT_EQ(graph.node(graph.value(negated.inputs.at(0)).producer).op, Ops::K_ABS.id);
    EXPECT_EQ(emitted_nodes(graph).at(2)->inputs.at(0), logits);
}

/** @brief Tests that the squared error multiplies the difference by itself. */
TEST(Small_DenseLosses, TheSquaredErrorSquaresTheDifference)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID outputs = builder.input(Shape{4, 2}, "outputs");
    const Graph::ValueID targets = builder.input(Shape{4, 2}, "targets");

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            Train::emit_squared_error(builder, graph, outputs, targets));

    EXPECT_EQ(operations_of(graph),
              (std::vector<OpID>{Ops::K_SUB.id, Ops::K_MUL.id, Ops::K_RESHAPE.id,
                                 Ops::K_SUM_ROWS.id, Ops::K_SCALE.id}));
    EXPECT_EQ(graph.value(loss).shape.num_elements(), 1);

    const Graph::Node& square = *emitted_nodes(graph).at(1);
    EXPECT_EQ(square.inputs.at(0), square.inputs.at(1));
    EXPECT_EQ(emitted_nodes(graph).at(0)->inputs,
              (std::vector<Graph::ValueID>{outputs, targets}));
}

/** @brief Tests that two values of different shapes stop the process, naming both. */
TEST(Small_DenseLossesDeathTest, OperandsOfTwoShapesAreFatal)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID outputs = builder.input(Shape{4, 2}, "outputs");
    const Graph::ValueID targets = builder.input(Shape{4, 3}, "targets");

    SA_EXPECT_FATAL((void)Train::emit_squared_error(builder, graph, outputs, targets),
                    "[4, 2]", "[4, 3]");
}
