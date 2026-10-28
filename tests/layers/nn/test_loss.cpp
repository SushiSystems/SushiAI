/**************************************************************************/
/* test_loss.cpp                                                          */
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
 * @file test_loss.cpp
 * @brief Tests the nodes CrossEntropyLoss emits and the operands it refuses.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/loss.hpp>

using namespace SushiAI;
using Graph::ValueID;
using NN::CrossEntropyLoss;

/** @brief Tests that the loss is a log-softmax of the logits read by an NLL against the target. */
TEST(Small_CrossEntropyLoss, TheLossIsTheLikelihoodOfALogSoftmax)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const ValueID logits = builder.input(Shape{4, 3}, "logits");
    const ValueID one_hot = builder.input(Shape{4, 3}, "one_hot");
    const std::size_t nodes_before = graph.node_count();

    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            CrossEntropyLoss{}.forward(builder, logits, one_hot));

    ASSERT_EQ(graph.node_count(), nodes_before + 2u);
    const Graph::Node& softmax = graph.nodes()[nodes_before];
    const Graph::Node& likelihood = graph.nodes()[nodes_before + 1u];
    EXPECT_EQ(softmax.op, Ops::K_LOG_SOFTMAX.id);
    EXPECT_EQ(softmax.inputs, (std::vector<ValueID>{logits}));
    EXPECT_EQ(likelihood.op, Ops::K_NLL_LOSS.id);
    EXPECT_EQ(likelihood.inputs, (std::vector<ValueID>{softmax.outputs[0], one_hot}));
    EXPECT_EQ(likelihood.outputs, (std::vector<ValueID>{loss}));
}

/** @brief Tests that the loss holds one element of the logits' element type. */
TEST(Small_CrossEntropyLoss, TheLossIsOneElementOfTheLogitsType)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const ValueID logits = builder.input(Shape{4, 3}, "logits", DType::Float64);
    const ValueID one_hot = builder.input(Shape{4, 3}, "one_hot", DType::Float64);

    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            CrossEntropyLoss{}.forward(builder, logits, one_hot));

    EXPECT_EQ(graph.value(loss).shape.num_elements(), 1);
    EXPECT_EQ(graph.value(loss).dtype, DType::Float64);
}

/** @brief Tests that logits and a target of two shapes are refused with the node's name. */
TEST(Small_CrossEntropyLoss, OperandsOfTwoShapesAreRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const ValueID logits = builder.input(Shape{4, 3}, "logits");
    const ValueID one_hot = builder.input(Shape{4, 2}, "one_hot");

    SA_EXPECT_FAILURE(CrossEntropyLoss{}.forward(builder, logits, one_hot),
                      StatusCode::InvalidDescription, "NLL_LOSS", "[4, 3]", "[4, 2]");
}
