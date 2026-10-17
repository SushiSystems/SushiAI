/**************************************************************************/
/* test_layer_node.cpp                                                    */
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
 * @file test_layer_node.cpp
 * @brief Tests what LayerNode adds to the Layer it owns: ownership, appending and its one form.
 * @author Mustafa Garip
 */

#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layer.hpp>
#include <SushiAI/nn/layer_node.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using NN::LayerAdapter;
using NN::LayerNode;
using NN::Linear;
using NN::Trace;

namespace
{
    /** @brief Views @p ids as the inputs of one forward call. */
    NN::ValueSpan span_of(const std::vector<Graph::ValueID>& ids)
    {
        return NN::ValueSpan(ids.data(), ids.size());
    }

    /** @brief Returns a 6-to-8 Linear wrapped as a run-time layer. */
    std::unique_ptr<NN::Layer> make_linear()
    {
        return std::make_unique<LayerAdapter<Linear>>(Linear{6, 8}, 8);
    }
} // namespace

/** @brief Tests that the node hands back the very layer it took ownership of. */
TEST(Small_LayerNode, TheNodeHandsBackTheLayerItOwns)
{
    std::unique_ptr<NN::Layer> layer = make_linear();
    const NN::Layer* const address = layer.get();

    LayerNode node(std::move(layer));
    const LayerNode& view = node;

    EXPECT_EQ(&node.layer(), address);
    EXPECT_EQ(&view.layer(), address);
    EXPECT_EQ(node.ports()[0].name, NN::K_SINGLE_PORT_NAME);
}

/** @brief Tests that forward appends its result behind the values the caller already held. */
TEST(Small_LayerNode, ForwardKeepsWhatTheOutputsAlreadyHeld)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 6}, "x");

    LayerNode node(make_linear());
    std::vector<Graph::ValueID> outputs{x};

    SA_EXPECT_OK(node.forward(Trace::evaluation(builder), span_of({x}), outputs));

    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0], x);
    EXPECT_EQ(graph.value(outputs[1]).shape, (Shape{2, 8}));
}

/** @brief Tests that a run-time layer has no incremental form and the node says so. */
TEST(Small_LayerNode, TheNodeRefusesIncrementalTracing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 6}, "x");

    LayerNode node(make_linear());
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_FAILURE(node.forward_incremental(Trace::evaluation(builder), span_of({x}), outputs),
                      StatusCode::Unsupported, "no incremental form");
    EXPECT_TRUE(outputs.empty());
}
