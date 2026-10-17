/**************************************************************************/
/* test_module_node.cpp                                                   */
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
 * @file test_module_node.cpp
 * @brief Tests the incremental form of ModuleNode and the layer it hands back.
 * @author Mustafa Garip
 */

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using NN::Linear;
using NN::ModuleNode;
using NN::Trace;

namespace
{
    /** @brief Traces a ReLU, or a GELU in its incremental form, and keeps the sample it saw. */
    struct IncrementalProbe
    {
        /** @brief Holds the per-sample shape the last incremental trace was given. */
        Shape sample{};

        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Traces a ReLU of @p x. */
        Graph::ValueID forward(const Trace& trace, Graph::ValueID x)
        {
            return trace.builder().relu(x);
        }

        /** @brief Traces a GELU of @p x and keeps @p seen. */
        Graph::ValueID forward_incremental(const Trace& trace, Graph::ValueID x,
                                           const Shape& seen)
        {
            sample = seen;
            return trace.builder().gelu(x);
        }

        /** @brief Returns @p input, which the probe keeps. */
        Shape output_shape(const Shape& input) const
        {
            return input;
        }
    };

    static_assert(NN::IncrementalModule<IncrementalProbe>,
                  "IncrementalProbe must have an incremental form");
    static_assert(!NN::IncrementalModule<Linear>, "Linear must have no incremental form");

    /** @brief Builds the node of @p layer over @p input and expects create() to succeed. */
    template<typename L>
    std::unique_ptr<ModuleNode<L>> node_of(L layer, const Shape& input)
    {
        Result<std::unique_ptr<ModuleNode<L>>> made =
            ModuleNode<L>::create(std::move(layer), input);
        EXPECT_TRUE(Test::status_is_ok(made.status()));
        return std::move(made).value();
    }

    /** @brief Views @p ids as the inputs of one forward call. */
    NN::ValueSpan span_of(const std::vector<Graph::ValueID>& ids)
    {
        return NN::ValueSpan(ids.data(), ids.size());
    }
} // namespace

/** @brief Tests that an incremental trace hands the layer the shape create() was given. */
TEST(Small_ModuleNode, AnIncrementalTraceGivesTheLayerTheShapeOfCreate)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID slice = builder.input(Shape{2, 1, 4}, "slice");

    const std::unique_ptr<ModuleNode<IncrementalProbe>> node =
        node_of(IncrementalProbe{}, Shape{5, 4});
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_OK(node->forward_incremental(Trace::evaluation(builder), span_of({slice}),
                                           outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(node->layer().sample, (Shape{5, 4}));
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{2, 1, 4}));
    EXPECT_EQ(graph.nodes().back().op, Ops::K_GELU.id);
}

/** @brief Tests that an incremental trace given two values fails and appends nothing. */
TEST(Small_ModuleNode, AnIncrementalTraceRejectsTwoInputs)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID slice = builder.input(Shape{2, 1, 4}, "slice");

    const std::unique_ptr<ModuleNode<IncrementalProbe>> node =
        node_of(IncrementalProbe{}, Shape{5, 4});
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_FAILURE(node->forward_incremental(Trace::evaluation(builder),
                                                span_of({slice, slice}), outputs),
                      StatusCode::InvalidDescription, "input count is 1", "given is 2");
    EXPECT_TRUE(outputs.empty());
}

/** @brief Tests that a layer without an incremental form is refused one slice at a time. */
TEST(Small_ModuleNode, ALayerWithoutAnIncrementalFormIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 6}, "x");

    const std::unique_ptr<ModuleNode<Linear>> node = node_of(Linear{6, 8}, Shape{6});
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_FAILURE(node->forward_incremental(Trace::evaluation(builder), span_of({x}),
                                                outputs),
                      StatusCode::Unsupported, "no incremental form");
    EXPECT_TRUE(outputs.empty());
}

/** @brief Tests that the node hands back the layer it wraps, fields intact. */
TEST(Small_ModuleNode, TheNodeHandsBackTheLayerItWraps)
{
    const std::unique_ptr<ModuleNode<Linear>> node = node_of(Linear{6, 8}, Shape{6});
    const ModuleNode<Linear>& view = *node;

    EXPECT_EQ(node->layer().in_features, 6);
    EXPECT_EQ(view.layer().out_features, 8);
    EXPECT_EQ(&view.layer(), &node->layer());
}
