/**************************************************************************/
/* test_rematerialise.cpp                                                 */
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
 * @file test_rematerialise.cpp
 * @brief Tests the structure of the graph that the IM2COL rematerialisation pass rewrites.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/rematerialise.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::ValueId;

namespace
{
    /** @brief The nodes of @p graph reading @p value, in graph order. */
    std::vector<Graph::NodeId> readers_of(const Graph::Graph& graph, ValueId value)
    {
        std::vector<Graph::NodeId> readers;
        for (std::size_t i = 0; i < graph.node_count(); ++i)
            for (ValueId in : graph.nodes()[i].inputs)
                if (in == value) { readers.push_back(static_cast<Graph::NodeId>(i)); break; }
        return readers;
    }

    /** @brief How many IM2COL nodes @p graph holds. */
    std::size_t im2col_count(const Graph::Graph& graph)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
            if (node.op == Ops::K_IM2COL.id) ++count;
        return count;
    }

    /**
     * @brief A convolution as a GEMM, carried on to a scalar NLL_LOSS root.
     *
     * The loss tail is what makes the graph differentiable at all; it is not
     * part of what is being tested.
     */
    struct ConvGraph
    {
        Graph::Graph graph;
        ValueId cols = Graph::K_NO_VALUE;
        ValueId weight = Graph::K_NO_VALUE;
        ValueId loss = Graph::K_NO_VALUE;
    };

    /** @brief Builds and differentiates a one-convolution graph. */
    void build_and_differentiate(ConvGraph& c, Shape image, Shape weight_shape)
    {
        const GraphBuilder builder(c.graph);
        const ValueId x = builder.input(image, "x");
        c.weight = builder.parameter(weight_shape, "w");

        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        c.cols = builder.im2col(x, g);
        const ValueId out = builder.matmul(c.cols, c.weight);

        const Shape logits = c.graph.value(out).shape;
        const ValueId target = builder.input(logits, "target");
        c.loss = builder.nll_loss(builder.log_softmax(out), target);
        c.graph.finalise();

        const std::array<ValueId, 1> wrt = {c.weight};
        (void)Autograd::differentiate(c.graph, c.loss,
            SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));
    }
}

TEST(Unit_Rematerialise, TheColumnBufferStopsBeingAnActivation)
{
    // The column buffer's lifetime drops to SCRATCH; see tests/unit/README.md.
    ConvGraph c;
    build_and_differentiate(c, Shape{2, 5, 5, 3}, Shape{27, 8});

    EXPECT_EQ(c.graph.values()[c.cols].lifetime, Graph::Lifetime::Activation);
    ASSERT_GT(readers_of(c.graph, c.cols).size(), std::size_t{1})
        << "the backward pass did not read the column matrix, so there is "
           "nothing for the pass to do and this test proves nothing";

    Graph::rematerialise_im2col(c.graph);

    EXPECT_EQ(c.graph.values()[c.cols].lifetime, Graph::Lifetime::Scratch);

    // And it really did die early: exactly one node still reads the original.
    EXPECT_EQ(readers_of(c.graph, c.cols).size(), std::size_t{1});
}

TEST(Unit_Rematerialise, TheCloneComputesTheSameThingAsTheOriginal)
{
    // The clone reads the same operands and attributes; see tests/unit/README.md.
    ConvGraph c;
    build_and_differentiate(c, Shape{1, 4, 4, 2}, Shape{18, 4});

    const std::size_t before = c.graph.node_count();
    ASSERT_EQ(im2col_count(c.graph), std::size_t{1});

    Graph::rematerialise_im2col(c.graph);

    EXPECT_GT(c.graph.node_count(), before) << "no clone was inserted";
    EXPECT_EQ(c.graph.node_count(), before + 1) << "one clone, shared by the later consumers";

    const Graph::NodeId original_id = c.graph.value(c.cols).producer;
    const Graph::Node& original = c.graph.node(original_id);

    std::size_t clones = 0;
    for (std::size_t i = 0; i < c.graph.node_count(); ++i)
    {
        const Graph::Node& n = c.graph.nodes()[i];
        if (n.op != Ops::K_IM2COL.id || i == original_id) continue;
        ++clones;
        EXPECT_EQ(n.inputs, original.inputs);
        for (std::size_t slot = 0; slot < Graph::Attributes::K_CAPACITY; ++slot)
            EXPECT_EQ(n.attributes.get(slot), original.attributes.get(slot))
                << "the clone lost attribute slot " << slot;

        // The clone keeps the description and the layout; see tests/unit/README.md.
        const Graph::Value& copy = c.graph.value(n.outputs.front());
        const Graph::Value& stored = c.graph.value(c.cols);
        EXPECT_EQ(copy.shape, stored.shape);
        EXPECT_EQ(copy.dtype, stored.dtype);
        EXPECT_EQ(copy.layout, stored.layout);
        EXPECT_EQ(copy.lifetime, Graph::Lifetime::Scratch);
    }
    EXPECT_EQ(clones, std::size_t{1});
}

TEST(Unit_Rematerialise, TheCloneIsInsertedAheadOfEveryConsumerItServes)
{
    // A clone precedes every consumer it serves; see tests/unit/README.md.
    ConvGraph c;
    build_and_differentiate(c, Shape{1, 4, 4, 2}, Shape{18, 4});
    Graph::rematerialise_im2col(c.graph);

    for (std::size_t i = 0; i < c.graph.node_count(); ++i)
        for (ValueId in : c.graph.nodes()[i].inputs)
        {
            const Graph::NodeId producer = c.graph.value(in).producer;
            if (producer == Graph::K_NO_NODE) continue;
            EXPECT_LT(producer, i) << "node " << i << " reads value " << in
                                   << ", which node " << producer << " writes";
        }

    // The invariant, asserted by the component that depends on it.
    EXPECT_NO_THROW((void)Graph::MemoryPlan::build(c.graph));

    // And the graph's own proofs still hold after the rewrite.
    EXPECT_NO_THROW(c.graph.finalise());
}

TEST(Unit_Rematerialise, AnIm2colWithNoBackwardConsumerIsLeftAlone)
{
    // An inference graph has no second consumer; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{1, 4, 4, 2}, "x");

    Graph::ConvGeometry g;
    g.kernel_h = 3; g.kernel_w = 3;

    const ValueId cols = builder.im2col(x, g);
    graph.finalise();

    const std::size_t before = graph.node_count();
    Graph::rematerialise_im2col(graph);

    EXPECT_EQ(graph.node_count(), before);
    EXPECT_EQ(graph.values()[cols].lifetime, Graph::Lifetime::Activation);
}

TEST(Unit_Rematerialise, AForwardOnlyConvolutionIsLeftAlone)
{
    // A column matrix with one reader is not cloned; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueId x = builder.input(Shape{1, 4, 4, 2}, "x");
    const ValueId w = builder.input(Shape{18, 4}, "w");

    Graph::ConvGeometry g;
    g.kernel_h = 3; g.kernel_w = 3;
    g.pad_top = 1;  g.pad_left = 1;

    const ValueId cols = builder.im2col(x, g);
    (void)builder.matmul(cols, w);
    graph.finalise();

    const std::size_t before = graph.node_count();
    Graph::rematerialise_im2col(graph);

    EXPECT_EQ(graph.node_count(), before);
    EXPECT_EQ(graph.values()[cols].lifetime, Graph::Lifetime::Activation);
}

TEST(Unit_Rematerialise, RunningItTwiceAddsNothingTheSecondTime)
{
    // The clone has one consumer, so a second run adds none; see tests/unit/README.md.
    ConvGraph c;
    build_and_differentiate(c, Shape{1, 4, 4, 2}, Shape{18, 4});

    Graph::rematerialise_im2col(c.graph);
    const std::size_t once = c.graph.node_count();
    Graph::rematerialise_im2col(c.graph);

    EXPECT_EQ(c.graph.node_count(), once);
}
