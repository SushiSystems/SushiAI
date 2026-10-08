/**************************************************************************/
/* test_scope_partitioner.cpp                                             */
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
 * @file test_scope_partitioner.cpp
 * @brief Tests the region keys ScopePartitioner gives the nodes of a differentiated graph.
 * @author Mustafa Garip
 * @see include/SushiAI/eager/README.md
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/scope.hpp>
#include <SushiAI/eager/scope_partitioner.hpp>
#include <SushiAI/eager/scope_trace.hpp>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include "../../common/test_throws.hpp"

using namespace SushiAI;
using Eager::Direction;
using Eager::K_NO_REGION;
using Eager::RegionKey;
using Graph::GraphBuilder;
using Graph::NodeId;
using Graph::ValueId;

namespace
{
    /** @brief Holds one traced, differentiated and partitioned training step. */
    struct Step
    {
        Graph::Graph graph;                   /**< The forward and backward nodes. */
        std::vector<Eager::Segment> segments; /**< The forward trace's segments. */
        Autograd::Provenance provenance;      /**< The origin of every backward node. */
        std::vector<ValueId> parameters;      /**< The values differentiated against. */
        std::vector<ValueId> gradients;       /**< One gradient per parameter. */
        Eager::Partition partition;           /**< What ScopePartitioner assigned. */
    };

    /** @brief Returns @p segments as the span assign() takes. */
    SushiRuntime::span<const Eager::Segment> as_span(const std::vector<Eager::Segment>& segments)
    {
        return SushiRuntime::span<const Eager::Segment>(segments.data(), segments.size());
    }

    /**
     * @brief Seals @p trace, differentiates @p loss and partitions the result into @p step.
     * @pre step.parameters lists the values to differentiate against.
     */
    void finish(Step& step, Eager::ScopeTrace& trace, ValueId loss)
    {
        step.segments = trace.seal();
        step.gradients = Autograd::differentiate(
            step.graph, loss,
            SushiRuntime::span<const ValueId>(step.parameters.data(), step.parameters.size()),
            Graph::K_NO_VALUE, &step.provenance);
        step.partition = Eager::ScopePartitioner{}.assign(step.graph, as_span(step.segments),
                                                          step.provenance);
    }

    /**
     * @brief Emits a linear layer with new parameters and appends them to @p parameters.
     * @return The layer's output, before any activation.
     */
    ValueId linear(const GraphBuilder& builder, ValueId x, int64_t inputs, int64_t outputs,
                   std::vector<ValueId>& parameters)
    {
        const ValueId weight = builder.parameter(Shape{inputs, outputs}, "weight");
        const ValueId bias = builder.parameter(Shape{outputs}, "bias");
        parameters.push_back(weight);
        parameters.push_back(bias);
        return builder.bias_add(builder.matmul(x, weight), bias);
    }

    /** @brief Builds two linear layers and a loss with no scope open. */
    Step unscoped_step()
    {
        Step step;
        const GraphBuilder builder(step.graph);
        Eager::ScopeTrace trace(step.graph);

        const ValueId x = builder.input(Shape{8, 6}, "x");
        const ValueId targets = builder.input(Shape{8, 3}, "targets");
        const ValueId hidden = builder.relu(linear(builder, x, 6, 4, step.parameters));
        const ValueId logits = linear(builder, hidden, 4, 3, step.parameters);

        finish(step, trace, builder.nll_loss(logits, targets));
        return step;
    }

    /**
     * @brief Builds scopes "first" and "second", one linear layer each, and a scope "loss".
     * @return A step of three segments whose parameters are weight, bias, weight, bias.
     */
    Step three_segment_step()
    {
        Step step;
        const GraphBuilder builder(step.graph);
        Eager::ScopeTrace trace(step.graph);

        ValueId hidden = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "first");
            const ValueId x = builder.input(Shape{8, 6}, "x");
            hidden = builder.relu(linear(builder, x, 6, 4, step.parameters));
        }

        ValueId logits = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "second");
            logits = builder.tanh(linear(builder, hidden, 4, 3, step.parameters));
        }

        ValueId loss = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "loss");
            const ValueId targets = builder.input(Shape{8, 3}, "targets");
            loss = builder.nll_loss(logits, targets);
        }

        finish(step, trace, loss);
        return step;
    }

    /**
     * @brief Builds a scope "weights" that declares one weight, scopes "first" and "second" that
     *        multiply by it, and a scope "loss".
     * @return A step of four segments, the first of them leaf-only, whose only parameter is the
     *         shared weight.
     */
    Step shared_weight_step()
    {
        Step step;
        const GraphBuilder builder(step.graph);
        Eager::ScopeTrace trace(step.graph);

        ValueId weight = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "weights");
            weight = builder.parameter(Shape{4, 4}, "weight");
        }

        ValueId hidden = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "first");
            const ValueId x = builder.input(Shape{8, 4}, "x");
            hidden = builder.relu(builder.matmul(x, weight));
        }

        ValueId logits = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "second");
            logits = builder.tanh(builder.matmul(hidden, weight));
        }

        ValueId loss = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "loss");
            const ValueId targets = builder.input(Shape{8, 4}, "targets");
            loss = builder.nll_loss(logits, targets);
        }

        step.parameters.push_back(weight);
        finish(step, trace, loss);
        return step;
    }

    /**
     * @brief Builds a scope "block" opened twice, one linear layer each time, and a scope "loss".
     * @return A step of three segments named "block", "block" and "loss".
     */
    Step repeated_block_step()
    {
        Step step;
        const GraphBuilder builder(step.graph);
        Eager::ScopeTrace trace(step.graph);

        ValueId hidden = builder.input(Shape{8, 4}, "x");
        for (int pass = 0; pass < 2; ++pass)
        {
            const Eager::Scope scope(trace, "block");
            hidden = builder.relu(linear(builder, hidden, 4, 4, step.parameters));
        }

        ValueId loss = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "loss");
            const ValueId targets = builder.input(Shape{8, 4}, "targets");
            loss = builder.nll_loss(hidden, targets);
        }

        finish(step, trace, loss);
        return step;
    }

    /** @brief Holds a hand-built graph of two one-RELU segments and its forward node count. */
    struct TwoReluGraph
    {
        Graph::Graph graph;                   /**< A parameter, then one RELU per segment. */
        std::vector<Eager::Segment> segments; /**< "first" and "second". */
        Autograd::Provenance provenance;      /**< Forward count set, origin left empty. */
        ValueId first = Graph::K_NO_VALUE;      /**< Output of the first segment's RELU. */
        ValueId second = Graph::K_NO_VALUE;     /**< Output of the second segment's RELU. */
    };

    /**
     * @brief Fills @p two with a parameter and two chained RELU nodes, one per segment.
     * @param two A default-constructed object; backward nodes are appended by the caller.
     */
    void build_two_relu_graph(TwoReluGraph& two)
    {
        const GraphBuilder builder(two.graph);
        Eager::ScopeTrace trace(two.graph);

        {
            const Eager::Scope scope(trace, "first");
            two.first = builder.relu(builder.parameter(Shape{4, 3}, "weight"));
        }
        {
            const Eager::Scope scope(trace, "second");
            two.second = builder.relu(two.first);
        }

        two.segments = trace.seal();
        two.provenance.forward_node_count = two.graph.node_count();
    }

    /** @brief Returns whether @p node of @p graph is a LEAF. */
    bool is_leaf(const Graph::Graph& graph, std::size_t node)
    {
        return graph.nodes()[node].op == Ops::K_LEAF.id;
    }

    /**
     * @brief Returns the number, counted from 1, of the segment that holds forward @p node.
     * @return 0 when no segment holds it.
     */
    RegionKey segment_number(const std::vector<Eager::Segment>& segments, std::size_t node)
    {
        for (std::size_t i = 0; i < segments.size(); ++i)
            if (node >= segments[i].begin && node < segments[i].end)
                return static_cast<RegionKey>(i + 1);
        return 0;
    }

    /** @brief Returns the keys of the backward nodes whose origin is a non-leaf of @p segment. */
    std::vector<RegionKey> backward_keys_of(const Step& step, RegionKey segment)
    {
        std::vector<RegionKey> keys;
        for (std::size_t i = 0; i < step.provenance.origin.size(); ++i)
        {
            const NodeId origin = step.provenance.origin[i];
            if (!is_leaf(step.graph, origin) && segment_number(step.segments, origin) == segment)
                keys.push_back(
                    step.partition.region_of_node[step.provenance.forward_node_count + i]);
        }
        return keys;
    }
} // namespace

/** @brief Tests that a step traced with no scope is one region with the optimizer key above it. */
TEST(Unit_ScopePartitioner, AnUnscopedStepIsOneRegion)
{
    const Step step = unscoped_step();

    ASSERT_EQ(step.segments.size(), 1u);
    ASSERT_EQ(step.partition.region_of_node.size(), step.graph.node_count());
    ASSERT_GT(step.graph.node_count(), step.provenance.forward_node_count);

    for (std::size_t node = 0; node < step.graph.node_count(); ++node)
        if (!is_leaf(step.graph, node))
            EXPECT_EQ(step.partition.region_of_node[node], 1u) << "node " << node;

    EXPECT_EQ(step.partition.optimizer_key, 2u);

    ASSERT_EQ(step.partition.regions.size(), 1u);
    EXPECT_EQ(step.partition.regions[0].key, 1u);
    EXPECT_EQ(step.partition.regions[0].name, "#0");
    EXPECT_EQ(step.partition.regions[0].direction, Direction::Forward);
}

/** @brief Tests that every forward node carries the number of its segment as its key. */
TEST(Unit_ScopePartitioner, ForwardKeysFollowTraceOrder)
{
    const Step step = three_segment_step();

    ASSERT_EQ(step.segments.size(), 3u);

    std::size_t checked = 0;
    for (std::size_t node = 0; node < step.provenance.forward_node_count; ++node)
    {
        if (is_leaf(step.graph, node))
            continue;

        EXPECT_EQ(step.partition.region_of_node[node], segment_number(step.segments, node))
            << "node " << node;
        ++checked;
    }

    // Two layers of three nodes each and the loss node.
    EXPECT_EQ(checked, 7u);
}

/** @brief Tests that three segments give their backward nodes keys 5, 4 and 3. */
TEST(Unit_ScopePartitioner, BackwardKeysMirrorForwardKeys)
{
    const Step step = three_segment_step();

    ASSERT_EQ(step.segments.size(), 3u);
    EXPECT_EQ(step.partition.optimizer_key, 6u);

    const RegionKey expected[] = {5u, 4u, 3u};
    for (RegionKey segment = 1; segment <= 3; ++segment)
    {
        const std::vector<RegionKey> keys = backward_keys_of(step, segment);

        ASSERT_FALSE(keys.empty()) << "segment " << segment << " has no backward node";
        for (RegionKey key : keys)
            EXPECT_EQ(key, expected[segment - 1]) << "segment " << segment;
    }

    ASSERT_EQ(step.partition.regions.size(), 5u);

    const std::string names[] = {"first#0", "second#0", "loss#0", "second#0", "first#0"};
    for (std::size_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(step.partition.regions[i].key, i + 1) << "region " << i;
        EXPECT_EQ(step.partition.regions[i].name, names[i]) << "region " << i;
        EXPECT_EQ(step.partition.regions[i].direction,
                  i < 3 ? Direction::Forward : Direction::Backward)
            << "region " << i;
    }
}

/** @brief Tests that the last segment's forward and backward nodes share one forward region. */
TEST(Unit_ScopePartitioner, TheLastSegmentSharesItsRegionWithItsBackward)
{
    const Step step = three_segment_step();

    ASSERT_EQ(step.segments.size(), 3u);

    const NodeId loss_node = static_cast<NodeId>(step.provenance.forward_node_count - 1);
    ASSERT_EQ(step.graph.node(loss_node).op, Ops::K_NLL_LOSS.id);
    EXPECT_EQ(step.partition.region_of_node[loss_node], 3u);

    const std::vector<RegionKey> keys = backward_keys_of(step, 3);
    ASSERT_FALSE(keys.empty());
    for (RegionKey key : keys)
        EXPECT_EQ(key, 3u);

    const auto shares_key = [](const Eager::RegionInfo& region)
    {
        return region.key == 3u;
    };
    const auto& regions = step.partition.regions;
    ASSERT_EQ(std::count_if(regions.begin(), regions.end(), shares_key), 1);
    EXPECT_EQ(std::find_if(regions.begin(), regions.end(), shares_key)->direction,
              Direction::Forward);
}

/** @brief Tests that no node of a differentiated graph reads a value a higher key produces. */
TEST(Unit_ScopePartitioner, EveryOperandComesFromALowerOrEqualKey)
{
    for (const Step& step : {unscoped_step(), three_segment_step(), shared_weight_step(),
                             repeated_block_step()})
    {
        ASSERT_EQ(step.partition.region_of_node.size(), step.graph.node_count());

        std::size_t checked = 0;
        for (std::size_t node = 0; node < step.graph.node_count(); ++node)
        {
            for (ValueId operand : step.graph.nodes()[node].inputs)
            {
                const NodeId producer = step.graph.value(operand).producer;
                if (producer == Graph::K_NO_NODE
                    || step.partition.region_of_node[producer] == K_NO_REGION)
                    continue;

                EXPECT_LE(step.partition.region_of_node[producer],
                          step.partition.region_of_node[node])
                    << "node " << node << " reads value " << operand << " of node " << producer;
                ++checked;
            }
        }

        EXPECT_GT(checked, 0u);
    }
}

/** @brief Tests that a weight gradient that reads an earlier segment stays in its own layer. */
TEST(Unit_ScopePartitioner, AWeightGradientLandsInItsOwnLayersBackwardRegion)
{
    const Step step = three_segment_step();

    ASSERT_EQ(step.gradients.size(), 4u);
    const ValueId second_weight_gradient = step.gradients[2];
    ASSERT_NE(second_weight_gradient, Graph::K_NO_VALUE);

    const NodeId node = step.graph.value(second_weight_gradient).producer;
    ASSERT_NE(node, Graph::K_NO_NODE);
    EXPECT_EQ(step.partition.region_of_node[node], 4u);

    bool reads_first_segment = false;
    for (ValueId operand : step.graph.node(node).inputs)
    {
        const NodeId producer = step.graph.value(operand).producer;
        if (producer != Graph::K_NO_NODE && step.partition.region_of_node[producer] == 1u)
            reads_first_segment = true;
    }
    EXPECT_TRUE(reads_first_segment)
        << "the gradient no longer reads the first segment's output, so this graph stopped "
           "exercising the case";
}

/** @brief Tests that the sum of a shared weight's gradients gets its highest contributor's key. */
TEST(Unit_ScopePartitioner, ASharedParametersAccumulateSitsAboveBothContributors)
{
    const Step step = shared_weight_step();

    ASSERT_EQ(step.gradients.size(), 1u);
    ASSERT_NE(step.gradients[0], Graph::K_NO_VALUE);

    const NodeId accumulate = step.graph.value(step.gradients[0]).producer;
    ASSERT_NE(accumulate, Graph::K_NO_NODE);
    ASSERT_EQ(step.graph.node(accumulate).op, Ops::K_ACCUMULATE.id);
    ASSERT_TRUE(is_leaf(step.graph, step.provenance.origin_of(accumulate)));

    std::vector<RegionKey> contributors;
    for (ValueId operand : step.graph.node(accumulate).inputs)
    {
        const NodeId producer = step.graph.value(operand).producer;
        ASSERT_NE(producer, Graph::K_NO_NODE);
        contributors.push_back(step.partition.region_of_node[producer]);
    }
    std::sort(contributors.begin(), contributors.end());

    // Of four segments, "second" has backward key 5, "first" 6 and the weight's own segment 7.
    EXPECT_EQ(contributors, (std::vector<RegionKey>{5u, 6u}));
    EXPECT_EQ(step.partition.region_of_node[accumulate], 6u);
}

/** @brief Tests that a LEAF-origin node gets its highest operand key in either operand order. */
TEST(Unit_ScopePartitioner, ALeafOriginNodeTakesItsHighestOperandKeyInEitherOrder)
{
    TwoReluGraph two;
    build_two_relu_graph(two);
    const GraphBuilder builder(two.graph);
    const NodeId leaf = 0;
    ASSERT_TRUE(is_leaf(two.graph, leaf));

    const ValueId later_first = builder.add(two.second, two.first);
    const ValueId earlier_first = builder.add(two.first, two.second);
    two.provenance.origin.assign(2, leaf);

    const Eager::Partition partition =
        Eager::ScopePartitioner{}.assign(two.graph, as_span(two.segments), two.provenance);

    EXPECT_EQ(partition.region_of_node[two.graph.value(later_first).producer], 2u);
    EXPECT_EQ(partition.region_of_node[two.graph.value(earlier_first).producer], 2u);
}

/** @brief Tests that a key which holds no node is left out of the region list. */
TEST(Unit_ScopePartitioner, RegionsListOnlyKeysThatHoldANode)
{
    const Step step = shared_weight_step();

    ASSERT_EQ(step.segments.size(), 4u);
    EXPECT_EQ(step.partition.optimizer_key, 8u);

    // Key 1 is the leaf-only segment "weights" and key 7 its backward mirror.
    const std::string names[] = {"first#0", "second#0", "loss#0", "second#0", "first#0"};
    ASSERT_EQ(step.partition.regions.size(), 5u);
    for (std::size_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(step.partition.regions[i].key, i + 2) << "region " << i;
        EXPECT_EQ(step.partition.regions[i].name, names[i]) << "region " << i;
        EXPECT_EQ(step.partition.regions[i].direction,
                  i < 3 ? Direction::Forward : Direction::Backward)
            << "region " << i;
    }
}

/** @brief Tests that a scope opened twice names its regions by occurrence, mirrored backward. */
TEST(Unit_ScopePartitioner, ARepeatedScopeIsNamedByItsOccurrence)
{
    const Step step = repeated_block_step();

    // The input declared before the first scope is a leaf-only segment with keys 1 and 7.
    ASSERT_EQ(step.segments.size(), 4u);
    EXPECT_EQ(step.partition.optimizer_key, 8u);

    const std::string names[] = {"block#0", "block#1", "loss#0", "block#1", "block#0"};
    ASSERT_EQ(step.partition.regions.size(), 5u);
    for (std::size_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(step.partition.regions[i].key, i + 2) << "region " << i;
        EXPECT_EQ(step.partition.regions[i].name, names[i]) << "region " << i;
        EXPECT_EQ(step.partition.regions[i].direction,
                  i < 3 ? Direction::Forward : Direction::Backward)
            << "region " << i;
    }
}

/** @brief Tests that a graph with no backward node gets forward keys and forward regions only. */
TEST(Unit_ScopePartitioner, AGraphWithNoBackwardNodeHasForwardRegionsOnly)
{
    TwoReluGraph two;
    build_two_relu_graph(two);

    const Eager::Partition partition =
        Eager::ScopePartitioner{}.assign(two.graph, as_span(two.segments), two.provenance);

    EXPECT_EQ(partition.region_of_node[two.graph.value(two.first).producer], 1u);
    EXPECT_EQ(partition.region_of_node[two.graph.value(two.second).producer], 2u);
    EXPECT_EQ(partition.optimizer_key, 4u);

    ASSERT_EQ(partition.regions.size(), 2u);
    EXPECT_EQ(partition.regions[0].key, 1u);
    EXPECT_EQ(partition.regions[0].name, "first#0");
    EXPECT_EQ(partition.regions[0].direction, Direction::Forward);
    EXPECT_EQ(partition.regions[1].key, 2u);
    EXPECT_EQ(partition.regions[1].name, "second#0");
    EXPECT_EQ(partition.regions[1].direction, Direction::Forward);
}

/** @brief Tests that LEAF nodes get no region and every other node gets one. */
TEST(Unit_ScopePartitioner, LeavesBelongToNoRegion)
{
    const Step step = three_segment_step();

    std::size_t leaves = 0;
    for (std::size_t node = 0; node < step.graph.node_count(); ++node)
    {
        if (is_leaf(step.graph, node))
        {
            EXPECT_EQ(step.partition.region_of_node[node], K_NO_REGION) << "node " << node;
            ++leaves;
        }
        else
        {
            EXPECT_NE(step.partition.region_of_node[node], K_NO_REGION) << "node " << node;
        }
    }

    // Two inputs and four parameters.
    EXPECT_EQ(leaves, 6u);
}

/** @brief Tests that segments which stop short of the forward graph are rejected. */
TEST(Unit_ScopePartitioner, SegmentsThatDoNotCoverTheForwardGraphThrow)
{
    const Step step = three_segment_step();
    const Eager::ScopePartitioner partitioner;

    std::vector<Eager::Segment> short_of_the_end = step.segments;
    short_of_the_end.pop_back();
    SA_EXPECT_ERROR((void)partitioner.assign(step.graph, as_span(short_of_the_end),
                                             step.provenance),
                    "The segments cover nodes 0 to", "forward graph has");

    std::vector<Eager::Segment> with_a_gap = step.segments;
    with_a_gap[1].begin += 1;
    SA_EXPECT_ERROR((void)partitioner.assign(step.graph, as_span(with_a_gap), step.provenance),
                    "Segment 1 ('second')", "contiguous");

    std::vector<Eager::Segment> with_an_empty_segment = step.segments;
    const NodeId boundary = step.segments[1].begin;
    with_an_empty_segment.insert(with_an_empty_segment.begin() + 1,
                                 Eager::Segment{"empty", 0, boundary, boundary});
    SA_EXPECT_ERROR((void)partitioner.assign(step.graph, as_span(with_an_empty_segment),
                                             step.provenance),
                    "Segment 1 ('empty')", "holds no node");

    const std::vector<Eager::Segment> none;
    SA_EXPECT_ERROR((void)partitioner.assign(step.graph, as_span(none), step.provenance),
                    "at least one segment");
}

/** @brief Tests that a provenance whose node counts differ from the graph's is rejected. */
TEST(Unit_ScopePartitioner, AProvenanceOfAnotherGraphThrows)
{
    const Step step = three_segment_step();
    const Eager::ScopePartitioner partitioner;

    Autograd::Provenance too_short = step.provenance;
    too_short.origin.pop_back();
    SA_EXPECT_ERROR((void)partitioner.assign(step.graph, as_span(step.segments), too_short),
                    "The provenance describes", "backward nodes");

    Autograd::Provenance out_of_range = step.provenance;
    out_of_range.origin.back() = static_cast<NodeId>(step.provenance.forward_node_count);
    SA_EXPECT_ERROR((void)partitioner.assign(step.graph, as_span(step.segments), out_of_range),
                    "has origin", "forward nodes");
}

/** @brief Tests that a backward node which reads a value of a higher key is rejected. */
TEST(Unit_ScopePartitioner, ABackwardNodeThatReadsAHigherKeyThrows)
{
    TwoReluGraph two;
    build_two_relu_graph(two);
    const GraphBuilder builder(two.graph);

    // Origins in forward order give the first backward node key 3 and its reader key 2.
    const ValueId of_first = builder.relu(two.first);
    (void)builder.relu(of_first);
    two.provenance.origin.push_back(two.graph.value(two.first).producer);
    two.provenance.origin.push_back(two.graph.value(two.second).producer);

    SA_EXPECT_ERROR((void)Eager::ScopePartitioner{}.assign(two.graph, as_span(two.segments),
                                                           two.provenance),
                    "has region key 2", "whose key is 3");
}

/** @brief Tests that a backward node with a LEAF origin that reads only leaves is rejected. */
TEST(Unit_ScopePartitioner, ALeafOriginNodeThatReadsOnlyLeavesThrows)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId left = builder.parameter(Shape{4, 3}, "left");
    const ValueId right = builder.parameter(Shape{4, 3}, "right");
    (void)builder.relu(left);

    Autograd::Provenance provenance;
    provenance.forward_node_count = graph.node_count();

    (void)builder.add(left, right);
    provenance.origin.push_back(graph.value(left).producer);

    const std::vector<Eager::Segment> segments = {
        Eager::Segment{"", 0, 0, static_cast<NodeId>(provenance.forward_node_count)}};

    SA_EXPECT_ERROR((void)Eager::ScopePartitioner{}.assign(graph, as_span(segments), provenance),
                    "as its origin", "reads only leaves");
}
