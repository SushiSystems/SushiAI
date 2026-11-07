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
 * @file test_scope_partitioner.cpp
 * @brief Tests the region keys ScopePartitioner gives the nodes of a differentiated graph.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/scope.hpp>
#include <SushiAI/eager/scope_partitioner.hpp>
#include <SushiAI/eager/scope_trace.hpp>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Eager::Direction;
using Eager::K_NO_REGION;
using Eager::RegionKey;
using Graph::GraphBuilder;
using Graph::NodeID;
using Graph::ValueID;

namespace
{
    /** @brief Holds one traced, differentiated and partitioned training step. */
    struct Step
    {
        Graph::Graph graph;                   /**< The forward and backward nodes. */
        std::vector<Eager::Segment> segments; /**< The forward trace's segments. */
        Autograd::Provenance provenance;      /**< The origin of every backward node. */
        std::vector<ValueID> parameters;      /**< The values differentiated against. */
        std::vector<ValueID> gradients;       /**< One gradient per parameter. */
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
    void finish(Step& step, Eager::ScopeTrace& trace, ValueID loss)
    {
        step.segments = trace.seal();
        Result<std::vector<ValueID>> gradients = Autograd::differentiate(
            step.graph, loss,
            SushiRuntime::span<const ValueID>(step.parameters.data(), step.parameters.size()),
            Graph::K_NO_VALUE, &step.provenance);
        // A refusal fails the calling test and leaves the step unpartitioned.
        if (!gradients.has_value())
        {
            ADD_FAILURE() << "differentiate() refused the step: " << gradients.status().message();
            return;
        }

        step.gradients = std::move(gradients).value();
        step.partition = Eager::ScopePartitioner{}.assign(step.graph, as_span(step.segments),
                                                          step.provenance);
    }

    /**
     * @brief Emits a linear layer with new parameters and appends them to @p parameters.
     * @return The layer's output, before any activation.
     */
    ValueID linear(const GraphBuilder& builder, ValueID x, int64_t inputs, int64_t outputs,
                   std::vector<ValueID>& parameters)
    {
        const ValueID weight = builder.parameter(Shape{inputs, outputs}, "weight");
        const ValueID bias = builder.parameter(Shape{outputs}, "bias");
        parameters.push_back(weight);
        parameters.push_back(bias);

        // The extents are fixed here, so a refusal is a defect: value() stops the test.
        return builder.bias_add(builder.matmul(x, weight), bias).value();
    }

    /** @brief Builds two linear layers and a loss with no scope open. */
    Step unscoped_step()
    {
        Step step;
        const GraphBuilder builder(step.graph);
        Eager::ScopeTrace trace(step.graph);

        const ValueID x = builder.input(Shape{8, 6}, "x");
        const ValueID targets = builder.input(Shape{8, 3}, "targets");
        // The extents are fixed here, so a refusal is a defect: value() stops the test.
        const ValueID hidden = builder.relu(linear(builder, x, 6, 4, step.parameters)).value();
        const ValueID logits = linear(builder, hidden, 4, 3, step.parameters);

        finish(step, trace, builder.nll_loss(logits, targets).value());
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

        ValueID hidden = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "first");
            const ValueID x = builder.input(Shape{8, 6}, "x");
            // The extents are fixed here, so a refusal is a defect: value() stops the test.
            hidden = builder.relu(linear(builder, x, 6, 4, step.parameters)).value();
        }

        ValueID logits = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "second");
            // The extents are fixed here, so a refusal is a defect: value() stops the test.
            logits = builder.tanh(linear(builder, hidden, 4, 3, step.parameters)).value();
        }

        ValueID loss = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "loss");
            const ValueID targets = builder.input(Shape{8, 3}, "targets");
            // The extents are fixed here, so a refusal is a defect: value() stops the test.
            loss = builder.nll_loss(logits, targets).value();
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

        ValueID weight = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "weights");
            weight = builder.parameter(Shape{4, 4}, "weight");
        }

        ValueID hidden = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "first");
            const ValueID x = builder.input(Shape{8, 4}, "x");
            // The extents are fixed here, so a refusal is a defect: value() stops the test.
            hidden = builder.relu(builder.matmul(x, weight)).value();
        }

        ValueID logits = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "second");
            // The extents are fixed here, so a refusal is a defect: value() stops the test.
            logits = builder.tanh(builder.matmul(hidden, weight)).value();
        }

        ValueID loss = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "loss");
            const ValueID targets = builder.input(Shape{8, 4}, "targets");
            // The extents are fixed here, so a refusal is a defect: value() stops the test.
            loss = builder.nll_loss(logits, targets).value();
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

        ValueID hidden = builder.input(Shape{8, 4}, "x");
        for (int pass = 0; pass < 2; ++pass)
        {
            const Eager::Scope scope(trace, "block");
            // The extents are fixed here, so a refusal is a defect: value() stops the test.
            hidden = builder.relu(linear(builder, hidden, 4, 4, step.parameters)).value();
        }

        ValueID loss = Graph::K_NO_VALUE;
        {
            const Eager::Scope scope(trace, "loss");
            const ValueID targets = builder.input(Shape{8, 4}, "targets");
            // The extents are fixed here, so a refusal is a defect: value() stops the test.
            loss = builder.nll_loss(hidden, targets).value();
        }

        finish(step, trace, loss);
        return step;
    }

    /** @brief Holds a hand-built graph of two one-RELU segments and its forward node count. */
    struct TwoReLUGraph
    {
        Graph::Graph graph;                   /**< A parameter, then one RELU per segment. */
        std::vector<Eager::Segment> segments; /**< "first" and "second". */
        Autograd::Provenance provenance;      /**< Forward count set, origin left empty. */
        ValueID first = Graph::K_NO_VALUE;      /**< Output of the first segment's RELU. */
        ValueID second = Graph::K_NO_VALUE;     /**< Output of the second segment's RELU. */
    };

    /**
     * @brief Fills @p two with a parameter and two chained RELU nodes, one per segment.
     * @param two A default-constructed object; backward nodes are appended by the caller.
     */
    void build_two_relu_graph(TwoReLUGraph& two)
    {
        const GraphBuilder builder(two.graph);
        Eager::ScopeTrace trace(two.graph);

        {
            const Eager::Scope scope(trace, "first");
            SA_ASSERT_OK_AND_ASSIGN(two.first,
                                    builder.relu(builder.parameter(Shape{4, 3}, "weight")));
        }
        {
            const Eager::Scope scope(trace, "second");
            SA_ASSERT_OK_AND_ASSIGN(two.second, builder.relu(two.first));
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
            const NodeID origin = step.provenance.origin[i];
            if (!is_leaf(step.graph, origin) && segment_number(step.segments, origin) == segment)
                keys.push_back(
                    step.partition.region_of_node[step.provenance.forward_node_count + i]);
        }
        return keys;
    }

    /** @brief Returns the region name ScopePartitioner gives @p segment, "name#occurrence". */
    std::string region_name_of(const Eager::Segment& segment)
    {
        return segment.name + "#" + std::to_string(segment.occurrence);
    }

    /** @brief Expects every node of @p step that is not a LEAF to hold region key @p key. */
    void expect_every_non_leaf_node_has_key(const Step& step, RegionKey key)
    {
        for (std::size_t node = 0; node < step.graph.node_count(); ++node)
            if (!is_leaf(step.graph, node))
                EXPECT_EQ(step.partition.region_of_node[node], key) << "node " << node;
    }

    /**
     * @brief Expects every forward node of @p step that is not a LEAF to hold its segment number.
     * @return The number of nodes compared.
     */
    std::size_t expect_forward_keys_follow_segments(const Step& step)
    {
        std::size_t checked = 0;
        for (std::size_t node = 0; node < step.provenance.forward_node_count; ++node)
        {
            if (is_leaf(step.graph, node))
                continue;

            EXPECT_EQ(step.partition.region_of_node[node], segment_number(step.segments, node))
                << "node " << node;
            ++checked;
        }
        return checked;
    }

    /**
     * @brief Expects the backward nodes of segment i, counted from 1, to hold key 2N - i for N
     *        segments, so the backward keys mirror the forward keys above the last segment.
     */
    void expect_backward_keys_mirror_segments(const Step& step)
    {
        const RegionKey count = step.segments.size();
        for (RegionKey segment = 1; segment <= count; ++segment)
        {
            const std::vector<RegionKey> keys = backward_keys_of(step, segment);

            ASSERT_FALSE(keys.empty()) << "segment " << segment << " has no backward node";
            for (RegionKey key : keys)
                EXPECT_EQ(key, 2 * count - segment) << "segment " << segment;
        }
    }

    /**
     * @brief Expects the regions of @p step to list its segments forward, then every segment but
     *        the last backward in reverse, the last sharing its forward region.
     */
    void expect_regions_mirror_segments(const Step& step)
    {
        const std::size_t count = step.segments.size();
        ASSERT_EQ(step.partition.regions.size(), 2 * count - 1);
        for (std::size_t i = 0; i < step.partition.regions.size(); ++i)
        {
            const bool forward = i < count;
            const Eager::Segment& segment = forward ? step.segments[i]
                                                    : step.segments[2 * count - 2 - i];
            EXPECT_EQ(step.partition.regions[i].key, i + 1) << "region " << i;
            EXPECT_EQ(step.partition.regions[i].name, region_name_of(segment)) << "region " << i;
            EXPECT_EQ(step.partition.regions[i].direction,
                      forward ? Direction::Forward : Direction::Backward)
                << "region " << i;
        }
    }

    /** @brief Expects region @p index of @p partition to hold @p key, @p name and @p direction. */
    void expect_region(const Eager::Partition& partition, std::size_t index, RegionKey key,
                       const std::string& name, Direction direction)
    {
        ASSERT_LT(index, partition.regions.size());
        EXPECT_EQ(partition.regions[index].key, key) << "region " << index;
        EXPECT_EQ(partition.regions[index].name, name) << "region " << index;
        EXPECT_EQ(partition.regions[index].direction, direction) << "region " << index;
    }

    /** @brief Expects every key of @p keys to equal @p key. */
    void expect_all_keys_equal(const std::vector<RegionKey>& keys, RegionKey key)
    {
        for (RegionKey held : keys)
            EXPECT_EQ(held, key);
    }

    /** @brief Returns how many regions of @p partition hold @p key. */
    std::size_t count_regions_with_key(const Eager::Partition& partition, RegionKey key)
    {
        std::size_t found = 0;
        for (const Eager::RegionInfo& region : partition.regions)
            found += region.key == key ? 1 : 0;
        return found;
    }

    /** @brief Returns the direction of the region of @p partition that holds @p key. */
    Direction direction_of_region_with_key(const Eager::Partition& partition, RegionKey key)
    {
        for (const Eager::RegionInfo& region : partition.regions)
            if (region.key == key)
                return region.direction;
        return Direction::Backward;
    }

    /** @brief Returns whether a node @p node of @p step reads a value that region @p key holds. */
    bool reads_from_region(const Step& step, NodeID node, RegionKey key)
    {
        for (ValueID operand : step.graph.node(node).inputs)
        {
            const NodeID producer = step.graph.value(operand).producer;
            if (producer != Graph::K_NO_NODE && step.partition.region_of_node[producer] == key)
                return true;
        }
        return false;
    }

    /**
     * @brief Expects no node of @p step to read a value that a higher region key produces.
     * @return The number of operands that another region produces.
     */
    std::size_t expect_operands_come_from_lower_or_equal_keys(const Step& step)
    {
        std::size_t checked = 0;
        for (std::size_t node = 0; node < step.graph.node_count(); ++node)
        {
            for (ValueID operand : step.graph.nodes()[node].inputs)
            {
                const NodeID producer = step.graph.value(operand).producer;
                if (producer == Graph::K_NO_NODE
                    || step.partition.region_of_node[producer] == K_NO_REGION)
                    continue;

                EXPECT_LE(step.partition.region_of_node[producer],
                          step.partition.region_of_node[node])
                    << "node " << node << " reads value " << operand << " of node " << producer;
                ++checked;
            }
        }
        return checked;
    }

    /** @brief Returns the sorted keys of the regions that produce the operands of @p node. */
    std::vector<RegionKey> contributor_keys(const Step& step, NodeID node)
    {
        std::vector<RegionKey> keys;
        for (ValueID operand : step.graph.node(node).inputs)
        {
            const NodeID producer = step.graph.value(operand).producer;
            EXPECT_NE(producer, Graph::K_NO_NODE);
            if (producer != Graph::K_NO_NODE)
                keys.push_back(step.partition.region_of_node[producer]);
        }
        std::sort(keys.begin(), keys.end());
        return keys;
    }

    /**
     * @brief Expects LEAF nodes of @p step to hold no region and every other node to hold one.
     * @return The number of LEAF nodes.
     */
    std::size_t expect_leaves_hold_no_region(const Step& step)
    {
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
        return leaves;
    }

    /** @brief Names the traced steps the operand-order rows run over. */
    enum class StepKind : std::uint8_t
    {
        Unscoped,      /**< Two layers and a loss with no scope. */
        ThreeSegments, /**< Scopes "first", "second" and "loss". */
        SharedWeight,  /**< One weight shared by two scopes. */
        RepeatedBlock, /**< A scope opened twice. */
    };

    /** @brief Traces the step @p kind names. */
    Step make_step(StepKind kind)
    {
        switch (kind)
        {
            case StepKind::Unscoped:
                return unscoped_step();
            case StepKind::ThreeSegments:
                return three_segment_step();
            case StepKind::SharedWeight:
                return shared_weight_step();
            case StepKind::RepeatedBlock:
                return repeated_block_step();
        }
        return unscoped_step();
    }
} // namespace

/** @brief Tests that a step traced with no scope is one region with the optimizer key above it. */
TEST(Small_ScopePartitioner, AnUnscopedStepIsOneRegion)
{
    const Step step = unscoped_step();

    ASSERT_EQ(step.segments.size(), 1u);
    ASSERT_EQ(step.partition.region_of_node.size(), step.graph.node_count());
    ASSERT_GT(step.graph.node_count(), step.provenance.forward_node_count);

    expect_every_non_leaf_node_has_key(step, 1u);

    EXPECT_EQ(step.partition.optimizer_key, 2u);

    ASSERT_EQ(step.partition.regions.size(), 1u);
    EXPECT_EQ(step.partition.regions[0].key, 1u);
    EXPECT_EQ(step.partition.regions[0].name, "#0");
    EXPECT_EQ(step.partition.regions[0].direction, Direction::Forward);
}

/** @brief Tests that every forward node carries the number of its segment as its key. */
TEST(Small_ScopePartitioner, ForwardKeysFollowTraceOrder)
{
    const Step step = three_segment_step();

    ASSERT_EQ(step.segments.size(), 3u);

    const std::size_t checked = expect_forward_keys_follow_segments(step);

    // Two layers of three nodes each and the loss node.
    EXPECT_EQ(checked, 7u);
}

/** @brief Tests that the backward keys of the segments mirror their forward keys. */
TEST(Small_ScopePartitioner, BackwardKeysMirrorForwardKeys)
{
    const Step step = three_segment_step();

    ASSERT_EQ(step.segments.size(), 3u);
    expect_backward_keys_mirror_segments(step);
}

/** @brief Tests that the regions list the segments forward, then backward in reverse order. */
TEST(Small_ScopePartitioner, RegionsMirrorTheSegments)
{
    const Step step = three_segment_step();

    expect_regions_mirror_segments(step);
}

/** @brief Tests that the last segment's forward and backward nodes share one forward region. */
TEST(Small_ScopePartitioner, TheLastSegmentSharesItsRegionWithItsBackward)
{
    const Step step = three_segment_step();

    ASSERT_EQ(step.segments.size(), 3u);

    const NodeID loss_node = static_cast<NodeID>(step.provenance.forward_node_count - 1);
    ASSERT_EQ(step.graph.node(loss_node).op, Ops::K_NLL_LOSS.id);
    EXPECT_EQ(step.partition.region_of_node[loss_node], 3u);

    const std::vector<RegionKey> keys = backward_keys_of(step, 3);
    ASSERT_FALSE(keys.empty());
    expect_all_keys_equal(keys, 3u);

    EXPECT_EQ(count_regions_with_key(step.partition, 3u), 1u);
    EXPECT_EQ(direction_of_region_with_key(step.partition, 3u), Direction::Forward);
}

class Small_ScopePartitionerOperands : public ::testing::TestWithParam<StepKind>
{
};

/** @brief Tests that no node of a differentiated graph reads a value a higher key produces. */
TEST_P(Small_ScopePartitionerOperands, EveryOperandComesFromALowerOrEqualKey)
{
    const Step step = make_step(GetParam());

    ASSERT_EQ(step.partition.region_of_node.size(), step.graph.node_count());
    EXPECT_GT(expect_operands_come_from_lower_or_equal_keys(step), 0u);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ScopePartitionerOperands,
                         ::testing::Values(StepKind::Unscoped, StepKind::ThreeSegments,
                                           StepKind::SharedWeight, StepKind::RepeatedBlock));

/** @brief Tests that a weight gradient that reads an earlier segment stays in its own layer. */
TEST(Small_ScopePartitioner, AWeightGradientLandsInItsOwnLayersBackwardRegion)
{
    const Step step = three_segment_step();

    ASSERT_EQ(step.gradients.size(), 4u);
    const ValueID second_weight_gradient = step.gradients[2];
    ASSERT_NE(second_weight_gradient, Graph::K_NO_VALUE);

    const NodeID node = step.graph.value(second_weight_gradient).producer;
    ASSERT_NE(node, Graph::K_NO_NODE);
    EXPECT_EQ(step.partition.region_of_node[node], 4u);

    EXPECT_TRUE(reads_from_region(step, node, 1u))
        << "the gradient no longer reads the first segment's output, so this graph stopped "
           "exercising the case";
}

/** @brief Tests that the sum of a shared weight's gradients gets its highest contributor's key. */
TEST(Small_ScopePartitioner, ASharedParametersAccumulateSitsAboveBothContributors)
{
    const Step step = shared_weight_step();

    ASSERT_EQ(step.gradients.size(), 1u);
    ASSERT_NE(step.gradients[0], Graph::K_NO_VALUE);

    const NodeID accumulate = step.graph.value(step.gradients[0]).producer;
    ASSERT_NE(accumulate, Graph::K_NO_NODE);
    ASSERT_EQ(step.graph.node(accumulate).op, Ops::K_ACCUMULATE.id);
    ASSERT_TRUE(is_leaf(step.graph, step.provenance.origin_of(accumulate)));

    const std::vector<RegionKey> contributors = contributor_keys(step, accumulate);

    // The sum takes the key of its highest contributor, which is above the other's.
    ASSERT_EQ(contributors.size(), 2u);
    EXPECT_LT(contributors[0], contributors[1]);
    EXPECT_EQ(step.partition.region_of_node[accumulate], contributors.back());
}

/** @brief Tests that a LEAF-origin node gets its highest operand key in either operand order. */
TEST(Small_ScopePartitioner, ALeafOriginNodeTakesItsHighestOperandKeyInEitherOrder)
{
    TwoReLUGraph two;
    build_two_relu_graph(two);
    const GraphBuilder builder(two.graph);
    const NodeID leaf = 0;
    ASSERT_TRUE(is_leaf(two.graph, leaf));

    SA_ASSERT_OK_AND_ASSIGN(const ValueID later_first, builder.add(two.second, two.first));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID earlier_first, builder.add(two.first, two.second));
    two.provenance.origin.assign(2, leaf);

    const Eager::Partition partition =
        Eager::ScopePartitioner{}.assign(two.graph, as_span(two.segments), two.provenance);

    EXPECT_EQ(partition.region_of_node[two.graph.value(later_first).producer], 2u);
    EXPECT_EQ(partition.region_of_node[two.graph.value(earlier_first).producer], 2u);
}

/** @brief Tests that a key which holds no node is left out of the region list. */
TEST(Small_ScopePartitioner, RegionsListOnlyKeysThatHoldANode)
{
    const Step step = shared_weight_step();

    ASSERT_EQ(step.segments.size(), 4u);
    EXPECT_EQ(step.partition.optimizer_key, 8u);

    // Key 1 is the leaf-only segment "weights" and key 7 its backward mirror.
    ASSERT_EQ(step.partition.regions.size(), 5u);
    expect_region(step.partition, 0, 2u, "first#0", Direction::Forward);
    expect_region(step.partition, 1, 3u, "second#0", Direction::Forward);
    expect_region(step.partition, 2, 4u, "loss#0", Direction::Forward);
    expect_region(step.partition, 3, 5u, "second#0", Direction::Backward);
    expect_region(step.partition, 4, 6u, "first#0", Direction::Backward);
}

/** @brief Tests that a scope opened twice names its regions by occurrence, mirrored backward. */
TEST(Small_ScopePartitioner, ARepeatedScopeIsNamedByItsOccurrence)
{
    const Step step = repeated_block_step();

    // The input declared before the first scope is a leaf-only segment with keys 1 and 7.
    ASSERT_EQ(step.segments.size(), 4u);
    EXPECT_EQ(step.partition.optimizer_key, 8u);

    ASSERT_EQ(step.partition.regions.size(), 5u);
    expect_region(step.partition, 0, 2u, "block#0", Direction::Forward);
    expect_region(step.partition, 1, 3u, "block#1", Direction::Forward);
    expect_region(step.partition, 2, 4u, "loss#0", Direction::Forward);
    expect_region(step.partition, 3, 5u, "block#1", Direction::Backward);
    expect_region(step.partition, 4, 6u, "block#0", Direction::Backward);
}

/** @brief Tests that a graph with no backward node gets forward keys and forward regions only. */
TEST(Small_ScopePartitioner, AGraphWithNoBackwardNodeHasForwardRegionsOnly)
{
    TwoReLUGraph two;
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
TEST(Small_ScopePartitioner, LeavesBelongToNoRegion)
{
    const Step step = three_segment_step();

    const std::size_t leaves = expect_leaves_hold_no_region(step);

    // Two inputs and four parameters.
    EXPECT_EQ(leaves, 6u);
}

/** @brief Tests that segments which stop short of the forward graph stop the process. */
TEST(Medium_ScopePartitionerFatal, SegmentsThatStopShortOfTheForwardGraphAreFatal)
{
    const Step step = three_segment_step();
    const Eager::ScopePartitioner partitioner;

    std::vector<Eager::Segment> short_of_the_end = step.segments;
    short_of_the_end.pop_back();

    SA_EXPECT_FATAL((void)partitioner.assign(step.graph, as_span(short_of_the_end),
                                             step.provenance),
                    "The segments cover nodes 0 to", "forward graph has");
}

/** @brief Tests that a gap between two segments stops the process. */
TEST(Medium_ScopePartitionerFatal, ASegmentThatLeavesAGapIsFatal)
{
    const Step step = three_segment_step();
    const Eager::ScopePartitioner partitioner;

    std::vector<Eager::Segment> with_a_gap = step.segments;
    with_a_gap[1].begin += 1;

    SA_EXPECT_FATAL((void)partitioner.assign(step.graph, as_span(with_a_gap), step.provenance),
                    "Segment 1 ('second')", "contiguous");
}

/** @brief Tests that a segment holding no node stops the process. */
TEST(Medium_ScopePartitionerFatal, AnEmptySegmentIsFatal)
{
    const Step step = three_segment_step();
    const Eager::ScopePartitioner partitioner;

    std::vector<Eager::Segment> with_an_empty_segment = step.segments;
    const NodeID boundary = step.segments[1].begin;
    with_an_empty_segment.insert(with_an_empty_segment.begin() + 1,
                                 Eager::Segment{"empty", 0, boundary, boundary});

    SA_EXPECT_FATAL((void)partitioner.assign(step.graph, as_span(with_an_empty_segment),
                                             step.provenance),
                    "Segment 1 ('empty')", "holds no node");
}

/** @brief Tests that an empty list of segments stops the process. */
TEST(Medium_ScopePartitionerFatal, NoSegmentAtAllIsFatal)
{
    const Step step = three_segment_step();
    const Eager::ScopePartitioner partitioner;
    const std::vector<Eager::Segment> none;

    SA_EXPECT_FATAL((void)partitioner.assign(step.graph, as_span(none), step.provenance),
                    "at least one segment");
}

/** @brief Tests that a provenance with too few origins stops the process. */
TEST(Medium_ScopePartitionerFatal, AProvenanceWithTooFewOriginsIsFatal)
{
    const Step step = three_segment_step();
    const Eager::ScopePartitioner partitioner;

    Autograd::Provenance too_short = step.provenance;
    too_short.origin.pop_back();

    SA_EXPECT_FATAL((void)partitioner.assign(step.graph, as_span(step.segments), too_short),
                    "The provenance describes", "backward nodes");
}

/** @brief Tests that a provenance with an origin outside the forward range stops the process. */
TEST(Medium_ScopePartitionerFatal, AnOriginOutsideTheForwardRangeIsFatal)
{
    const Step step = three_segment_step();
    const Eager::ScopePartitioner partitioner;

    Autograd::Provenance out_of_range = step.provenance;
    out_of_range.origin.back() = static_cast<NodeID>(step.provenance.forward_node_count);

    SA_EXPECT_FATAL((void)partitioner.assign(step.graph, as_span(step.segments), out_of_range),
                    "has origin", "forward nodes");
}

/** @brief Tests that a backward node which reads a value of a higher key stops the process. */
TEST(Medium_ScopePartitionerFatal, ABackwardNodeThatReadsAHigherKeyIsFatal)
{
    TwoReLUGraph two;
    build_two_relu_graph(two);
    const GraphBuilder builder(two.graph);

    // Origins in forward order give the first backward node key 3 and its reader key 2.
    SA_ASSERT_OK_AND_ASSIGN(const ValueID of_first, builder.relu(two.first));
    SA_EXPECT_OK(builder.relu(of_first));
    two.provenance.origin.push_back(two.graph.value(two.first).producer);
    two.provenance.origin.push_back(two.graph.value(two.second).producer);

    SA_EXPECT_FATAL((void)Eager::ScopePartitioner{}.assign(two.graph, as_span(two.segments),
                                                           two.provenance),
                    "has region key 2", "whose key is 3");
}

/** @brief Tests that a LEAF-origin backward node that reads only leaves stops the process. */
TEST(Medium_ScopePartitionerFatal, ALeafOriginNodeThatReadsOnlyLeavesIsFatal)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID left = builder.parameter(Shape{4, 3}, "left");
    const ValueID right = builder.parameter(Shape{4, 3}, "right");
    SA_EXPECT_OK(builder.relu(left));

    Autograd::Provenance provenance;
    provenance.forward_node_count = graph.node_count();

    SA_EXPECT_OK(builder.add(left, right));
    provenance.origin.push_back(graph.value(left).producer);

    const std::vector<Eager::Segment> segments = {
        Eager::Segment{"", 0, 0, static_cast<NodeID>(provenance.forward_node_count)}};

    SA_EXPECT_FATAL((void)Eager::ScopePartitioner{}.assign(graph, as_span(segments), provenance),
                    "as its origin", "reads only leaves");
}
