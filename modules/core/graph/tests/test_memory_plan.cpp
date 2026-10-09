/**************************************************************************/
/* test_memory_plan.cpp                                                   */
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
 * @file test_memory_plan.cpp
 * @brief Tests that the memory planner never shares a byte between two values live together.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::Lifetime;
using Graph::MemoryPlan;
using Graph::K_NO_ARENA;
using Graph::K_NO_NODE;
using Graph::NodeId;
using Graph::PlanOptions;
using Graph::ValueId;

namespace
{
    /** @brief 128 bytes of float32 - exactly one alignment unit, so offsets read cleanly. */
    Shape unit_shape() { return Shape{32}; }

    /**
     * @brief Every node reachable from @p from, by a plain breadth-first walk.
     *
     * Deliberately not the planner's bitset closure: this is the independent
     * second opinion the contract check is measured against.
     */
    std::set<NodeId> descendants(const Graph::Graph& graph, NodeId from)
    {
        std::vector<std::vector<NodeId>> successors(graph.node_count());
        for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
            for (ValueId in : graph.node(id).inputs)
                if (graph.value(in).producer != K_NO_NODE)
                    successors[graph.value(in).producer].push_back(id);

        std::set<NodeId> seen;
        std::vector<NodeId> pending{from};

        while (!pending.empty())
        {
            const NodeId here = pending.back();
            pending.pop_back();
            for (NodeId next : successors[here])
                if (seen.insert(next).second) pending.push_back(next);
        }

        return seen;
    }

    /** @brief Every node that writes or reads @p id. */
    std::vector<NodeId> uses_of(const Graph::Graph& graph, ValueId id)
    {
        std::vector<NodeId> uses;
        if (graph.value(id).producer != K_NO_NODE) uses.push_back(graph.value(id).producer);

        for (NodeId node = 0; node < static_cast<NodeId>(graph.node_count()); ++node)
            for (ValueId in : graph.node(node).inputs)
                if (in == id) uses.push_back(node);

        return uses;
    }

    /** @brief Whether every use of @p earlier is a strict ancestor of @p later's producer. */
    bool finished_before(const Graph::Graph& graph, ValueId earlier, ValueId later)
    {
        const NodeId write = graph.value(later).producer;
        if (write == K_NO_NODE) return false;

        for (NodeId use : uses_of(graph, earlier))
            if (descendants(graph, use).count(write) == 0) return false;

        return true;
    }

    /** @brief Two values are live together unless one is finished before the other starts. */
    bool live_together(const Graph::Graph& graph, ValueId a, ValueId b)
    {
        return !finished_before(graph, a, b) && !finished_before(graph, b, a);
    }

    /**
     * @brief The planner's actual contract, checked over every pair of values.
     *
     * Exhaustive rather than sampled, because it is only quadratic in a handful
     * of values and a packing bug is exactly the kind that hides in the one
     * pair nobody thought to name.
     */
    void expect_no_live_overlap(const Graph::Graph& graph, const MemoryPlan& plan)
    {
        for (ValueId a = 0; a < static_cast<ValueId>(graph.value_count()); ++a)
            for (ValueId b = a + 1; b < static_cast<ValueId>(graph.value_count()); ++b)
            {
                if (!plan.is_packed(a) || !plan.is_packed(b)) continue;
                if (plan.placement(a).arena != plan.placement(b).arena) continue;
                if (plan.placement(a).buffer == plan.placement(b).buffer) continue;
                if (!live_together(graph, a, b)) continue;

                const auto& pa = plan.placement(a);
                const auto& pb = plan.placement(b);

                EXPECT_TRUE(pa.offset + pa.bytes <= pb.offset || pb.offset + pb.bytes <= pa.offset)
                    << "values " << a << " and " << b << " are live at the same time but share "
                    << "bytes: [" << pa.offset << ", " << pa.offset + pa.bytes << ") against ["
                    << pb.offset << ", " << pb.offset + pb.bytes << ")";
            }
    }
}

TEST(Unit_MemoryPlan, AnEmptyGraphAsksForNothing)
{
    const Graph::Graph graph;
    const MemoryPlan plan = MemoryPlan::build(graph);

    EXPECT_EQ(plan.value_count(), 0u);
    EXPECT_TRUE(plan.arenas().empty());
    EXPECT_EQ(plan.arena_bytes(), 0u);
}

TEST(Unit_MemoryPlan, ASingleActivationGetsOneArenaAndOffsetZero)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    const ValueId y = builder.relu(x);

    const MemoryPlan plan = MemoryPlan::build(graph);

    ASSERT_EQ(plan.arenas().size(), 1u);
    EXPECT_EQ(plan.arenas()[0].lifetime, Lifetime::Activation);
    EXPECT_EQ(plan.arenas()[0].buffers, 1u);
    EXPECT_EQ(plan.arena_bytes(), 128u);

    EXPECT_FALSE(plan.is_packed(x)) << "an input leaf is written from outside the step";
    ASSERT_TRUE(plan.is_packed(y));
    EXPECT_EQ(plan.placement(y).offset, 0u);
    EXPECT_EQ(plan.placement(y).bytes, 128u);
}

TEST(Unit_MemoryPlan, AGraphOfOnlyLeavesPacksNothing)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    builder.input(unit_shape(), "x");
    builder.parameter(unit_shape(), "w");

    const MemoryPlan plan = MemoryPlan::build(graph);

    EXPECT_TRUE(plan.arenas().empty());
    EXPECT_EQ(plan.arena_bytes(), 0u);
}

TEST(Unit_MemoryPlan, AChainPingPongsBetweenTwoBuffers)
{
    // Four chained activations need exactly two buffers; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    const ValueId a = builder.relu(x);
    const ValueId b = builder.relu(a);
    const ValueId c = builder.relu(b);
    const ValueId d = builder.relu(c);

    const MemoryPlan plan = MemoryPlan::build(graph);

    EXPECT_TRUE(plan.reuse_enabled());
    EXPECT_EQ(plan.arena_bytes(), 256u) << "four 128-byte activations should fit in two slots";
    EXPECT_EQ(plan.unpacked_bytes(), 512u);

    EXPECT_EQ(plan.placement(a).offset, plan.placement(c).offset);
    EXPECT_EQ(plan.placement(b).offset, plan.placement(d).offset);
    EXPECT_NE(plan.placement(a).offset, plan.placement(b).offset);

    expect_no_live_overlap(graph, plan);
}

TEST(Unit_MemoryPlan, TwoUnorderedBranchesNeverShareBytes)
{
    // Nothing orders node 2 before node 3; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");

    const ValueId a1 = builder.relu(x);
    const ValueId a2 = builder.relu(a1);

    const ValueId b1 = builder.sigmoid(x);
    const ValueId b2 = builder.sigmoid(b1);

    const ValueId out = builder.add(a2, b2);

    const MemoryPlan plan = MemoryPlan::build(graph);

    ASSERT_TRUE(plan.is_packed(a1));
    ASSERT_TRUE(plan.is_packed(b1));

    const auto& left = plan.placement(a1);
    const auto& right = plan.placement(b1);

    EXPECT_TRUE(left.offset + left.bytes <= right.offset
                || right.offset + right.bytes <= left.offset)
        << "the two branches of a diamond are unordered, so a1 and b1 must not share bytes; "
        << "a1 at " << left.offset << ", b1 at " << right.offset;

    EXPECT_TRUE(plan.is_packed(out));
    expect_no_live_overlap(graph, plan);
}

TEST(Unit_MemoryPlan, OnlyTheNonPersistentClassesArePacked)
{
    // Only ACTIVATION and SCRATCH reach an arena; see tests/unit/README.md.
    Graph::Graph graph;

    const ValueId parameter = graph.add_value(unit_shape(), DType::Float32, Lifetime::Parameter, "w");
    graph.add_node(Ops::K_LEAF.id, {}, {parameter});

    const ValueId input = graph.add_value(unit_shape(), DType::Float32, Lifetime::Input, "x");
    graph.add_node(Ops::K_LEAF.id, {}, {input});

    const ValueId activation = graph.add_value(unit_shape(), DType::Float32, Lifetime::Activation, "h");
    graph.add_node(Ops::K_RELU.id, {input}, {activation});

    const ValueId scratch = graph.add_value(unit_shape(), DType::Float32, Lifetime::Scratch, "t");
    graph.add_node(Ops::K_RELU.id, {activation}, {scratch});

    const ValueId gradient = graph.add_value(unit_shape(), DType::Float32, Lifetime::Gradient, "dw");
    graph.add_node(Ops::K_RELU.id, {scratch}, {gradient});

    const ValueId state = graph.add_value(unit_shape(), DType::Float32, Lifetime::OptimizerState, "m");
    graph.add_node(Ops::K_RELU.id, {gradient}, {state});

    const MemoryPlan plan = MemoryPlan::build(graph);

    EXPECT_FALSE(plan.is_packed(parameter));
    EXPECT_FALSE(plan.is_packed(input));
    EXPECT_FALSE(plan.is_packed(gradient));
    EXPECT_FALSE(plan.is_packed(state));

    ASSERT_TRUE(plan.is_packed(activation));
    ASSERT_TRUE(plan.is_packed(scratch));
    EXPECT_NE(plan.placement(activation).arena, plan.placement(scratch).arena)
        << "one arena per lifetime class, so the tracker keeps a shard per class";

    ASSERT_EQ(plan.arenas().size(), 2u);
    EXPECT_EQ(plan.arenas()[0].lifetime, Lifetime::Activation);
    EXPECT_EQ(plan.arenas()[1].lifetime, Lifetime::Scratch);
}

TEST(Unit_MemoryPlan, AValueNothingWritesKeepsItsOwnAllocation)
{
    // An ACTIVATION with no producer lives the whole run; see tests/unit/README.md.
    Graph::Graph graph;

    const ValueId orphan = graph.add_value(unit_shape(), DType::Float32, Lifetime::Activation, "?");
    const ValueId out = graph.add_value(unit_shape(), DType::Float32, Lifetime::Activation, "y");
    graph.add_node(Ops::K_RELU.id, {orphan}, {out});

    const MemoryPlan plan = MemoryPlan::build(graph);

    EXPECT_FALSE(plan.is_packed(orphan));
    EXPECT_TRUE(plan.is_packed(out));
}

TEST(Unit_MemoryPlan, AnAccumulateChainIsOneBufferWithOneLiveRange)
{
    // ACCUMULATE's result is its accumulator's storage; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    const ValueId first = builder.relu(x);
    const ValueId second = builder.sigmoid(x);
    const ValueId accumulated = builder.accumulate(first, second);
    const ValueId sink = builder.relu(accumulated);

    const MemoryPlan plan = MemoryPlan::build(graph);

    EXPECT_EQ(plan.buffer_of(accumulated), plan.buffer_of(first));
    EXPECT_EQ(plan.placement(accumulated).offset, plan.placement(first).offset);
    EXPECT_EQ(plan.placement(accumulated).arena, plan.placement(first).arena);

    EXPECT_NE(plan.placement(sink).offset, plan.placement(first).offset)
        << "sink is produced by the node that last reads the accumulator's bytes";

    // The alias class counts once, not twice.
    ASSERT_EQ(plan.arenas().size(), 1u);
    EXPECT_EQ(plan.arenas()[0].buffers, 3u);
}

TEST(Unit_MemoryPlan, EveryStartIsAWholeNumberOfItsOwnElements)
{
    // The float64 buffer starts at 24, not at 20; see tests/unit/README.md.
    Graph::Graph graph;

    const ValueId narrow_leaf = graph.add_value(Shape{5}, DType::Float32, Lifetime::Input, "x");
    graph.add_node(Ops::K_LEAF.id, {}, {narrow_leaf});

    const ValueId wide_leaf = graph.add_value(Shape{2}, DType::Float64, Lifetime::Input, "y");
    graph.add_node(Ops::K_LEAF.id, {}, {wide_leaf});

    const ValueId narrow = graph.add_value(Shape{5}, DType::Float32, Lifetime::Activation, "a");
    graph.add_node(Ops::K_RELU.id, {narrow_leaf}, {narrow});

    const ValueId wide = graph.add_value(Shape{2}, DType::Float64, Lifetime::Activation, "b");
    graph.add_node(Ops::K_RELU.id, {wide_leaf}, {wide});

    PlanOptions options;
    options.alignment = 1;

    const MemoryPlan plan = MemoryPlan::build(graph, options);

    EXPECT_EQ(plan.placement(narrow).offset % dtype_size(DType::Float32), 0u);
    EXPECT_EQ(plan.placement(wide).offset % dtype_size(DType::Float64), 0u);

    EXPECT_EQ(plan.placement(narrow).offset, 0u);
    EXPECT_EQ(plan.placement(wide).offset, 24u)
        << "20 bytes of float32 rounded up to the next whole float64";
}

TEST(Unit_MemoryPlan, ReuseCanBeSwitchedOffAndSaysSo)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    const ValueId a = builder.relu(x);
    const ValueId b = builder.relu(a);
    const ValueId c = builder.relu(b);
    (void)c;

    PlanOptions options;
    options.reuse = false;

    const MemoryPlan plan = MemoryPlan::build(graph, options);

    EXPECT_FALSE(plan.reuse_enabled());
    EXPECT_FALSE(plan.reuse_note().empty());
    EXPECT_EQ(plan.arena_bytes(), 384u) << "three activations, three slots";
}

TEST(Unit_MemoryPlan, TooManyNodesDeclinesOverlapRatherThanApproximatingIt)
{
    // Over the node limit the planner declines to overlap; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    ValueId last = builder.relu(x);
    for (int i = 0; i < 4; ++i) last = builder.relu(last);
    (void)last;

    PlanOptions options;
    options.reuse_node_limit = 2;

    const MemoryPlan plan = MemoryPlan::build(graph, options);

    EXPECT_FALSE(plan.reuse_enabled());
    EXPECT_NE(plan.reuse_note().find("reuse_node_limit"), std::string_view::npos);
    EXPECT_EQ(plan.arena_bytes(), plan.unpacked_bytes()) << "declined means no saving at all";
}

TEST(Unit_MemoryPlan, APinnedValueKeepsItsOwnAllocation)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    const ValueId a = builder.relu(x);
    const ValueId b = builder.relu(a);

    const ValueId pinned[] = {a};

    PlanOptions options;
    options.pinned = SushiRuntime::span<const ValueId>(pinned, 1);

    const MemoryPlan plan = MemoryPlan::build(graph, options);

    EXPECT_FALSE(plan.is_packed(a));
    EXPECT_TRUE(plan.is_packed(b));
}

TEST(Unit_MemoryPlan, PinningAValueThatDoesNotExistThrows)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    (void)builder.input(unit_shape(), "x");

    const ValueId pinned[] = {99};

    PlanOptions options;
    options.pinned = SushiRuntime::span<const ValueId>(pinned, 1);

    EXPECT_THROW((void)MemoryPlan::build(graph, options), Error);
}

TEST(Unit_MemoryPlan, ZeroAlignmentThrows)
{
    Graph::Graph graph;
    PlanOptions options;
    options.alignment = 0;

    EXPECT_THROW((void)MemoryPlan::build(graph, options), Error);
}

TEST(Unit_MemoryPlan, AGraphThatIsNotInTopologicalOrderThrows)
{
    // The pass relies on topological order; see tests/unit/README.md.
    Graph::Graph graph;

    const ValueId a = graph.add_value(unit_shape(), DType::Float32, Lifetime::Activation, "a");
    const ValueId b = graph.add_value(unit_shape(), DType::Float32, Lifetime::Activation, "b");

    graph.add_node(Ops::K_RELU.id, {a}, {b});
    graph.add_node(Ops::K_RELU.id, {b}, {a});

    EXPECT_THROW((void)MemoryPlan::build(graph), Error);
}

TEST(Unit_MemoryPlan, ThePlanIsDeterministicForTheSameGraph)
{
    // Two builds give byte-identical offsets; see tests/unit/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(unit_shape(), "x");
    const ValueId a = builder.relu(x);
    const ValueId b = builder.sigmoid(x);
    const ValueId c = builder.add(a, b);
    const ValueId d = builder.tanh(c);
    (void)d;

    const MemoryPlan first = MemoryPlan::build(graph);
    const MemoryPlan second = MemoryPlan::build(graph);

    ASSERT_EQ(first.value_count(), second.value_count());
    for (ValueId id = 0; id < static_cast<ValueId>(first.value_count()); ++id)
    {
        EXPECT_EQ(first.placement(id).arena, second.placement(id).arena) << "value " << id;
        EXPECT_EQ(first.placement(id).offset, second.placement(id).offset) << "value " << id;
    }
}

TEST(Unit_MemoryPlan, TheTwoResultsOfOneNodeAreNeverGivenTheSameBytes)
{
    // A node does not reach itself, so its results conflict; see tests/unit/README.md.
    Graph::Graph graph;

    const ValueId input = graph.add_value(Shape{8, 4}, DType::Float32, Lifetime::Activation, "x");
    graph.add_node(Ops::K_LEAF.id, {}, {input});

    const ValueId result = graph.add_value(Shape{8, 4}, DType::Float32,
                                           Lifetime::Activation, "y");
    const ValueId spill = graph.add_value(Shape{8, 4}, DType::Float32,
                                          Lifetime::Activation, "pre");
    graph.add_node(Ops::K_GEMM_BIAS_GELU.id, {input}, {result, spill});

    // Both are read afterwards, as the forward and the backward pass read them.
    const ValueId downstream = graph.add_value(Shape{8, 4}, DType::Float32,
                                               Lifetime::Activation, "h");
    graph.add_node(Ops::K_RELU.id, {result}, {downstream});

    const ValueId gradient = graph.add_value(Shape{8, 4}, DType::Float32,
                                             Lifetime::Activation, "dx");
    graph.add_node(Ops::K_GELU_BACKWARD.id, {downstream, spill}, {gradient});

    const MemoryPlan plan = MemoryPlan::build(graph);

    ASSERT_TRUE(plan.is_packed(result));
    ASSERT_TRUE(plan.is_packed(spill));
    EXPECT_NE(plan.placement(result).buffer, plan.placement(spill).buffer)
        << "the fused node's two results were aliased onto one buffer";
    EXPECT_TRUE(plan.placement(result).offset + plan.placement(result).bytes
                    <= plan.placement(spill).offset
                || plan.placement(spill).offset + plan.placement(spill).bytes
                    <= plan.placement(result).offset)
        << "the fused node's two results overlap in the arena";

    expect_no_live_overlap(graph, plan);
}
