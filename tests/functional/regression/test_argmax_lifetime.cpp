/**************************************************************************/
/* test_argmax_lifetime.cpp                                               */
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
 * @file test_argmax_lifetime.cpp
 * @brief Tests that the memory planner gives MAXPOOL2D's argmax bytes no other buffer overlaps.
 * @author Mustafa Garip
 * @see tests/functional/regression/README.md
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>

using namespace SushiAI;

using Graph::GraphBuilder;
using Graph::MemoryPlan;
using Graph::Placement;
using Graph::ValueId;

namespace
{
    /** @brief Whether two placements in one arena claim any byte in common. */
    bool overlaps(const Placement& a, const Placement& b)
    {
        if (a.arena != b.arena || a.arena == Graph::K_NO_ARENA) return false;
        return a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes;
    }
}

TEST(Regression_ArgmaxLifetime, ThePlannerKeepsMaxPoolsIndicesAliveUntilItsBackward)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    // x is a parameter so MAXPOOL2D_BACKWARD is emitted; see tests/functional/regression/README.md.
    const ValueId x = builder.parameter(Shape{2, 8, 8, 4}, "x", DType::Float64);
    const ValueId w = builder.parameter(Shape{4, 4}, "w", DType::Float64);

    Graph::ConvGeometry g;
    g.kernel_h = 2; g.kernel_w = 2;
    g.stride_h = 2; g.stride_w = 2;

    const Graph::PoolResult pooled = builder.maxpool2d(x, g);

    // Activations live and die while the argmax waits; see tests/functional/regression/README.md.
    const ValueId mean = builder.spatial_mean(pooled.output);
    const ValueId rows = builder.reshape(mean, Shape{2, 4});
    const ValueId hidden = builder.relu(builder.matmul(rows, w));
    const ValueId target = builder.input(Shape{2, 4}, "target", DType::Float64);
    const ValueId loss = builder.nll_loss(builder.log_softmax(hidden), target);
    graph.finalise();

    const std::array<ValueId, 2> wrt = {w, x};
    (void)Autograd::differentiate(graph, loss,
        SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

    // Differentiation emitted a reader of the indices; see tests/functional/regression/README.md.
    bool backward_reads_the_indices = false;
    for (const Graph::Node& node : graph.nodes())
        if (node.op == Ops::K_MAXPOOL2D_BACKWARD.id)
            for (ValueId in : node.inputs)
                if (in == pooled.indices) backward_reads_the_indices = true;

    ASSERT_TRUE(backward_reads_the_indices)
        << "no MAXPOOL2D_BACKWARD consumes the forward's indices; this test's premise is gone";

    const MemoryPlan plan = MemoryPlan::build(graph);
    ASSERT_TRUE(plan.reuse_enabled())
        << "the planner declined to overlap anything, so nothing here is under test: "
        << plan.reuse_note();

    // An unpacked argmax would pass every check below; see tests/functional/regression/README.md.
    ASSERT_TRUE(plan.is_packed(pooled.indices))
        << "the argmax is not in an arena, so the overlap check proves nothing about it";

    const Placement& argmax = plan.placement(pooled.indices);
    ASSERT_NE(argmax.arena, Graph::K_NO_ARENA);
    ASSERT_NE(argmax.bytes, std::size_t{0});
    ASSERT_EQ(argmax.bytes, std::size_t{2 * 4 * 4 * 4} * sizeof(std::int64_t))
        << "the argmax is not the INT64 buffer this test thinks it is";

    // No other buffer may overlap the argmax; see tests/functional/regression/README.md.
    std::size_t overlapping_neighbours = 0;
    for (std::size_t i = 0; i < plan.value_count(); ++i)
    {
        const ValueId other = static_cast<ValueId>(i);
        if (!plan.is_packed(other)) continue;
        if (plan.buffer_of(other) == plan.buffer_of(pooled.indices)) continue;

        const Placement& p = plan.placement(other);
        if (!overlaps(argmax, p)) continue;

        ++overlapping_neighbours;
        ADD_FAILURE()
            << "value " << other << " ('" << graph.value(other).debug_name
            << "') was packed onto MAXPOOL2D's argmax buffer. The pool's backward reads those "
               "indices, so whatever this value writes becomes the routing table for every "
               "pooled gradient - silently, and only in the trained weights.";
    }
    EXPECT_EQ(overlapping_neighbours, std::size_t{0});

    // The planner overlapped something in this graph; see tests/functional/regression/README.md.
    bool the_planner_reused_anything = false;
    for (const Graph::Arena& arena : plan.arenas())
        if (arena.unpacked_bytes > arena.bytes) the_planner_reused_anything = true;

    EXPECT_TRUE(the_planner_reused_anything)
        << "no buffer in this plan shares bytes with any other, so the argmax surviving says "
           "nothing about the planner. Lengthen the chain between the pool and its backward.";
}
