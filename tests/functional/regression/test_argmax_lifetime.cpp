/**************************************************************************/
/* test_argmax_lifetime.cpp                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

// MAXPOOL2D's argmax buffer has to survive from the forward node all the way
// to MAXPOOL2D_BACKWARD, and nothing pinned that.
//
// The gradchecks are evidence, but they are the wrong kind. If the planner
// handed the argmax's bytes to some later activation, the backward would read
// whatever that activation wrote and route every gradient to a plausible wrong
// pixel - and the gradchecks would keep passing right up until the planner
// changed, then fail somewhere else entirely with no mention of pooling.
//
// So this asserts on the plan, not on a number. Reuse in a MemoryPlan is byte
// overlap and nothing else: two values share storage exactly when their
// intervals in one arena intersect. An argmax whose interval intersects no
// other buffer's therefore owns its bytes for the whole step, which is the
// property "spans forward to backward" actually means. The check is written
// against the plan's own arena/placement tables rather than a lifetime enum,
// because the enum is the input to the decision and this is about the output.

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
        if (a.arena != b.arena || a.arena == Graph::NO_ARENA) return false;
        return a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes;
    }
}

TEST(Regression_ArgmaxLifetime, ThePlannerKeepsMaxPoolsIndicesAliveUntilItsBackward)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    // A pool feeding a chain long enough that the planner has real candidates
    // to reuse the argmax's bytes for. On a two-node graph the assertion below
    // would hold because there was nothing to overlap with, which proves
    // nothing; the non-vacuity check at the end is what makes it mean
    // something.
    // x is a parameter, not an input: the pool's backward is only emitted
    // when something upstream of it requires a gradient, and an input does
    // not. Differentiating with respect to x is what puts MAXPOOL2D_BACKWARD
    // in the graph at all.
    const ValueId x = builder.parameter(Shape{2, 8, 8, 4}, "x", DType::FLOAT64);
    const ValueId w = builder.parameter(Shape{4, 4}, "w", DType::FLOAT64);

    Graph::ConvGeometry g;
    g.kernel_h = 2; g.kernel_w = 2;
    g.stride_h = 2; g.stride_w = 2;

    const Graph::PoolResult pooled = builder.maxpool2d(x, g);

    // Between the pool and its backward: a spatial mean down to [2,1,1,4], a
    // reshape to a matrix and a small model, so several activations are born
    // and die while the argmax is waiting to be read.
    const ValueId mean = builder.spatial_mean(pooled.output);
    const ValueId rows = builder.reshape(mean, Shape{2, 4});
    const ValueId hidden = builder.relu(builder.matmul(rows, w));
    const ValueId target = builder.input(Shape{2, 4}, "target", DType::FLOAT64);
    const ValueId loss = builder.nll_loss(builder.log_softmax(hidden), target);
    graph.finalise();

    const std::array<ValueId, 2> wrt = {w, x};
    (void)Autograd::differentiate(graph, loss,
        SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

    // The premise: differentiation really did emit a consumer for the indices.
    // Without this the rest is a statement about a value nobody reads.
    bool backward_reads_the_indices = false;
    for (const Graph::Node& node : graph.nodes())
        if (node.op == Ops::MAXPOOL2D_BACKWARD.id)
            for (ValueId in : node.inputs)
                if (in == pooled.indices) backward_reads_the_indices = true;

    ASSERT_TRUE(backward_reads_the_indices)
        << "no MAXPOOL2D_BACKWARD consumes the forward's indices; this test's premise is gone";

    const MemoryPlan plan = MemoryPlan::build(graph);
    ASSERT_TRUE(plan.reuse_enabled())
        << "the planner declined to overlap anything, so nothing here is under test: "
        << plan.reuse_note();

    // Packed first. overlaps() answers false for an unpacked placement, so an
    // argmax that had been given its own allocation would satisfy every
    // assertion below for a reason this test does not document.
    ASSERT_TRUE(plan.is_packed(pooled.indices))
        << "the argmax is not in an arena, so the overlap check proves nothing about it";

    const Placement& argmax = plan.placement(pooled.indices);
    ASSERT_NE(argmax.arena, Graph::NO_ARENA);
    ASSERT_NE(argmax.bytes, std::size_t{0});
    ASSERT_EQ(argmax.bytes, std::size_t{2 * 4 * 4 * 4} * sizeof(std::int64_t))
        << "the argmax is not the INT64 buffer this test thinks it is";

    // The assertion. Every other buffer in the plan must miss it entirely.
    // Aliased names for the same bytes (ACCUMULATE) are not reuse, so they are
    // compared by owning buffer rather than by value id.
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

    // Non-vacuity: the planner did overlap *something* in this graph, so the
    // result above is a decision it made rather than an opportunity it never
    // had.
    bool the_planner_reused_anything = false;
    for (const Graph::Arena& arena : plan.arenas())
        if (arena.unpacked_bytes > arena.bytes) the_planner_reused_anything = true;

    EXPECT_TRUE(the_planner_reused_anything)
        << "no buffer in this plan shares bytes with any other, so the argmax surviving says "
           "nothing about the planner. Lengthen the chain between the pool and its backward.";
}
