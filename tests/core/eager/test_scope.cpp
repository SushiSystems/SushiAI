/**************************************************************************/
/* test_scope.cpp                                                         */
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
 * @file test_scope.cpp
 * @brief Tests that Scope opens its name on a ScopeTrace and closes it when it ends.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/eager/scope.hpp>
#include <SushiAI/eager/scope_trace.hpp>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Eager::Scope;
using Eager::ScopeTrace;
using Eager::Segment;

namespace
{
    /** @brief Appends one K_ADD node to @p graph that adds an input value to itself. */
    void trace_node(Graph::Graph& graph)
    {
        const Graph::ValueID source =
            graph.add_value(Shape{2}, DType::Float32, Graph::Lifetime::Input);
        const Graph::ValueID result =
            graph.add_value(Shape{2}, DType::Float32, Graph::Lifetime::Activation);
        graph.add_node(Ops::K_ADD.id, {source, source}, {result});
    }
} // namespace

/** @brief Tests that a scope is open for exactly the lifetime of its guard. */
TEST(Small_Scope, TheScopeIsOpenWhileTheGuardLives)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    {
        const Scope scope(trace, "body");
        EXPECT_EQ(trace.depth(), 1u);
    }

    EXPECT_EQ(trace.depth(), 0u);
}

/** @brief Tests that nested guards close innermost first, so the trace can be sealed. */
TEST(Small_Scope, NestedGuardsCloseInnermostFirst)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    {
        const Scope outer(trace, "outer");
        {
            const Scope inner(trace, "inner");
            EXPECT_EQ(trace.depth(), 2u);
            trace_node(graph);
        }
        EXPECT_EQ(trace.depth(), 1u);
        trace_node(graph);
    }

    const std::vector<Segment> segments = trace.seal();
    ASSERT_EQ(segments.size(), 2u);
    EXPECT_EQ(segments[0].name, "outer/inner");
    EXPECT_EQ(segments[1].name, "outer");
}

/** @brief Tests that a guard can be neither copied nor moved, so it closes exactly once. */
TEST(Small_Scope, AGuardCannotBeCopiedOrMoved)
{
    EXPECT_FALSE(std::is_copy_constructible<Scope>::value);
    EXPECT_FALSE(std::is_move_constructible<Scope>::value);
    EXPECT_FALSE(std::is_copy_assignable<Scope>::value);
    EXPECT_FALSE(std::is_move_assignable<Scope>::value);
}
