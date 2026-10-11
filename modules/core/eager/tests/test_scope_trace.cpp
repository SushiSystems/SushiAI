/**************************************************************************/
/* test_scope_trace.cpp                                                   */
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
 * @file test_scope_trace.cpp
 * @brief Tests how ScopeTrace cuts a trace into segments and which calls it rejects.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/eager/scope.hpp>
#include <SushiAI/eager/scope_trace.hpp>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/graph/ir.hpp>
#include <common/test_throws.hpp>

using namespace SushiAI;
using Eager::Scope;
using Eager::ScopeTrace;
using Eager::Segment;
using Graph::Lifetime;
using Graph::NodeID;
using Graph::ValueID;

namespace
{
    /**
     * @brief Appends @p count nodes to @p graph, each adding an input to itself.
     */
    void trace_nodes(Graph::Graph& graph, std::size_t count)
    {
        const ValueID source = graph.add_value(Shape{2}, DType::Float32, Lifetime::Input);
        for (std::size_t i = 0; i < count; ++i)
        {
            const ValueID result = graph.add_value(Shape{2}, DType::Float32, Lifetime::Activation);
            graph.add_node(Ops::K_ADD.id, {source, source}, {result});
        }
    }

    /**
     * @brief Checks one segment against the name, occurrence and node range it should have.
     * @return Success, or a failure printing the segment's four fields.
     */
    ::testing::AssertionResult segment_is(const Segment& segment, const std::string& name,
                                          std::uint32_t occurrence, NodeID begin, NodeID end)
    {
        if (segment.name == name && segment.occurrence == occurrence && segment.begin == begin &&
            segment.end == end)
            return ::testing::AssertionSuccess();

        return ::testing::AssertionFailure()
               << "segment is \"" << segment.name << "\"#" << segment.occurrence << " ["
               << segment.begin << ", " << segment.end << "), expected \"" << name << "\"#"
               << occurrence << " [" << begin << ", " << end << ")";
    }
} // namespace

/** @brief Tests that nodes traced with no scope open form one segment with an empty name. */
TEST(Unit_ScopeTrace, NodesOutsideAnyScopeFormOneUnnamedSegment)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    trace_nodes(graph, 3);
    const std::vector<Segment> segments = trace.seal();

    ASSERT_EQ(segments.size(), 1u);
    EXPECT_TRUE(segment_is(segments[0], "", 0, 0, 3));
}

/** @brief Tests that a scope in the middle of a trace yields a segment before, in and after it. */
TEST(Unit_ScopeTrace, AScopeSplitsTheTraceIntoThreeSegments)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    trace_nodes(graph, 1);
    {
        const Scope scope(trace, "body");
        trace_nodes(graph, 2);
    }
    trace_nodes(graph, 3);
    const std::vector<Segment> segments = trace.seal();

    ASSERT_EQ(segments.size(), 3u);
    EXPECT_TRUE(segment_is(segments[0], "", 0, 0, 1));
    EXPECT_TRUE(segment_is(segments[1], "body", 0, 1, 3));
    EXPECT_TRUE(segment_is(segments[2], "", 1, 3, 6));
}

/**
 * @brief Tests that a nested scope is named "outer/inner" and the outer scope resumes with
 *        occurrence 1.
 */
TEST(Unit_ScopeTrace, ANestedScopeIsNamedByItsPath)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    {
        const Scope outer(trace, "outer");
        trace_nodes(graph, 1);
        {
            const Scope inner(trace, "inner");
            EXPECT_EQ(trace.depth(), 2u);
            trace_nodes(graph, 2);
        }
        trace_nodes(graph, 1);
    }
    const std::vector<Segment> segments = trace.seal();

    ASSERT_EQ(segments.size(), 3u);
    EXPECT_TRUE(segment_is(segments[0], "outer", 0, 0, 1));
    EXPECT_TRUE(segment_is(segments[1], "outer/inner", 0, 1, 3));
    EXPECT_TRUE(segment_is(segments[2], "outer", 1, 3, 4));
}

/** @brief Tests that a scope opened three times in a loop gets occurrences 0, 1 and 2. */
TEST(Unit_ScopeTrace, AScopeOpenedTwiceGetsRisingOccurrences)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    for (int step = 0; step < 3; ++step)
    {
        const Scope cell(trace, "cell");
        trace_nodes(graph, 2);
    }
    const std::vector<Segment> segments = trace.seal();

    ASSERT_EQ(segments.size(), 3u);
    EXPECT_TRUE(segment_is(segments[0], "cell", 0, 0, 2));
    EXPECT_TRUE(segment_is(segments[1], "cell", 1, 2, 4));
    EXPECT_TRUE(segment_is(segments[2], "cell", 2, 4, 6));
}

/** @brief Tests that a scope with no node leaves no segment and does not use up an occurrence. */
TEST(Unit_ScopeTrace, AnEmptyScopeLeavesNoSegment)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    trace_nodes(graph, 1);
    {
        const Scope empty(trace, "cell");
    }
    {
        const Scope filled(trace, "cell");
        trace_nodes(graph, 1);
    }
    const std::vector<Segment> segments = trace.seal();

    ASSERT_EQ(segments.size(), 2u);
    EXPECT_TRUE(segment_is(segments[0], "", 0, 0, 1));
    EXPECT_TRUE(segment_is(segments[1], "cell", 0, 1, 2));
}

/**
 * @brief Tests that seal() throws while a scope is open, names that scope and leaves the running
 *        segment whole.
 */
TEST(Unit_ScopeTrace, SealWithAScopeStillOpenThrows)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    trace.open("outer");
    trace.open("inner");
    trace_nodes(graph, 1);

    SA_EXPECT_ERROR((void)trace.seal(), "seal()", "still open", "outer/inner");
    EXPECT_EQ(trace.depth(), 2u);

    trace_nodes(graph, 1);
    trace.close();
    trace.close();
    const std::vector<Segment> segments = trace.seal();

    ASSERT_EQ(segments.size(), 1u);
    EXPECT_TRUE(segment_is(segments[0], "outer/inner", 0, 0, 2));
}

/** @brief Tests that close() throws when no scope is open. */
TEST(Unit_ScopeTrace, CloseWithNoScopeOpenThrows)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    SA_EXPECT_ERROR(trace.close(), "close()", "no scope open");

    trace.open("once");
    trace.close();
    SA_EXPECT_ERROR(trace.close(), "close()", "no scope open");
}

/** @brief Tests that the segments' ranges join end to begin from node 0 to the node count. */
TEST(Unit_ScopeTrace, SegmentsCoverEveryNodeExactlyOnce)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    trace_nodes(graph, 2);
    for (int step = 0; step < 2; ++step)
    {
        const Scope outer(trace, "outer");
        trace_nodes(graph, 1);
        {
            const Scope empty(trace, "empty");
        }
        {
            const Scope inner(trace, "inner");
            trace_nodes(graph, 3);
        }
        trace_nodes(graph, 1);
    }
    trace_nodes(graph, 1);
    const std::vector<Segment> segments = trace.seal();

    NodeID next = 0;
    for (const Segment& segment : segments)
    {
        EXPECT_EQ(segment.begin, next);
        EXPECT_LT(segment.begin, segment.end);
        next = segment.end;
    }
    EXPECT_EQ(next, graph.node_count());
}

/** @brief Tests that open() rejects an empty name and leaves the trace as it was. */
TEST(Unit_ScopeTrace, OpenRejectsAnEmptyName)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    SA_EXPECT_ERROR(trace.open(""), "open()", "empty scope name");
    EXPECT_EQ(trace.depth(), 0u);
}

/** @brief Tests that open() rejects a name containing the path separator '/'. */
TEST(Unit_ScopeTrace, OpenRejectsANameWithASlash)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    SA_EXPECT_ERROR(trace.open("outer/inner"), "open()", "outer/inner", "may not contain");
    EXPECT_EQ(trace.depth(), 0u);
}

/** @brief Tests that open() rejects a name containing the occurrence separator '#'. */
TEST(Unit_ScopeTrace, OpenRejectsANameWithAHash)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    SA_EXPECT_ERROR(trace.open("cell#0"), "open()", "cell#0", "may not contain");
    EXPECT_EQ(trace.depth(), 0u);
}

/** @brief Tests that a rejected open() does not end the running segment. */
TEST(Unit_ScopeTrace, ARejectedNameDoesNotSplitTheRunningSegment)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    trace_nodes(graph, 1);
    SA_EXPECT_ERROR(trace.open("a/b"), "may not contain");
    trace_nodes(graph, 1);
    const std::vector<Segment> segments = trace.seal();

    ASSERT_EQ(segments.size(), 1u);
    EXPECT_TRUE(segment_is(segments[0], "", 0, 0, 2));
}

/** @brief Tests that a rejected close() does not end the running segment. */
TEST(Unit_ScopeTrace, ARejectedCloseDoesNotSplitTheRunningSegment)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    trace_nodes(graph, 1);
    SA_EXPECT_ERROR(trace.close(), "no scope open");
    trace_nodes(graph, 1);
    const std::vector<Segment> segments = trace.seal();

    ASSERT_EQ(segments.size(), 1u);
    EXPECT_TRUE(segment_is(segments[0], "", 0, 0, 2));
}

/** @brief Tests that a second seal() throws. */
TEST(Unit_ScopeTrace, ASecondSealThrows)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    trace_nodes(graph, 1);
    const std::vector<Segment> segments = trace.seal();
    ASSERT_EQ(segments.size(), 1u);

    SA_EXPECT_ERROR((void)trace.seal(), "seal()", "sealed trace");
}

/** @brief Tests that open() and close() throw once the trace is sealed. */
TEST(Unit_ScopeTrace, OpenAndCloseAfterSealThrow)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    (void)trace.seal();

    SA_EXPECT_ERROR(trace.open("late"), "open()", "sealed trace");
    SA_EXPECT_ERROR(trace.close(), "close()", "sealed trace");
    EXPECT_EQ(trace.depth(), 0u);
}
