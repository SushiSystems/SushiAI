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
 * @file test_scope_trace.cpp
 * @brief Tests how ScopeTrace cuts a trace into segments and which calls stop the process.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/eager/scope.hpp>
#include <SushiAI/eager/scope_trace.hpp>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>

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

    /**
     * @brief Traces one round of nested scopes into @p graph: a node, an empty scope, an inner
     *        scope of three nodes and a node, all inside "outer".
     */
    void trace_one_round(ScopeTrace& trace, Graph::Graph& graph)
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
} // namespace

/** @brief Tests that nodes traced with no scope open form one segment with an empty name. */
TEST(Small_ScopeTrace, NodesOutsideAnyScopeFormOneUnnamedSegment)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    trace_nodes(graph, 3);
    const std::vector<Segment> segments = trace.seal();

    ASSERT_EQ(segments.size(), 1u);
    EXPECT_TRUE(segment_is(segments[0], "", 0, 0, 3));
}

/** @brief Tests that a scope in the middle of a trace yields a segment before, in and after it. */
TEST(Small_ScopeTrace, AScopeSplitsTheTraceIntoThreeSegments)
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
TEST(Small_ScopeTrace, ANestedScopeIsNamedByItsPath)
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

/** @brief Tests that a scope with no node leaves no segment and does not use up an occurrence. */
TEST(Small_ScopeTrace, AnEmptyScopeLeavesNoSegment)
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

/** @brief Tests that seal() stops the process while a scope is open and names that scope. */
TEST(Medium_ScopeTraceFatal, SealWithAScopeStillOpenIsFatal)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    trace.open("outer");
    trace.open("inner");
    trace_nodes(graph, 1);

    SA_EXPECT_FATAL((void)trace.seal(), "seal()", "still open", "outer/inner");
}

/** @brief Tests that close() stops the process when no scope is open. */
TEST(Medium_ScopeTraceFatal, CloseWithNoScopeOpenIsFatal)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    SA_EXPECT_FATAL(trace.close(), "close()", "no scope open");

    trace.open("once");
    trace.close();
    SA_EXPECT_FATAL(trace.close(), "close()", "no scope open");
}

/** @brief Tests that the segments of a long trace join end to begin over every node once. */
TEST(Small_ScopeTrace, SegmentsCoverEveryNodeExactlyOnce)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    trace_nodes(graph, 2);
    trace_one_round(trace, graph);
    trace_one_round(trace, graph);
    trace_nodes(graph, 1);
    const std::vector<Segment> segments = trace.seal();

    ASSERT_EQ(segments.size(), 8u);
    EXPECT_TRUE(segment_is(segments[0], "", 0, 0, 2));
    EXPECT_TRUE(segment_is(segments[1], "outer", 0, 2, 3));
    EXPECT_TRUE(segment_is(segments[2], "outer/inner", 0, 3, 6));
    EXPECT_TRUE(segment_is(segments[3], "outer", 1, 6, 7));
    EXPECT_TRUE(segment_is(segments[4], "outer", 2, 7, 8));
    EXPECT_TRUE(segment_is(segments[5], "outer/inner", 1, 8, 11));
    EXPECT_TRUE(segment_is(segments[6], "outer", 3, 11, 12));
    EXPECT_TRUE(segment_is(segments[7], "", 1, 12, 13));
    EXPECT_EQ(segments.back().end, graph.node_count());
}

namespace
{
/** @brief Holds an illegal scope name and the two words open() reports for it. */
struct ScopeNameRow
{
    const char* name;   /**< The name handed to open(). */
    const char* first;  /**< The first word the report names after "open()". */
    const char* second; /**< The second word the report names after the first. */
};
} // namespace

namespace
{
class Small_ScopeTraceLegalName : public ::testing::TestWithParam<const char*>
{
};
} // namespace

/** @brief Tests that a name is legal exactly when open() accepts it. */
TEST_P(Small_ScopeTraceLegalName, ALegalNameIsAcceptedByOpen)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    EXPECT_TRUE(ScopeTrace::is_legal_scope_name(GetParam()));
    trace.open(GetParam());

    EXPECT_EQ(trace.depth(), 1u);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ScopeTraceLegalName,
                         ::testing::Values("encoder", "block 3"));

namespace
{
class Medium_ScopeTraceRefusedName : public ::testing::TestWithParam<ScopeNameRow>
{
};
} // namespace

/** @brief Tests that an empty name or a name with a separator is illegal and stops open(). */
TEST_P(Medium_ScopeTraceRefusedName, AnIllegalNameIsRefusedByOpen)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    EXPECT_FALSE(ScopeTrace::is_legal_scope_name(GetParam().name));
    SA_EXPECT_FATAL(trace.open(GetParam().name), "open()", GetParam().first, GetParam().second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_ScopeTraceRefusedName,
                         ::testing::Values(ScopeNameRow{"", "was given", "empty scope name"},
                                           ScopeNameRow{"outer/inner", "outer/inner",
                                                        "may not contain"},
                                           ScopeNameRow{"cell#0", "cell#0", "may not contain"}));

/** @brief Tests that a second seal() stops the process. */
TEST(Medium_ScopeTraceFatal, ASecondSealIsFatal)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    trace_nodes(graph, 1);
    const std::vector<Segment> segments = trace.seal();
    ASSERT_EQ(segments.size(), 1u);

    SA_EXPECT_FATAL((void)trace.seal(), "seal()", "sealed trace");
}

/** @brief Tests that open() and close() stop the process once the trace is sealed. */
TEST(Medium_ScopeTraceFatal, OpenAndCloseAfterSealAreFatal)
{
    Graph::Graph graph;
    ScopeTrace trace(graph);

    (void)trace.seal();

    SA_EXPECT_FATAL(trace.open("late"), "open()", "sealed trace");
    SA_EXPECT_FATAL(trace.close(), "close()", "sealed trace");
}
