/**************************************************************************/
/* test_gather_ops.cpp                                                    */
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
 * @file test_gather_ops.cpp
 * @brief Tests the shape rules and the emitted IR of GATHER_ROWS and SCATTER_ADD_ROWS, and the
 *        INT32 leaf under the cast pass.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::Traced;
using Graph::ValueID;

namespace
{
    /** @brief Holds the row count of the table every case reads. */
    constexpr int64_t K_TABLE_ROWS = 7;

    /** @brief Holds the row length of the table every case reads. */
    constexpr int64_t K_ROW_LENGTH = 3;

    /** @brief Expects @p out to be the single result of an @p op node that reads @p inputs. */
    void expect_produced_by(const Graph::Graph& graph, ValueID out, OpID op,
                            const std::vector<ValueID>& inputs)
    {
        const Graph::NodeID producer = graph.value(out).producer;
        ASSERT_NE(producer, Graph::K_NO_NODE);

        const Graph::Node& node = graph.node(producer);
        EXPECT_EQ(node.op, op) << "written by " << Ops::name_of(node.op);
        EXPECT_EQ(node.inputs, inputs);
        ASSERT_EQ(node.outputs.size(), 1u);
        EXPECT_EQ(node.outputs[0], out);
    }

    /** @brief Expects @p refused to name @p part, and the graph to hold no node beyond @p nodes. */
    void expect_refused(const Graph::Graph& graph, const Traced& refused, std::size_t nodes,
                        const char* part)
    {
        SA_EXPECT_FAILURE(refused, StatusCode::InvalidDescription, part);
        EXPECT_EQ(graph.node_count(), nodes) << "a refused call left a node behind";
        EXPECT_TRUE(graph.has_refusal());
    }
}

/** @brief Tests that a gather over a rank-1 index writes GATHER_ROWS over [table, indices]. */
TEST(Unit_GatherOps, GatherRowsWritesItsIdentityOverTheTableAndTheIndex)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);

    const Traced out = builder.gather_rows(table, indices);

    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());
    expect_produced_by(graph, out.value(), Ops::K_GATHER_ROWS.id, {table, indices});
    EXPECT_EQ(graph.value(out.value()).shape, (Shape{5, K_ROW_LENGTH}));
    EXPECT_EQ(graph.value(out.value()).dtype, DType::Float32);
    EXPECT_EQ(graph.value(out.value()).lifetime, Graph::Lifetime::Activation);
    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that the result keeps every extent of the index and the table's element type. */
TEST(Unit_GatherOps, GatherRowsAppendsTheRowLengthToAnIndexOfAnyRank)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table =
        builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table", DType::Float64);
    const ValueID indices = builder.input(Shape{2, 4}, "indices", DType::Int32);
    const ValueID deep = builder.input(Shape{2, 1, 3, 1, 2}, "deep", DType::Int32);

    const Traced out = builder.gather_rows(table, indices);
    const Traced deepest = builder.gather_rows(table, deep);

    ASSERT_TRUE(out.has_value());
    ASSERT_TRUE(deepest.has_value());
    EXPECT_EQ(graph.value(out.value()).shape, (Shape{2, 4, K_ROW_LENGTH}));
    EXPECT_EQ(graph.value(out.value()).dtype, DType::Float64);
    EXPECT_EQ(graph.value(deepest.value()).shape, (Shape{2, 1, 3, 1, 2, K_ROW_LENGTH}));
}

/** @brief Tests that a table of rank 1 or 3 is refused and no node is added. */
TEST(Unit_GatherOps, GatherRowsRefusesATableThatIsNotRankTwo)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID flat = builder.parameter(Shape{K_TABLE_ROWS}, "flat");
    const ValueID cube = builder.parameter(Shape{K_TABLE_ROWS, 2, K_ROW_LENGTH}, "cube");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.gather_rows(flat, indices), nodes, "rank-2 [V, D] table");
    expect_refused(graph, builder.gather_rows(cube, indices), nodes, "[7, 2, 3]");
}

/** @brief Tests that a FLOAT32 index and an INT64 index are both refused. */
TEST(Unit_GatherOps, GatherRowsRefusesAnIndexThatIsNotInt32)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID real = builder.input(Shape{5}, "real", DType::Float32);
    const ValueID wide = builder.input(Shape{5}, "wide", DType::Int64);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.gather_rows(table, real), nodes, "INT32 index");
    expect_refused(graph, builder.gather_rows(table, wide), nodes, "INT32 index");
}

/** @brief Tests that an integer table is refused. */
TEST(Unit_GatherOps, GatherRowsRefusesATableThatIsNotFloating)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.input(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table", DType::Int32);
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.gather_rows(table, indices), nodes,
                   "half, float32 or float64 table");
}

/** @brief Tests that an index of the highest rank is refused, as its result has no rank left. */
TEST(Unit_GatherOps, GatherRowsRefusesAnIndexWhoseResultWouldPassTheHighestRank)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID indices = builder.input(Shape{1, 1, 1, 1, 1, 1}, "indices", DType::Int32);
    const std::size_t nodes = graph.node_count();

    ASSERT_EQ(graph.value(indices).shape.rank(), K_MAX_RANK);
    expect_refused(graph, builder.gather_rows(table, indices), nodes, "rank 1 to 5");
}

/** @brief Tests that a scatter-add writes SCATTER_ADD_ROWS over [rows, indices] as [V, D]. */
TEST(Unit_GatherOps, ScatterAddRowsWritesItsIdentityAndTheTableShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.input(Shape{2, 4, K_ROW_LENGTH}, "rows", DType::Float64);
    const ValueID indices = builder.input(Shape{2, 4}, "indices", DType::Int32);

    const Traced out = builder.scatter_add_rows(rows, indices, K_TABLE_ROWS);

    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());
    expect_produced_by(graph, out.value(), Ops::K_SCATTER_ADD_ROWS.id, {rows, indices});
    EXPECT_EQ(graph.value(out.value()).shape, (Shape{K_TABLE_ROWS, K_ROW_LENGTH}));
    EXPECT_EQ(graph.value(out.value()).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out.value()).lifetime, Graph::Lifetime::Activation);
}

/** @brief Tests each refusal of the scatter-add: its rows, its index and its row count. */
TEST(Unit_GatherOps, ScatterAddRowsRefusesOperandsThatDoNotFit)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.input(Shape{5, K_ROW_LENGTH}, "rows");
    const ValueID flat = builder.input(Shape{5}, "flat");
    const ValueID whole = builder.input(Shape{5, K_ROW_LENGTH}, "whole", DType::Int32);
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const ValueID fewer = builder.input(Shape{4}, "fewer", DType::Int32);
    const ValueID real = builder.input(Shape{5}, "real", DType::Float32);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.scatter_add_rows(flat, indices, K_TABLE_ROWS), nodes,
                   "rank 2 or more");
    expect_refused(graph, builder.scatter_add_rows(whole, indices, K_TABLE_ROWS), nodes,
                   "half, float32 or float64 rows");
    expect_refused(graph, builder.scatter_add_rows(rows, real, K_TABLE_ROWS), nodes,
                   "INT32 index");
    expect_refused(graph, builder.scatter_add_rows(rows, fewer, K_TABLE_ROWS), nodes,
                   "one index per row");
    expect_refused(graph, builder.scatter_add_rows(rows, indices, 0), nodes, "a table of 0 rows");
}

/** @brief Tests that finalise leaves an INT32 input without a gradient and marks the gather. */
TEST(Unit_GatherOps, AnInt32InputLeafRequiresNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const Traced out = builder.gather_rows(table, indices);
    ASSERT_TRUE(out.has_value());

    graph.finalise();

    EXPECT_EQ(graph.value(indices).lifetime, Graph::Lifetime::Input);
    EXPECT_FALSE(graph.value(indices).requires_grad);
    EXPECT_TRUE(graph.value(out.value()).requires_grad);
}

/**
 * @brief Tests that the cast pass under mixed precision narrows the product that reads a gather
 *        and neither casts nor retypes the INT32 leaf, the table or the gather.
 */
TEST(Unit_GatherOps, TheCastPassLeavesAnInt32LeafAndItsGatherAsDeclared)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const ValueID weight = builder.parameter(Shape{K_ROW_LENGTH, 2}, "weight");
    const Traced gathered = builder.gather_rows(table, indices);
    ASSERT_TRUE(gathered.has_value());
    const ValueID product = builder.matmul(gathered.value(), weight);

    const Graph::MixedPrecisionPolicy policy;
    Graph::CastOptions options;
    options.policy = &policy;
    const Graph::CastReport report = Graph::insert_casts(graph, options);

    EXPECT_EQ(graph.value(indices).dtype, DType::Int32);
    EXPECT_EQ(graph.value(table).dtype, DType::Float32);
    EXPECT_EQ(graph.value(gathered.value()).dtype, DType::Float32);
    EXPECT_EQ(graph.value(product).dtype, DType::Half);

    const Graph::Node& gather = graph.node(graph.value(gathered.value()).producer);
    EXPECT_EQ(gather.op, Ops::K_GATHER_ROWS.id);
    EXPECT_EQ(gather.inputs, (std::vector<ValueID>{table, indices}));

    for (const Graph::InsertedCast& cast : report.casts())
    {
        EXPECT_NE(cast.source, indices) << "the index was cast";
        EXPECT_NE(cast.op, Ops::K_GATHER_ROWS.id) << "a cast was added at the gather";
    }

    EXPECT_TRUE(Graph::find_dtype_mismatches(graph, policy).empty());
}
