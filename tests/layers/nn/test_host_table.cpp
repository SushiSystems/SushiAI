/**************************************************************************/
/* test_host_table.cpp                                                    */
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
 * @file test_host_table.cpp
 * @brief Tests the host tables a trace declares: the leaf, the list and the refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/host_table.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using NN::HostTableList;
using NN::Trace;

/** @brief Tests that a table is a buffer leaf no state binds, listed with its values. */
TEST(Unit_HostTable, ATableIsALeafListedWithItsValues)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    const std::size_t nodes_before = graph.node_count();

    SA_ASSERT_OK_AND_ASSIGN(
        const Graph::ValueID leaf,
        trace.host_table(Shape{2, 3}, "angles", DType::Float32, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));

    EXPECT_EQ(graph.node_count(), nodes_before + 1);
    EXPECT_EQ(graph.value(leaf).shape, (Shape{2, 3}));
    EXPECT_EQ(graph.value(leaf).dtype, DType::Float32);
    EXPECT_EQ(graph.value(leaf).lifetime, Graph::Lifetime::Buffer);
    EXPECT_FALSE(graph.value(leaf).requires_grad);
    EXPECT_TRUE(graph.state_leaves().empty());

    ASSERT_EQ(tables.tables().size(), 1u);
    EXPECT_EQ(tables.tables()[0].leaf, leaf);
    EXPECT_EQ(tables.tables()[0].values, (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
    SA_EXPECT_OK(trace.status());
}

/** @brief Tests that two tables of one trace are two leaves, in the order they were declared. */
TEST(Unit_HostTable, EveryDeclarationIsItsOwnLeaf)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    HostTableList tables;
    Trace trace = Trace::training(builder, NN::discard_updates());
    trace.collect_tables(tables);

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID first,
                            trace.host_table(Shape{2}, "first", DType::Float64, {1.0, 2.0}));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID second,
                            trace.host_table(Shape{2}, "second", DType::Float64, {3.0, 4.0}));

    EXPECT_NE(first, second);
    ASSERT_EQ(tables.tables().size(), 2u);
    EXPECT_EQ(tables.tables()[0].leaf, first);
    EXPECT_EQ(tables.tables()[1].leaf, second);
}

/** @brief Tests that a trace with no sink, another type and a wrong count are each refused. */
TEST(Unit_HostTable, ATableThatNoStepCanFillIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const std::size_t nodes_before = graph.node_count();

    const Trace plain = Trace::evaluation(builder);
    SA_EXPECT_FAILURE(plain.host_table(Shape{2}, "angles", DType::Float32, {1.0, 2.0}),
                      StatusCode::InvalidDescription, "'angles'", "collect_tables");

    HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    SA_EXPECT_FAILURE(trace.host_table(Shape{2}, "angles", DType::Int32, {1.0, 2.0}),
                      StatusCode::InvalidDescription, "'angles'", "float32 or a float64");
    SA_EXPECT_FAILURE(trace.host_table(Shape{2, 3}, "angles", DType::Float32, {1.0, 2.0}),
                      StatusCode::InvalidDescription, "'angles'", "6 elements", "2 values");

    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_TRUE(tables.tables().empty());
}
