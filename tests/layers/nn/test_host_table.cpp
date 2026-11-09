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
 * @file test_host_table.cpp
 * @brief Tests the host tables a trace declares: the leaf, the list and the refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/host_table.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using NN::HostTableList;
using NN::Trace;

/** @brief Tests that a table is a buffer leaf no state binds, listed with its values. */
TEST(Small_HostTable, ATableIsALeafListedWithItsValues)
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
TEST(Small_HostTable, EveryDeclarationIsItsOwnLeaf)
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

/** @brief Holds a trace, a table and the substrings of the refusal. */
struct RefusedTableRow
{
    std::string name;           /**< The row's name. */
    bool collects;              /**< Whether the trace collects its tables. */
    Shape shape;                /**< The shape of the table. */
    DType dtype;                /**< The element type of the table. */
    std::vector<double> values; /**< The values of the table. */
    std::string first;          /**< A substring the message holds. */
    std::string second;         /**< A second substring the message holds; empty for none. */
};

/** @brief Runs the table refusal over each table no step can fill. */
class Small_HostTableRefusal : public ::testing::TestWithParam<RefusedTableRow>
{
};

/** @brief Tests that a table no step can fill is refused and leaves the graph and the list alone. */
TEST_P(Small_HostTableRefusal, ATableThatNoStepCanFillIsRefused)
{
    const RefusedTableRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    if (row.collects)
        trace.collect_tables(tables);
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(trace.host_table(row.shape, "angles", row.dtype, row.values),
                      StatusCode::InvalidDescription, "'angles'", row.first, row.second);

    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_TRUE(tables.tables().empty());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_HostTableRefusal,
    ::testing::Values(
        RefusedTableRow{"NoCollector", false, Shape{2}, DType::Float32, {1.0, 2.0},
                        "collect_tables", ""},
        RefusedTableRow{"AnIntegerType", true, Shape{2}, DType::Int32, {1.0, 2.0},
                        "float32 or a float64", ""},
        RefusedTableRow{"ACountMismatch", true, Shape{2, 3}, DType::Float32, {1.0, 2.0},
                        "6 elements", "2 values"}),
    [](const ::testing::TestParamInfo<RefusedTableRow>& info) { return info.param.name; });
