/**************************************************************************/
/* test_row_index.cpp                                                     */
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
 * @file test_row_index.cpp
 * @brief Tests the gather, scatter, take along, put along and argmax lowerings on a device.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/row_index.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/graph/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::LoweringRule;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /** @brief Provides the Engine every row-index case records on. */
    class Medium_RowIndexOperations : public SaTest::EngineTest
    {

    };

    /**
     * @brief Lowers @p graph into a step over @p pool and runs it once.
     *
     * The leaves of @p graph must already be materialised in @p pool and written.
     */
    void compile_and_run(SushiAI::ExecutionBackend& backend, const Graph::Graph& graph,
                         TensorPool& pool)
    {
        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(backend, graph, pool));
        SA_EXPECT_OK(step->run());
    }

    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        OpID op;           /**< The operation. */
        LoweringRule rule; /**< The row that lowers it. */
    };

    /** @brief Runs once per row the header names for the table. */
    class Small_RowIndexRow : public ::testing::TestWithParam<ExpectedRow>
    {

    };

    /** @brief Lists the rows of the table. */
    const std::array<ExpectedRow, 5> K_EXPECTED_ROWS = {{
        {Ops::K_GATHER_ROWS.id, &Graph::lower_gather_rows},
        {Ops::K_SCATTER_ADD_ROWS.id, &Graph::lower_scatter_add_rows},
        {Ops::K_TAKE_ALONG.id, &Graph::lower_take_along},
        {Ops::K_PUT_ALONG.id, &Graph::lower_put_along},
        {Ops::K_ARGMAX_ROWS.id, &Graph::lower_argmax_rows},
    }};

    /** @brief Returns the row of @p op in the table, or nullptr when none names it. */
    const Graph::LoweringEntry* find_entry(OpID op)
    {
        for (const Graph::LoweringEntry& entry : Graph::K_ROW_INDEX_LOWERINGS)
        {
            if (entry.op == op)
                return &entry;
        }
        return nullptr;
    }

    /** @brief Holds the [4, 3] table of the values 1 to 12, one row per four-element step. */
    const std::vector<double> K_TABLE = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    /**
     * @brief Returns the column of the largest element of each row of @p values, taken on the host.
     *
     * std::max_element returns the first of equal maxima, the rule SushiBLAS states.
     */
    std::vector<double> host_argmax_rows(const std::vector<double>& values, std::size_t columns)
    {
        std::vector<double> result;
        for (std::size_t first = 0; first < values.size(); first += columns)
        {
            const auto row = values.begin() + static_cast<std::ptrdiff_t>(first);
            result.push_back(static_cast<double>(
                std::max_element(row, row + static_cast<std::ptrdiff_t>(columns)) - row));
        }
        return result;
    }

    /** @brief Expects a mixed policy to answer @p dtype to a query about the argmax. */
    void expect_argmax_keeps(DType dtype)
    {
        const Graph::MixedPrecisionPolicy policy;

        EXPECT_EQ(policy.choose(Graph::PrecisionQuery{Ops::K_ARGMAX_ROWS.id, dtype}).dtype, dtype)
            << dtype_name(dtype);
    }
}

/** @brief Tests that the table holds one row for each row the header names, and no other. */
TEST(Small_RowIndexLowerings, TheTableHoldsExactlyTheRowsTheHeaderNames)
{
    EXPECT_EQ(Graph::K_ROW_INDEX_LOWERINGS.size(), K_EXPECTED_ROWS.size());
}

/** @brief Tests that a row pairs its operation with the function the header names. */
TEST_P(Small_RowIndexRow, EachRowPairsItsOperationWithItsLowering)
{
    const ExpectedRow row = GetParam();
    const std::string name(Ops::name_of(row.op));
    const Graph::LoweringEntry* entry = find_entry(row.op);
    ASSERT_NE(entry, nullptr) << name;

    EXPECT_EQ(entry->rule, row.rule) << name;
    EXPECT_EQ(entry->scratch, nullptr) << name;
    EXPECT_EQ(Graph::find_lowering(row.op), row.rule) << name;
}

INSTANTIATE_TEST_SUITE_P(, Small_RowIndexRow, ::testing::ValuesIn(K_EXPECTED_ROWS),
                         [](const ::testing::TestParamInfo<ExpectedRow>& info)
                         {
                             std::string name(Ops::name_of(info.param.op));
                             name.replace(0, name.find('.') + 1, "");
                             return ::SushiAI::Test::parameter_name(name);
                         });

/** @brief Tests that the gather copies the rows its INT32 index names, in the index's order. */
TEST_F(Medium_RowIndexOperations, GatherRowsCopiesTheRowsTheIndexNames)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.input(Shape{4, 3}, "table");
    const ValueID index = builder.input(Shape{5}, "index", DType::Int32);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID rows, builder.gather_rows(table, index));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(table), K_TABLE);
    write_tensor(pool.tensor(index), std::vector<double>{2, 0, 2, 3, 1});

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    // Rows 2, 0, 2, 3 and 1 of the table, one after another.
    const std::vector<double> expected = {7, 8, 9, 1, 2, 3, 7, 8, 9, 10, 11, 12, 4, 5, 6};
    EXPECT_EQ(read_tensor(pool.tensor(rows)), expected);
}

/** @brief Tests that the gather reads an INT64 index as it reads an INT32 one. */
TEST_F(Medium_RowIndexOperations, GatherRowsReadsAnInt64Index)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.input(Shape{4, 3}, "table");
    const ValueID index = builder.input(Shape{5}, "index", DType::Int64);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID rows, builder.gather_rows(table, index));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(table), K_TABLE);
    write_tensor(pool.tensor(index), std::vector<double>{2, 0, 2, 3, 1});

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    const std::vector<double> expected = {7, 8, 9, 1, 2, 3, 7, 8, 9, 10, 11, 12, 4, 5, 6};
    EXPECT_EQ(read_tensor(pool.tensor(rows)), expected);
}

/** @brief Tests that the gather keeps the shape of a rank-2 index and adds the row length. */
TEST_F(Medium_RowIndexOperations, GatherRowsOverARankTwoIndexKeepsItsShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.input(Shape{4, 3}, "table");
    const ValueID index = builder.input(Shape{2, 2}, "index", DType::Int32);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID rows, builder.gather_rows(table, index));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(table), K_TABLE);
    write_tensor(pool.tensor(index), std::vector<double>{3, 3, 0, 1});

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    EXPECT_EQ(graph.value(rows).shape, (Shape{2, 2, 3}));
    const std::vector<double> expected = {10, 11, 12, 10, 11, 12, 1, 2, 3, 4, 5, 6};
    EXPECT_EQ(read_tensor(pool.tensor(rows)), expected);
}

/** @brief Tests that an index of -1 and an index at the bound each gather a row of zeros. */
TEST_F(Medium_RowIndexOperations, GatherRowsReadsZerosForAnIndexOutsideTheTable)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.input(Shape{4, 3}, "table");
    const ValueID index = builder.input(Shape{3}, "index", DType::Int32);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID rows, builder.gather_rows(table, index));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(table), K_TABLE);

    write_tensor(pool.tensor(index), std::vector<double>{-1, 4, 1});

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    const std::vector<double> expected = {0, 0, 0, 0, 0, 0, 4, 5, 6};
    EXPECT_EQ(read_tensor(pool.tensor(rows)), expected);
}

/** @brief Tests that the scatter-add sums each row into the row its index names, in order. */
TEST_F(Medium_RowIndexOperations, ScatterAddRowsSumsRepeatedIndexesAndLeavesTheOthersZero)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.input(Shape{5, 3}, "rows");
    const ValueID index = builder.input(Shape{5}, "index", DType::Int32);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID sums, builder.scatter_add_rows(rows, index, 4));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(rows), std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                        13, 14, 15});
    write_tensor(pool.tensor(index), std::vector<double>{1, 1, 3, 0, 1});

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    // Row 1 takes rows 0, 1 and 4: (1 + 4 + 13, 2 + 5 + 14, 3 + 6 + 15). Row 2 is named by none.
    const std::vector<double> expected = {10, 11, 12, 18, 21, 24, 0, 0, 0, 7, 8, 9};
    EXPECT_EQ(read_tensor(pool.tensor(sums)), expected);
}

/** @brief Tests that the scatter-add reads an INT64 index as it reads an INT32 one. */
TEST_F(Medium_RowIndexOperations, ScatterAddRowsReadsAnInt64Index)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.input(Shape{5, 3}, "rows");
    const ValueID index = builder.input(Shape{5}, "index", DType::Int64);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID sums, builder.scatter_add_rows(rows, index, 4));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(rows), std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                        13, 14, 15});
    write_tensor(pool.tensor(index), std::vector<double>{1, 1, 3, 0, 1});

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    const std::vector<double> expected = {10, 11, 12, 18, 21, 24, 0, 0, 0, 7, 8, 9};
    EXPECT_EQ(read_tensor(pool.tensor(sums)), expected);
}

/** @brief Tests that an index of -1 and an index at the bound are each skipped by the scatter-add. */
TEST_F(Medium_RowIndexOperations, ScatterAddRowsSkipsAnIndexOutsideTheTable)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.input(Shape{5, 3}, "rows");
    const ValueID index = builder.input(Shape{5}, "index", DType::Int32);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID sums, builder.scatter_add_rows(rows, index, 4));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(rows), std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                        13, 14, 15});

    // Rows 0 and 2 name -1 and the bound 4, so only rows 1, 3 and 4 reach the result.
    write_tensor(pool.tensor(index), std::vector<double>{-1, 0, 4, 2, 0});

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    const std::vector<double> expected = {17, 19, 21, 0, 0, 0, 10, 11, 12, 0, 0, 0};
    EXPECT_EQ(read_tensor(pool.tensor(sums)), expected);
}

/** @brief Tests that the pick reads the element of each row that its index names. */
TEST_F(Medium_RowIndexOperations, TakeAlongPicksOneElementPerRow)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID values = builder.input(Shape{3, 4}, "values");
    const ValueID index = builder.input(Shape{3}, "index", DType::Int32);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID picked, builder.take_along(values, index));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(values), K_TABLE);
    write_tensor(pool.tensor(index), std::vector<double>{2, 0, 3});

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    EXPECT_EQ(graph.value(picked).shape, (Shape{3}));
    EXPECT_EQ(read_tensor(pool.tensor(picked)), (std::vector<double>{3, 5, 12}));
}

/** @brief Tests that the pick reads an INT64 index, and zero for -1 and for the bound. */
TEST_F(Medium_RowIndexOperations, TakeAlongReadsZeroForAnIndexOutsideTheRow)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID values = builder.input(Shape{3, 4}, "values");
    const ValueID index = builder.input(Shape{3}, "index", DType::Int64);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID picked, builder.take_along(values, index));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(values), K_TABLE);

    write_tensor(pool.tensor(index), std::vector<double>{-1, 4, 1});

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    EXPECT_EQ(read_tensor(pool.tensor(picked)), (std::vector<double>{0, 0, 10}));
}

/** @brief Tests that the put writes each element at its column of a zeroed result. */
TEST_F(Medium_RowIndexOperations, PutAlongWritesOneElementPerRowAndZeroElsewhere)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID picked = builder.input(Shape{3}, "picked");
    const ValueID index = builder.input(Shape{3}, "index", DType::Int32);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID values, builder.put_along(picked, index, 4));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(picked), std::vector<double>{7, 8, 9});
    write_tensor(pool.tensor(index), std::vector<double>{2, 0, 3});

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    EXPECT_EQ(graph.value(values).shape, (Shape{3, 4}));
    const std::vector<double> expected = {0, 0, 7, 0, 8, 0, 0, 0, 0, 0, 0, 9};
    EXPECT_EQ(read_tensor(pool.tensor(values)), expected);
}

/** @brief Tests that the put skips an INT64 index of -1 and one at the bound. */
TEST_F(Medium_RowIndexOperations, PutAlongSkipsAnIndexOutsideTheRow)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID picked = builder.input(Shape{3}, "picked");
    const ValueID index = builder.input(Shape{3}, "index", DType::Int64);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID values, builder.put_along(picked, index, 4));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(picked), std::vector<double>{7, 8, 9});
    write_tensor(pool.tensor(index), std::vector<double>{-1, 4, 1});

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    // The first two rows are zeros; the third holds 9 at column 1.
    const std::vector<double> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0};
    EXPECT_EQ(read_tensor(pool.tensor(values)), expected);
}

/** @brief Tests that the argmax of each row equals the host argmax, ties giving the lowest column. */
TEST_F(Medium_RowIndexOperations, ArgmaxRowsEqualsTheHostArgmaxIncludingATie)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 5}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID columns, builder.argmax_rows(x));

    const std::vector<double> values = {1, 5, 2, 5, 0,
                                        -3, -1, -2, -9, -1,
                                        4, 4, 4, 4, 4,
                                        0, 1, 2, 3, 9};

    const std::vector<double> expected = host_argmax_rows(values, 5);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), values);

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    EXPECT_EQ(graph.value(columns).dtype, DType::Int64);
    EXPECT_EQ(read_tensor(pool.tensor(columns)), expected);
    EXPECT_EQ(expected, (std::vector<double>{1, 1, 0, 4}));
}

/** @brief Tests that no mixed policy narrows the argmax, whatever the type it is asked about. */
TEST(Small_RowIndexLowerings, TheArgmaxStaysInTheGraphsPrecisionUnderAMixedPolicy)
{
    EXPECT_FALSE(Graph::narrows_under_mixed_precision(Ops::K_ARGMAX_ROWS.id));
    expect_argmax_keeps(DType::Half);
    expect_argmax_keeps(DType::Float32);
    expect_argmax_keeps(DType::Float64);
}
