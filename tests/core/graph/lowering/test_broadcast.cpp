/**************************************************************************/
/* test_broadcast.cpp                                                     */
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
 * @file test_broadcast.cpp
 * @brief Tests the broadcast lowering table, the four broadcast operations and the sum along.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/broadcast.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Graph::BroadcastSide;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::LoweringRule;
using Graph::TensorPool;
using Graph::ValueID;

namespace
{
    /** @brief Provides the Engine every broadcast case records on. */
    class Medium_BroadcastLowerings : public SaTest::EngineTest
    {

    };

    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        std::size_t index; /**< The position of the row in the table. */
        OpID op;           /**< The operation. */
        LoweringRule rule; /**< The row that lowers it. */
    };

    /** @brief Runs once per row the header names for the broadcast table. */
    class Small_BroadcastRow : public ::testing::TestWithParam<ExpectedRow>
    {

    };

    /** @brief Lists the rows of the broadcast table in the order the header writes them. */
    const std::array<ExpectedRow, 5> K_EXPECTED_ROWS = {{
        {0, Ops::K_BROADCAST_ADD.id, &Graph::lower_broadcast_add},
        {1, Ops::K_BROADCAST_SUB.id, &Graph::lower_broadcast_sub},
        {2, Ops::K_BROADCAST_MUL.id, &Graph::lower_broadcast_mul},
        {3, Ops::K_BROADCAST_DIV.id, &Graph::lower_broadcast_div},
        {4, Ops::K_SUM_ALONG.id, &Graph::lower_sum_along},
    }};

    /** @brief The four broadcast operations. */
    enum class Kind : std::uint8_t
    {
        Add,
        Sub,
        Mul,
        Div,
    };

    /** @brief Names one operation on one side with the values its [2, 3] result must hold. */
    struct HandCase
    {
        const char* name;             /**< The case name. */
        Kind kind;                    /**< The operation. */
        BroadcastSide side;           /**< The side the vector sits on. */
        std::vector<double> vector;   /**< The vector: [2] on the leading side, [3] on the other. */
        std::vector<double> expected; /**< The [2, 3] result over the values 1 to 6. */
    };

    /** @brief Runs once per operation and side. */
    class Medium_BroadcastHandValues : public SaTest::EngineTest,
                                       public ::testing::WithParamInterface<HandCase>
    {

    };

    /** @brief Emits the operation @p kind of @p x and @p vector at @p side. */
    Graph::Traced emit(const GraphBuilder& builder, Kind kind, ValueID x, ValueID vector,
                       BroadcastSide side)
    {
        switch (kind)
        {
            case Kind::Add: return builder.broadcast_add(x, vector, side);
            case Kind::Sub: return builder.broadcast_sub(x, vector, side);
            case Kind::Mul: return builder.broadcast_mul(x, vector, side);
            case Kind::Div: return builder.broadcast_div(x, vector, side);
        }

        return builder.broadcast_add(x, vector, side);
    }

    /** @brief Holds the values 1 to @p count as a tensor's elements. */
    std::vector<double> counting(std::size_t count)
    {
        std::vector<double> values(count);
        for (std::size_t i = 0; i < count; ++i)
            values[i] = static_cast<double>(i + 1);
        return values;
    }

    /** @brief Returns @p count values that FLOAT32 cannot hold exactly: 0.1 i + 1/3. */
    std::vector<double> inexact_values(std::size_t count)
    {
        std::vector<double> values(count);
        for (std::size_t i = 0; i < count; ++i)
            values[i] = 0.1 * static_cast<double>(i) + 1.0 / 3.0;
        return values;
    }

    /** @brief Returns @p count position values that FLOAT32 cannot hold exactly: 0.7 / (i + 1). */
    std::vector<double> inexact_positions(std::size_t count)
    {
        std::vector<double> values(count);
        for (std::size_t i = 0; i < count; ++i)
            values[i] = 0.7 / (static_cast<double>(i) + 1.0);
        return values;
    }

    /** @brief Compiles @p graph over @p pool, whose leaves are written, and runs the step once. */
    void run_once(ExecutionBackend& backend, const Graph::Graph& graph, TensorPool& pool)
    {
        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(backend, graph, pool));
        SA_EXPECT_OK(step->run());
    }
} // namespace

/** @brief Tests that the table holds one row for each row the header names, and no other. */
TEST(Small_BroadcastLowerings, TheTableHoldsExactlyTheRowsTheHeaderNames)
{
    EXPECT_EQ(Graph::K_BROADCAST_LOWERINGS.size(), K_EXPECTED_ROWS.size());
}

/** @brief Tests that a row pairs its operation with the function the header names. */
TEST_P(Small_BroadcastRow, EachRowPairsItsOperationWithItsLowering)
{
    const ExpectedRow row = GetParam();
    const std::string name(Ops::name_of(row.op));
    ASSERT_LT(row.index, Graph::K_BROADCAST_LOWERINGS.size());
    const Graph::LoweringEntry& entry = Graph::K_BROADCAST_LOWERINGS[row.index];

    EXPECT_EQ(entry.op, row.op) << name;
    EXPECT_EQ(entry.rule, row.rule) << name;
    EXPECT_EQ(entry.scratch, nullptr) << name;
    EXPECT_EQ(Graph::find_lowering(row.op), row.rule) << name;
}

INSTANTIATE_TEST_SUITE_P(, Small_BroadcastRow, ::testing::ValuesIn(K_EXPECTED_ROWS),
                         [](const ::testing::TestParamInfo<ExpectedRow>& info)
                         {
                             std::string name(Ops::name_of(info.param.op));
                             name.replace(0, name.find('.') + 1, "");
                             return ::SushiAI::Test::parameter_name(name);
                         });

/** @brief Tests that each operation on each side writes the hand values over a [2, 3] tensor. */
TEST_P(Medium_BroadcastHandValues, EachOperationOnEachSideWritesTheHandValues)
{
    const HandCase& row = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3}, "x");
    const bool leading = row.side == BroadcastSide::Leading;
    const ValueID vector = builder.input(leading ? Shape{2} : Shape{3}, "vector");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, emit(builder, row.kind, x, vector, row.side));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(x), counting(6));
    SaTest::write_tensor(pool.tensor(vector), row.vector);

    ASSERT_NO_FATAL_FAILURE(run_once(backend(), graph, pool));

    // The vectors hold powers of two, so every quotient and product is exact in FLOAT32.
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 3}));
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(y)), row.expected);
}

INSTANTIATE_TEST_SUITE_P(
    Cases, Medium_BroadcastHandValues,
    ::testing::Values(
        HandCase{"AddLeading", Kind::Add, BroadcastSide::Leading, {2, 4},
                 {3, 4, 5, 8, 9, 10}},
        HandCase{"SubLeading", Kind::Sub, BroadcastSide::Leading, {2, 4},
                 {-1, 0, 1, 0, 1, 2}},
        HandCase{"MulLeading", Kind::Mul, BroadcastSide::Leading, {2, 4},
                 {2, 4, 6, 16, 20, 24}},
        HandCase{"DivLeading", Kind::Div, BroadcastSide::Leading, {2, 4},
                 {0.5, 1, 1.5, 1, 1.25, 1.5}},
        HandCase{"AddTrailing", Kind::Add, BroadcastSide::Trailing, {1, 2, 4},
                 {2, 4, 7, 5, 7, 10}},
        HandCase{"SubTrailing", Kind::Sub, BroadcastSide::Trailing, {1, 2, 4},
                 {0, 0, -1, 3, 3, 2}},
        HandCase{"MulTrailing", Kind::Mul, BroadcastSide::Trailing, {1, 2, 4},
                 {1, 4, 12, 4, 10, 24}},
        HandCase{"DivTrailing", Kind::Div, BroadcastSide::Trailing, {1, 2, 4},
                 {1, 1, 0.75, 4, 2.5, 1.5}}),
    [](const ::testing::TestParamInfo<HandCase>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

/** @brief Tests that a rank-3 tensor takes a leading vector of two axes and a trailing one. */
TEST_F(Medium_BroadcastLowerings, ARankThreeTensorTakesALeadingAndATrailingVector)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 2, 3}, "x");
    const ValueID leading = builder.input(Shape{2, 2}, "leading");
    const ValueID trailing = builder.input(Shape{2, 3}, "trailing");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID by_rows,
                            builder.broadcast_mul(x, leading, BroadcastSide::Leading));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID by_columns,
                            builder.broadcast_add(x, trailing, BroadcastSide::Trailing));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(x), counting(12));
    SaTest::write_tensor(pool.tensor(leading), std::vector<double>{1, 2, 3, 4});
    SaTest::write_tensor(pool.tensor(trailing), std::vector<double>{10, 20, 30, 40, 50, 60});

    ASSERT_NO_FATAL_FAILURE(run_once(backend(), graph, pool));

    const std::vector<double> scaled = {1, 2, 3, 8, 10, 12, 21, 24, 27, 40, 44, 48};
    const std::vector<double> shifted = {11, 22, 33, 44, 55, 66, 17, 28, 39, 50, 61, 72};
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(by_rows)), scaled);
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(by_columns)), shifted);
}

/** @brief Tests that the sum along sums each side of a rank-2 and a rank-3 tensor. */
TEST_F(Medium_BroadcastLowerings, SumAlongSumsDownToAPrefixAndToASuffix)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID matrix = builder.input(Shape{2, 3}, "matrix");
    const ValueID cube = builder.input(Shape{2, 2, 3}, "cube");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID rows,
                            builder.sum_along(matrix, Shape{2}, BroadcastSide::Leading));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID columns,
                            builder.sum_along(matrix, Shape{3}, BroadcastSide::Trailing));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID pairs,
                            builder.sum_along(cube, Shape{2, 2}, BroadcastSide::Leading));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID planes,
                            builder.sum_along(cube, Shape{2, 3}, BroadcastSide::Trailing));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID first_axis,
                            builder.sum_along(cube, Shape{2}, BroadcastSide::Leading));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID last_axis,
                            builder.sum_along(cube, Shape{3}, BroadcastSide::Trailing));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(matrix), counting(6));
    SaTest::write_tensor(pool.tensor(cube), counting(12));

    ASSERT_NO_FATAL_FAILURE(run_once(backend(), graph, pool));

    EXPECT_EQ(SaTest::read_tensor(pool.tensor(rows)), (std::vector<double>{6, 15}));
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(columns)), (std::vector<double>{5, 7, 9}));
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(pairs)), (std::vector<double>{6, 15, 24, 33}));
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(planes)),
              (std::vector<double>{8, 10, 12, 14, 16, 18}));
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(first_axis)), (std::vector<double>{21, 57}));
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(last_axis)), (std::vector<double>{22, 26, 30}));
}

/**
 * @brief Tests that a position table added by broadcast_add equals the reshape form bit for bit.
 *
 * The reshape form views the [B, T, D] value as [B, T * D] rows and adds the table as a
 * [T * D] bias, which is how PositionEmbedding adds it today.
 */
TEST_F(Medium_BroadcastLowerings, APositionTableThroughBroadcastAddEqualsTheReshapeForm)
{
    constexpr std::int64_t K_BATCH = 2;
    constexpr std::int64_t K_LENGTH = 3;
    constexpr std::int64_t K_WIDTH = 4;
    constexpr std::size_t K_ELEMENTS = K_BATCH * K_LENGTH * K_WIDTH;
    constexpr std::size_t K_TABLE_ELEMENTS = K_LENGTH * K_WIDTH;

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{K_BATCH, K_LENGTH, K_WIDTH}, "x");
    const ValueID table = builder.input(Shape{K_LENGTH, K_WIDTH}, "table");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID broadcast,
                            builder.broadcast_add(x, table, BroadcastSide::Trailing));

    SA_ASSERT_OK_AND_ASSIGN(const ValueID rows,
                            builder.reshape(x, Shape{K_BATCH, K_LENGTH * K_WIDTH}));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID flat_table,
                            builder.reshape(table, Shape{K_LENGTH * K_WIDTH}));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID biased, builder.bias_add(rows, flat_table));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(x), inexact_values(K_ELEMENTS));
    SaTest::write_tensor(pool.tensor(table), inexact_positions(K_TABLE_ELEMENTS));

    ASSERT_NO_FATAL_FAILURE(run_once(backend(), graph, pool));

    EXPECT_EQ(SaTest::read_tensor_bytes(pool.tensor(broadcast)),
              SaTest::read_tensor_bytes(pool.tensor(biased)));
}

/** @brief Tests that a broadcast refuses a vector that is not the side it is declared on. */
TEST(Small_BroadcastLowerings, ABroadcastRefusesAVectorThatIsNotThatSideOfTheTensor)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3}, "x");
    const ValueID rows = builder.input(Shape{2}, "rows");
    const ValueID columns = builder.input(Shape{3}, "columns");
    const std::size_t nodes = graph.node_count();

    SA_EXPECT_FAILURE(builder.broadcast_add(x, rows, BroadcastSide::Trailing),
                      StatusCode::InvalidDescription, "a suffix");
    SA_EXPECT_FAILURE(builder.broadcast_div(x, columns, BroadcastSide::Leading),
                      StatusCode::InvalidDescription, "a prefix");
    SA_EXPECT_FAILURE(builder.sum_along(x, Shape{3}, BroadcastSide::Leading),
                      StatusCode::InvalidDescription, "a prefix");
    EXPECT_EQ(graph.node_count(), nodes);
}
