/**************************************************************************/
/* test_rotary_embedding.cpp                                              */
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
 * @file test_rotary_embedding.cpp
 * @brief Tests the trace, the tables, the refusals and the turn of RotaryEmbedding.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/blocks/rotary_embedding.hpp>
#include <SushiAI/language/blocks/rotary_positions.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/host_table.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/steps.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Language::Blocks::RotaryEmbedding;
using NN::Trace;
// SaTest stays reachable inside a TEST_F body; see tests/language/blocks/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the channels every case turns. */
    constexpr int64_t K_WIDTH = 8;

    /** @brief Holds the groups K_WIDTH is cut into. */
    constexpr int64_t K_GROUPS = 2;

    /** @brief Holds the pairs of a group of K_WIDTH / K_GROUPS channels. */
    constexpr int64_t K_PAIRS = 2;

    /** @brief Holds T, the positions of the sequence the window cases read. */
    constexpr int64_t K_LENGTH = 5;

    /**
     * @brief Bounds a table entry against its hand-worked cosine or sine, absolutely: 16 times
     *        the double unit roundoff, the smallest power of two not tighter than 1e-15.
     */
    constexpr double K_TOLERANCE = 16.0 * TestSupport::unit_roundoff<double>();

    /**
     * @brief Holds the rounded operations behind a turned value as a float32 run would round
     *        it: the table entry, two products and their sum.
     */
    constexpr std::size_t K_TURN_OPERATIONS = 4;

    /**
     * @brief Holds the rounded operations behind a score: the four turns, four products and
     *        three sums, each of the turns counted at its K_TURN_OPERATIONS.
     */
    constexpr std::size_t K_SCORE_OPERATIONS = 4 * K_TURN_OPERATIONS + 7;

    /** @brief Holds the channels of a group in the turn cases: two pairs. */
    constexpr std::size_t K_GROUP = 4;

    /** @brief Holds a query whose values a float32 holds exactly. */
    const std::vector<double> K_QUERY{1.0, 2.0, 3.0, 4.0};

    /** @brief Holds a key whose values a float32 holds exactly. */
    const std::vector<double> K_KEY{0.5, -1.0, 2.0, 0.25};

    /** @brief Returns @p rows copies of @p row, one after the other. */
    std::vector<double> repeated(const std::vector<double>& row, std::size_t rows)
    {
        std::vector<double> values;
        values.reserve(rows * row.size());
        for (std::size_t at = 0; at < rows; ++at)
            values.insert(values.end(), row.begin(), row.end());

        return values;
    }

    /**
     * @brief Traces a RotaryEmbedding of one group over [1, rows, 4], writes its tables and
     *        @p input, and evaluates it on the host.
     * @return The turned rows, or the failure of the trace or of the evaluator.
     */
    Result<std::vector<double>> turn_on_host(std::size_t rows, const std::vector<double>& input)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(
            Shape{1, static_cast<int64_t>(rows), static_cast<int64_t>(K_GROUP)}, "x");
        NN::HostTableList tables;
        Trace trace = Trace::evaluation(builder);
        trace.collect_tables(tables);
        RotaryEmbedding block(static_cast<int64_t>(K_GROUP), 1);
        const Graph::ValueID y = block.forward(trace, x);
        SA_TRY(trace.status());

        Graph::HostEvaluator evaluator(graph);
        for (const NN::HostTable& table : tables.tables())
            SA_TRY(evaluator.write(table.leaf, table.values));

        SA_TRY(evaluator.write(x, input));
        SA_TRY(evaluator.run());
        return evaluator.read(y);
    }

    /** @brief Returns the product of row @p query of @p queries with row @p key of @p keys. */
    double score(const std::vector<double>& queries, std::size_t query,
                 const std::vector<double>& keys, std::size_t key)
    {
        double sum = 0.0;
        for (std::size_t channel = 0; channel < K_GROUP; ++channel)
            sum += queries[query * K_GROUP + channel] * keys[key * K_GROUP + channel];

        return sum;
    }

    /** @brief Holds one table entry worked by hand: the angle is row times 1 or 0.01. */
    struct TableRow
    {
        std::string name;  /**< The row's name. */
        std::size_t at;    /**< The entry, row * K_PAIRS + pair. */
        double cosine;     /**< The cosine of the angle. */
        double sine;       /**< The sine of the angle. */
    };

    /** @brief Runs one table entry. */
    class Small_BlockRotaryEmbeddingTable : public ::testing::TestWithParam<TableRow>
    {

    };

    /** @brief Holds one pair of widths the width check refuses and its message. */
    struct WidthRow
    {
        std::string name;   /**< The row's name. */
        int64_t width;      /**< The width. */
        int64_t groups;     /**< The groups. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A second word of the message, or empty. */
    };

    /** @brief Runs one refused pair of widths. */
    class Small_BlockRotaryEmbeddingWidths : public ::testing::TestWithParam<WidthRow>
    {

    };

    /** @brief Holds one turned element worked by hand from the turn of its pair. */
    struct TurnRow
    {
        std::string name;  /**< The row's name. */
        std::size_t at;    /**< The element of the three turned rows. */
        double expected;   /**< The hand-worked value. */
    };

    /** @brief Runs one turned element. */
    class Small_RotaryEmbeddingTurn : public ::testing::TestWithParam<TurnRow>
    {

    };

    /** @brief Names a row by its name field. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that a trace keeps the shape and collects a table of cosines and of sines. */
TEST(Small_BlockRotaryEmbedding, ATraceWeighsTheValueAndItsSwappedHalves)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, K_LENGTH, K_WIDTH}, "x");
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);

    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, K_LENGTH, K_WIDTH}));
    EXPECT_EQ(tables.tables().size(), 2u);
}

/** @brief Tests that the two tables are float32 leaves of a row of pairs per position. */
TEST(Small_BlockRotaryEmbedding, TheTablesAreLeavesOfARowPerPosition)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);

    static_cast<void>(block.forward(trace, builder.input(Shape{2, K_LENGTH, K_WIDTH}, "x")));

    SA_EXPECT_OK(trace.status());
    ASSERT_EQ(tables.tables().size(), 2u);
    const NN::HostTable& cosines = tables.tables()[0];
    const NN::HostTable& sines = tables.tables()[1];
    EXPECT_EQ(graph.value(cosines.leaf).shape, (Shape{K_LENGTH, K_PAIRS}));
    EXPECT_EQ(graph.value(sines.leaf).shape, (Shape{K_LENGTH, K_PAIRS}));
    EXPECT_EQ(graph.value(cosines.leaf).dtype, DType::Float32);
    EXPECT_EQ(cosines.values.size(), static_cast<std::size_t>(K_LENGTH * K_PAIRS));
    EXPECT_EQ(sines.values.size(), cosines.values.size());
}

/** @brief Tests that a table entry holds the cosine and the sine of its hand-worked angle. */
TEST_P(Small_BlockRotaryEmbeddingTable, TheTablesHoldTheAnglesOfEveryPosition)
{
    const TableRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);

    static_cast<void>(block.forward(trace, builder.input(Shape{2, K_LENGTH, K_WIDTH}, "x")));

    ASSERT_EQ(tables.tables().size(), 2u);
    ASSERT_EQ(tables.tables()[0].values.size(), static_cast<std::size_t>(K_LENGTH * K_PAIRS));
    EXPECT_NEAR(tables.tables()[0].values[row.at], row.cosine, K_TOLERANCE);
    EXPECT_NEAR(tables.tables()[1].values[row.at], row.sine, K_TOLERANCE);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockRotaryEmbeddingTable,
    ::testing::Values(TableRow{"PositionZero", 0, 1.0, 0.0},
                      TableRow{"PositionOnePairZero", 2, 0.5403023058681398,
                               0.8414709848078965},
                      TableRow{"PositionTwoPairZero", 4, -0.4161468365471424,
                               0.9092974268256817},
                      TableRow{"PositionFourPairOne", 9, 0.9992001066609779,
                               0.03998933418663416}),
    row_name<TableRow>);

/** @brief Tests that a float64 value reads float64 tables. */
TEST(Small_BlockRotaryEmbedding, ADoubleValueReadsDoubleTables)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x =
        builder.input(Shape{1, K_LENGTH, K_WIDTH}, "x", DType::Float64);
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);

    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).dtype, DType::Float64);
    ASSERT_EQ(tables.tables().size(), 2u);
    EXPECT_EQ(graph.value(tables.tables()[0].leaf).dtype, DType::Float64);
    EXPECT_EQ(graph.value(tables.tables()[1].leaf).dtype, DType::Float64);
}

/** @brief Tests that the block lists no parameter and no buffer. */
TEST(Small_BlockRotaryEmbedding, TheBlockHoldsNoState)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    static_cast<void>(block.forward(trace, builder.input(Shape{2, K_LENGTH, K_WIDTH}, "x")));

    std::vector<std::string> names;
    const auto on_parameter = [&names](const std::string& name, NN::Parameter&)
    {
        names.push_back(name);
    };
    const auto on_buffer = [&names](const std::string& name, NN::Buffer&)
    {
        names.push_back(name);
    };
    const NN::StateVisitor visitor(on_parameter, on_buffer);
    NN::walk_state(block, "rotary", visitor);

    EXPECT_TRUE(names.empty());
    EXPECT_TRUE(graph.state_leaves().empty());
    EXPECT_EQ(block.output_shape(Shape{K_LENGTH, K_WIDTH}).value(), (Shape{K_LENGTH, K_WIDTH}));
}

/** @brief Tests that one position is fed one row of cosines and one of sines, and no table. */
TEST(Small_BlockRotaryEmbedding, TheIncrementalFormIsFedTheAnglesOfItsPosition)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_WIDTH}, "x");
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y = block.forward_incremental(trace, x, Shape{K_LENGTH, K_WIDTH});

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_WIDTH}));
    EXPECT_TRUE(carried.values().empty());
    ASSERT_EQ(fed.leaves().size(), 2u);
    EXPECT_EQ(fed.leaves()[0].name, Language::Blocks::K_ROTARY_COSINE_LEAF);
    EXPECT_EQ(fed.leaves()[1].name, Language::Blocks::K_ROTARY_SINE_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[0].value).shape, (Shape{1, K_PAIRS}));
    EXPECT_EQ(graph.value(fed.leaves()[1].value).shape, (Shape{1, K_PAIRS}));
    EXPECT_EQ(graph.value(fed.leaves()[0].value).dtype, DType::Float32);
    EXPECT_EQ(graph.value(fed.leaves()[1].value).dtype, DType::Float32);
}

/** @brief Tests that the width check accepts a width of whole pairs in every group. */
TEST(Small_BlockRotaryEmbedding, TheWidthCheckAcceptsWholePairs)
{
    SA_EXPECT_OK(Language::Blocks::check_rotary_widths(K_WIDTH, K_GROUPS));
}

/** @brief Tests that the width check refuses widths that leave no whole pair. */
TEST_P(Small_BlockRotaryEmbeddingWidths, TheWidthCheckRefusesWidthsWithoutWholePairs)
{
    const WidthRow& row = GetParam();

    SA_EXPECT_FAILURE(Language::Blocks::check_rotary_widths(row.width, row.groups),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockRotaryEmbeddingWidths,
    ::testing::Values(WidthRow{"OddGroup", 6, 2, "pairs", "width=6 groups=2"},
                      WidthRow{"UnevenGroups", 8, 3, "width=8 groups=3", ""},
                      WidthRow{"NoWidth", 0, 0, "width=0 groups=0", ""}),
    row_name<WidthRow>);

/** @brief Tests that an input of another width fails and collects no table. */
TEST(Small_BlockRotaryEmbedding, AnInputOfAnotherWidthIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, K_LENGTH, K_WIDTH + 2}, "x");
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);

    const Graph::ValueID y = block.forward(trace, x);

    EXPECT_EQ(y, x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "RotaryEmbedding",
                      "[2, 5, 10]");
    EXPECT_TRUE(tables.tables().empty());
}

/** @brief Tests that a trace that collects no tables is refused, naming the cosine table. */
TEST(Small_BlockRotaryEmbedding, ATraceWithoutTablesIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, K_LENGTH, K_WIDTH}, "x");
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    const Trace trace = Trace::evaluation(builder);

    const Graph::ValueID y = block.forward(trace, x);

    EXPECT_EQ(y, x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "'rotary_cosine'",
                      "collect_tables");
}

/** @brief Tests that an incremental trace that carries nothing is refused. */
TEST(Small_BlockRotaryEmbedding, ANonCarryingIncrementalTraceIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_WIDTH}, "x");
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    const Trace trace = Trace::evaluation(builder);

    const Graph::ValueID y = block.forward_incremental(trace, x, Shape{K_LENGTH, K_WIDTH});

    EXPECT_EQ(y, x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "'rotary_cosine'",
                      "carrying trace");
}

/**
 * @brief Tests each turned element of three rows against its hand-worked value.
 *
 * The rows are (1, 2, 3, 4), (1, -1, 3, 0.25) and (-3, 1, 0, 2); pair 0 of row r turns by r
 * radians and pair 1 by r / 100, so pair 0 of row 1 is (cos 1 - 3 sin 1, sin 1 + 3 cos 1).
 */
TEST_P(Small_RotaryEmbeddingTurn, ATurnMatchesTheHostFormula)
{
    const TurnRow& row = GetParam();
    const std::vector<double> input{1.0, 2.0, 3.0, 4.0, 1.0, -1.0, 3.0, 0.25,
                                    -3.0, 1.0, 0.0, 2.0};

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> turned, turn_on_host(3, input));

    ASSERT_EQ(turned.size(), input.size());
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
        turned[row.at], row.expected, K_TURN_OPERATIONS)));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_RotaryEmbeddingTurn,
    ::testing::Values(TurnRow{"RowZeroPairZeroA", 0, 1.0}, TurnRow{"RowZeroPairZeroB", 2, 3.0},
                      TurnRow{"RowOnePairZeroA", 4, -1.9841106485555495},
                      TurnRow{"RowOnePairZeroB", 6, 2.4623779024123156},
                      TurnRow{"RowOnePairOneA", 5, -1.002449958750207},
                      TurnRow{"RowOnePairOneB", 7, 0.23998766676999966},
                      TurnRow{"RowTwoPairZeroA", 8, 1.2484405096414273},
                      TurnRow{"RowTwoPairZeroB", 10, -2.727892280477045},
                      TurnRow{"RowTwoPairOneA", 9, 0.9598026732799116},
                      TurnRow{"RowTwoPairOneB", 11, 2.0195986800264887}),
    row_name<TurnRow>);

/**
 * @brief Tests that the score of a turned query and key depends on their distance alone.
 *
 * Two positions apart, the query leads by 2 and 0.02 radians, so the score is
 * sum over pairs of (ac + bd) cos t + (ad - bc) sin t = -3.1601117306901636.
 */
TEST(Small_RotaryEmbedding, AScoreDependsOnTheDistanceOfThePositionsAlone)
{
    const std::size_t rows = 6;
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> queries,
                            turn_on_host(rows, repeated(K_QUERY, rows)));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> keys,
                            turn_on_host(rows, repeated(K_KEY, rows)));

    const double two_apart = score(queries, 2, keys, 0);

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
        two_apart, -3.1601117306901636, K_SCORE_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
        score(queries, 3, keys, 1), two_apart, K_SCORE_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
        score(queries, 4, keys, 2), two_apart, K_SCORE_OPERATIONS)));
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
        score(queries, 5, keys, 3), two_apart, K_SCORE_OPERATIONS)));
    EXPECT_GT(std::abs(score(queries, 3, keys, 0) - two_apart), 1.0);
}

namespace
{
    /** @brief Provides the Engine the compiled-step cases record on. */
    class Medium_RotaryEmbedding : public SaTest::EngineTest
    {

    };

    /**
     * @brief Traces a RotaryEmbedding of @p groups groups over [1, rows, width], fills its
     *        tables, runs it over @p input and writes the result to @p turned.
     */
    void turn_on_engine(SushiAI::ExecutionBackend& backend, std::size_t rows, std::size_t groups,
                        const std::vector<double>& input, std::vector<double>& turned)
    {
        const std::int64_t width = static_cast<std::int64_t>(groups * K_GROUP);
        ASSERT_EQ(input.size(), rows * groups * K_GROUP);

        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x =
            builder.input(Shape{1, static_cast<std::int64_t>(rows), width}, "x");

        NN::HostTableList tables;
        NN::Trace trace = NN::Trace::evaluation(builder);
        trace.collect_tables(tables);
        RotaryEmbedding block(width, static_cast<std::int64_t>(groups));
        const Graph::ValueID y = block.forward(trace, x);
        SA_EXPECT_OK(trace.status());
        ASSERT_FALSE(graph.has_refusal());

        Graph::TensorPool pool(backend.engine());
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        SA_EXPECT_OK(Train::fill_host_tables(tables, pool));
        SaTest::write_tensor(pool.tensor(x), input);

        std::unique_ptr<Graph::CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, Graph::CompiledStep::compile(backend, graph, pool));
        SA_EXPECT_OK(step->run());
        turned = SaTest::read_tensor(pool.tensor(y));
    }

    /** @brief Returns the channels of group @p group of every row of two groups. */
    std::vector<double> group_of(const std::vector<double>& turned, std::size_t group)
    {
        std::vector<double> channels;
        for (std::size_t start = group * K_GROUP; start < turned.size(); start += 2 * K_GROUP)
            channels.insert(channels.end(), turned.begin() + static_cast<std::ptrdiff_t>(start),
                            turned.begin() + static_cast<std::ptrdiff_t>(start + K_GROUP));

        return channels;
    }
} // namespace

/** @brief Tests that position 0 turns nothing: its row leaves the compiled step as it entered. */
TEST_F(Medium_RotaryEmbedding, PositionZeroIsTheIdentity)
{
    std::vector<double> turned;

    turn_on_engine(backend(), 2, 1, repeated(K_QUERY, 2), turned);

    ASSERT_EQ(turned.size(), 2 * K_GROUP);
    EXPECT_EQ(std::vector<double>(turned.begin(), turned.begin() + K_GROUP), K_QUERY);
    EXPECT_GT(std::abs(turned[K_GROUP] - K_QUERY[0]), 1.0);
}

/** @brief Tests that two groups of one row holding the same values turn to the same values. */
TEST_F(Medium_RotaryEmbedding, EveryGroupTurnsByTheSameAngles)
{
    const std::size_t rows = 3;
    std::vector<double> turned;

    turn_on_engine(backend(), rows, 2, repeated(K_KEY, 2 * rows), turned);

    ASSERT_EQ(turned.size(), 2 * rows * K_GROUP);
    EXPECT_EQ(group_of(turned, 0), group_of(turned, 1));
}
