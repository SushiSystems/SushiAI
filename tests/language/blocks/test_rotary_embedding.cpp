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
 * @file test_rotary_embedding.cpp
 * @brief Tests RotaryEmbedding: its trace and the values it computes on an engine against the host
 *        formula.
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
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
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

using namespace SushiAI;
using Language::Blocks::RotaryEmbedding;
using NN::Trace;
// SaTest stays reachable inside a TEST_F body; see tests/language/blocks/README.md.
namespace SaTest = SushiAI::Test;
using Language::Blocks::rotary_angle;

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

    /** @brief Holds the nodes a trace adds besides the two leaves of its angles. */
    constexpr std::size_t K_TURN_NODES = 18;

    /** @brief Bounds a table entry against the host's cosine or sine, absolutely. */
    constexpr double K_TOLERANCE = 1e-15;

    /** @brief Counts the nodes of @p graph that carry @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                ++count;
        }

        return count;
    }
} // namespace

/** @brief Tests that a trace adds two table leaves and eighteen nodes, two of them affine. */
TEST(Medium_BlockRotaryEmbedding, ATraceWeighsTheValueAndItsSwappedHalves)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, K_LENGTH, K_WIDTH}, "x");
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    const std::size_t nodes_before = graph.node_count();

    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, K_LENGTH, K_WIDTH}));
    EXPECT_EQ(graph.node_count(), nodes_before + 2 + K_TURN_NODES);
    EXPECT_EQ(count_nodes(graph, Ops::K_CHANNEL_AFFINE.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SLICE_CHANNELS.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 3u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SCALE.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);

    // A weight per position and channel stands as one channel of a [N, 1, 1, T * C] view.
    for (const Graph::Node& node : graph.nodes())
    {
        if (node.op != Ops::K_CHANNEL_AFFINE.id)
            continue;

        EXPECT_EQ(graph.value(node.inputs[0]).shape, (Shape{2, 1, 1, K_LENGTH * K_WIDTH}));
        EXPECT_EQ(graph.value(node.inputs[1]).shape, (Shape{K_LENGTH * K_WIDTH}));
        EXPECT_FALSE(graph.value(node.inputs[1]).requires_grad);
    }
}

/** @brief Tests that the two tables hold the cosine and the sine of every row and pair. */
TEST(Medium_BlockRotaryEmbedding, TheTablesHoldTheAnglesOfEveryPosition)
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
    ASSERT_EQ(cosines.values.size(), static_cast<std::size_t>(K_LENGTH * K_PAIRS));
    ASSERT_EQ(sines.values.size(), cosines.values.size());

    for (int64_t row = 0; row < K_LENGTH; ++row)
    {
        for (int64_t pair = 0; pair < K_PAIRS; ++pair)
        {
            const double angle = Language::Blocks::rotary_angle(row, pair, K_PAIRS);
            const std::size_t at = static_cast<std::size_t>(row * K_PAIRS + pair);
            EXPECT_NEAR(cosines.values[at], std::cos(angle), K_TOLERANCE) << "element " << at;
            EXPECT_NEAR(sines.values[at], std::sin(angle), K_TOLERANCE) << "element " << at;
        }
    }
}

/** @brief Tests that a float64 value reads float64 tables and adds no cast. */
TEST(Medium_BlockRotaryEmbedding, ADoubleValueReadsDoubleTables)
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
    EXPECT_EQ(count_nodes(graph, Ops::K_CAST.id), 0u);
}

/** @brief Tests that the block lists no parameter and no buffer. */
TEST(Medium_BlockRotaryEmbedding, TheBlockHoldsNoState)
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
    EXPECT_EQ(block.output_shape(Shape{K_LENGTH, K_WIDTH}), (Shape{K_LENGTH, K_WIDTH}));
}

/** @brief Tests that one position is fed one row of cosines and one of sines, and no table. */
TEST(Medium_BlockRotaryEmbedding, TheIncrementalFormIsFedTheAnglesOfItsPosition)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_WIDTH}, "x");
    RotaryEmbedding block(K_WIDTH, K_GROUPS);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y =
        block.forward_incremental(trace, x, Shape{K_LENGTH, K_WIDTH});

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_WIDTH}));
    EXPECT_EQ(graph.node_count(), nodes_before + 2 + K_TURN_NODES);
    EXPECT_TRUE(carried.values().empty());

    ASSERT_EQ(fed.leaves().size(), 2u);
    EXPECT_EQ(fed.leaves()[0].name, Language::Blocks::K_ROTARY_COSINE_LEAF);
    EXPECT_EQ(fed.leaves()[1].name, Language::Blocks::K_ROTARY_SINE_LEAF);
    for (const NN::FedLeaf& leaf : fed.leaves())
    {
        EXPECT_EQ(graph.value(leaf.value).shape, (Shape{1, K_PAIRS}));
        EXPECT_EQ(graph.value(leaf.value).dtype, DType::Float32);
    }
}

/** @brief Tests that widths that leave no pair, another input and a plain trace are refused. */
TEST(Medium_BlockRotaryEmbedding, ABlockOrAnInputThatDoesNotFitIsRefused)
{
    SA_EXPECT_OK(Language::Blocks::check_rotary_widths(K_WIDTH, K_GROUPS));
    SA_EXPECT_FAILURE(Language::Blocks::check_rotary_widths(6, 2),
                      StatusCode::InvalidDescription, "pairs", "width=6 groups=2");
    SA_EXPECT_FAILURE(Language::Blocks::check_rotary_widths(8, 3),
                      StatusCode::InvalidDescription, "width=8 groups=3");
    SA_EXPECT_FAILURE(Language::Blocks::check_rotary_widths(0, 0),
                      StatusCode::InvalidDescription, "width=0 groups=0");

    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, K_LENGTH, K_WIDTH + 2}, "x");
        RotaryEmbedding block(K_WIDTH, K_GROUPS);
        NN::HostTableList tables;
        Trace trace = Trace::evaluation(builder);
        trace.collect_tables(tables);

        EXPECT_EQ(block.forward(trace, x), x);
        SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "RotaryEmbedding",
                          "[2, 5, 10]");
        EXPECT_TRUE(tables.tables().empty());
    }
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, K_LENGTH, K_WIDTH}, "x");
        RotaryEmbedding block(K_WIDTH, K_GROUPS);
        const Trace trace = Trace::evaluation(builder);

        EXPECT_EQ(block.forward(trace, x), x);
        SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "'rotary_cosine'",
                          "collect_tables");
    }
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{1, 1, K_WIDTH}, "x");
        RotaryEmbedding block(K_WIDTH, K_GROUPS);
        const Trace trace = Trace::evaluation(builder);

        EXPECT_EQ(block.forward_incremental(trace, x, Shape{K_LENGTH, K_WIDTH}), x);
        SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "'rotary_cosine'",
                          "carrying trace");
    }
}

namespace
{
    /** @brief Provides the Engine every case records on. */
    class Medium_RotaryEmbedding : public SaTest::EngineTest
    {

    };

    /** @brief Bounds a float32 result against the host's double, absolutely. */
    constexpr double K_VALUE_TOLERANCE = 1e-5;

    /** @brief Holds the channels of a group in every case: two pairs. */
    constexpr std::size_t K_GROUP = 4;

    /** @brief Holds the pairs of a group in every case. */
    constexpr std::size_t K_VALUE_PAIRS = 2;

    /** @brief Holds K_VALUE_PAIRS as rotary_angle() reads it. */
    constexpr std::int64_t K_PAIR_COUNT = 2;

    /** @brief Holds a query whose values a float32 holds exactly. */
    const std::vector<double> K_QUERY{1.0, 2.0, 3.0, 4.0};

    /** @brief Holds a key whose values a float32 holds exactly. */
    const std::vector<double> K_KEY{0.5, -1.0, 2.0, 0.25};

    /**
     * @brief Traces a RotaryEmbedding of @p groups groups over [1, rows, width], fills its
     *        tables, runs it over @p input and writes the result to @p turned.
     */
    void turn_on_engine(SushiBLAS::Engine& engine, std::size_t rows, std::size_t groups,
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

        Graph::TensorPool pool(engine);
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        SA_EXPECT_OK(Train::fill_host_tables(tables, pool));
        SaTest::write_tensor(pool.tensor(x), input);

        std::unique_ptr<Graph::CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, Graph::CompiledStep::compile(engine, graph, pool));
        SA_EXPECT_OK(step->run());
        turned = SaTest::read_tensor(pool.tensor(y));
    }

    /** @brief Returns @p rows copies of @p row, one after the other. */
    std::vector<double> repeated(const std::vector<double>& row, std::size_t rows)
    {
        std::vector<double> values;
        values.reserve(rows * row.size());
        for (std::size_t at = 0; at < rows; ++at)
            values.insert(values.end(), row.begin(), row.end());

        return values;
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
} // namespace

/** @brief Tests each pair (a, b) of each row against (a cos - b sin, a sin + b cos). */
TEST_F(Medium_RotaryEmbedding, ATurnMatchesTheHostFormula)
{
    const std::size_t rows = 3;
    const std::vector<double> input{1.0, 2.0, 3.0, 4.0, 1.0, -1.0, 3.0, 0.25,
                                    -3.0, 1.0, 0.0, 2.0};
    std::vector<double> turned;
    turn_on_engine(*engine, rows, 1, input, turned);
    ASSERT_EQ(turned.size(), input.size());

    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t pair = 0; pair < K_VALUE_PAIRS; ++pair)
        {
            const double angle = rotary_angle(static_cast<std::int64_t>(row),
                                              static_cast<std::int64_t>(pair), K_PAIR_COUNT);
            const double a = input[row * K_GROUP + pair];
            const double b = input[row * K_GROUP + K_VALUE_PAIRS + pair];
            EXPECT_NEAR(turned[row * K_GROUP + pair], a * std::cos(angle) - b * std::sin(angle),
                        K_VALUE_TOLERANCE)
                << "row " << row << " pair " << pair;
            EXPECT_NEAR(turned[row * K_GROUP + K_VALUE_PAIRS + pair],
                        a * std::sin(angle) + b * std::cos(angle), K_VALUE_TOLERANCE)
                << "row " << row << " pair " << pair;
        }
    }

    // Pair 0 of row 1 is (1, 3) turned by one radian: cos 1 - 3 sin 1 and sin 1 + 3 cos 1.
    EXPECT_NEAR(turned[K_GROUP], -1.984110649, K_VALUE_TOLERANCE);
    EXPECT_NEAR(turned[K_GROUP + K_VALUE_PAIRS], 2.462377903, K_VALUE_TOLERANCE);
}

/** @brief Tests that position 0 turns nothing: its row leaves the block as it entered. */
TEST_F(Medium_RotaryEmbedding, PositionZeroIsTheIdentity)
{
    std::vector<double> turned;
    turn_on_engine(*engine, 2, 1, repeated(K_QUERY, 2), turned);
    ASSERT_EQ(turned.size(), 2 * K_GROUP);

    for (std::size_t channel = 0; channel < K_GROUP; ++channel)
        EXPECT_EQ(turned[channel], K_QUERY[channel]) << "channel " << channel;

    // Position 1 turns pair 0 by one radian, so its row differs.
    EXPECT_GT(std::abs(turned[K_GROUP] - K_QUERY[0]), 1.0);
}

/** @brief Tests that two groups of one row holding the same values turn to the same values. */
TEST_F(Medium_RotaryEmbedding, EveryGroupTurnsByTheSameAngles)
{
    const std::size_t rows = 3;
    std::vector<double> input;
    for (std::size_t row = 0; row < rows; ++row)
    {
        input.insert(input.end(), K_KEY.begin(), K_KEY.end());
        input.insert(input.end(), K_KEY.begin(), K_KEY.end());
    }

    std::vector<double> turned;
    turn_on_engine(*engine, rows, 2, input, turned);
    ASSERT_EQ(turned.size(), input.size());

    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t channel = 0; channel < K_GROUP; ++channel)
        {
            EXPECT_EQ(turned[row * 2 * K_GROUP + channel],
                      turned[row * 2 * K_GROUP + K_GROUP + channel])
                << "row " << row << " channel " << channel;
        }
    }
}

/** @brief Tests that the score of a turned query and key depends on their distance alone. */
TEST_F(Medium_RotaryEmbedding, AScoreDependsOnTheDistanceOfThePositionsAlone)
{
    const std::size_t rows = 6;
    std::vector<double> queries;
    std::vector<double> keys;
    turn_on_engine(*engine, rows, 1, repeated(K_QUERY, rows), queries);
    turn_on_engine(*engine, rows, 1, repeated(K_KEY, rows), keys);
    ASSERT_EQ(queries.size(), rows * K_GROUP);
    ASSERT_EQ(keys.size(), rows * K_GROUP);

    // Both positions moved by the same amount leave the score where it was.
    const double two_apart = score(queries, 2, keys, 0);
    EXPECT_NEAR(score(queries, 3, keys, 1), two_apart, K_VALUE_TOLERANCE);
    EXPECT_NEAR(score(queries, 4, keys, 2), two_apart, K_VALUE_TOLERANCE);
    EXPECT_NEAR(score(queries, 5, keys, 3), two_apart, K_VALUE_TOLERANCE);

    // The host gives the same number from the distance: the query is 2 theta_i ahead.
    double expected = 0.0;
    for (std::size_t pair = 0; pair < K_VALUE_PAIRS; ++pair)
    {
        const double angle = rotary_angle(2, static_cast<std::int64_t>(pair), K_PAIR_COUNT);
        const double a = K_QUERY[pair];
        const double b = K_QUERY[K_VALUE_PAIRS + pair];
        const double c = K_KEY[pair];
        const double d = K_KEY[K_VALUE_PAIRS + pair];
        expected += (a * c + b * d) * std::cos(angle) + (a * d - b * c) * std::sin(angle);
    }
    EXPECT_NEAR(two_apart, expected, K_VALUE_TOLERANCE);

    // Another distance gives another score for these two vectors.
    EXPECT_GT(std::abs(score(queries, 3, keys, 0) - two_apart), 1.0);
}
