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
 * @brief Tests the values RotaryEmbedding computes on an engine against the host formula.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/blocks/rotary_embedding.hpp>
#include <SushiAI/language/blocks/rotary_positions.hpp>
#include <SushiAI/nn/host_table.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/train/steps.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
using Language::Blocks::RotaryEmbedding;
using Language::Blocks::rotary_angle;

namespace
{
    /** @brief Provides the Engine every case records on. */
    class Integration_RotaryEmbedding : public SaTest::EngineTest
    {

    };

    /** @brief Bounds a float32 result against the host's double, absolutely. */
    constexpr double K_TOLERANCE = 1e-5;

    /** @brief Holds the channels of a group in every case: two pairs. */
    constexpr std::size_t K_GROUP = 4;

    /** @brief Holds the pairs of a group in every case. */
    constexpr std::size_t K_PAIRS = 2;

    /** @brief Holds K_PAIRS as rotary_angle() reads it. */
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
TEST_F(Integration_RotaryEmbedding, ATurnMatchesTheHostFormula)
{
    const std::size_t rows = 3;
    const std::vector<double> input{1.0, 2.0, 3.0, 4.0, 1.0, -1.0, 3.0, 0.25,
                                    -3.0, 1.0, 0.0, 2.0};
    std::vector<double> turned;
    turn_on_engine(*engine, rows, 1, input, turned);
    ASSERT_EQ(turned.size(), input.size());

    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t pair = 0; pair < K_PAIRS; ++pair)
        {
            const double angle = rotary_angle(static_cast<std::int64_t>(row),
                                              static_cast<std::int64_t>(pair), K_PAIR_COUNT);
            const double a = input[row * K_GROUP + pair];
            const double b = input[row * K_GROUP + K_PAIRS + pair];
            EXPECT_NEAR(turned[row * K_GROUP + pair], a * std::cos(angle) - b * std::sin(angle),
                        K_TOLERANCE)
                << "row " << row << " pair " << pair;
            EXPECT_NEAR(turned[row * K_GROUP + K_PAIRS + pair],
                        a * std::sin(angle) + b * std::cos(angle), K_TOLERANCE)
                << "row " << row << " pair " << pair;
        }
    }

    // Pair 0 of row 1 is (1, 3) turned by one radian: cos 1 - 3 sin 1 and sin 1 + 3 cos 1.
    EXPECT_NEAR(turned[K_GROUP], -1.984110649, K_TOLERANCE);
    EXPECT_NEAR(turned[K_GROUP + K_PAIRS], 2.462377903, K_TOLERANCE);
}

/** @brief Tests that position 0 turns nothing: its row leaves the block as it entered. */
TEST_F(Integration_RotaryEmbedding, PositionZeroIsTheIdentity)
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
TEST_F(Integration_RotaryEmbedding, EveryGroupTurnsByTheSameAngles)
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
TEST_F(Integration_RotaryEmbedding, AScoreDependsOnTheDistanceOfThePositionsAlone)
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
    EXPECT_NEAR(score(queries, 3, keys, 1), two_apart, K_TOLERANCE);
    EXPECT_NEAR(score(queries, 4, keys, 2), two_apart, K_TOLERANCE);
    EXPECT_NEAR(score(queries, 5, keys, 3), two_apart, K_TOLERANCE);

    // The host gives the same number from the distance: the query is 2 theta_i ahead.
    double expected = 0.0;
    for (std::size_t pair = 0; pair < K_PAIRS; ++pair)
    {
        const double angle = rotary_angle(2, static_cast<std::int64_t>(pair), K_PAIR_COUNT);
        const double a = K_QUERY[pair];
        const double b = K_QUERY[K_PAIRS + pair];
        const double c = K_KEY[pair];
        const double d = K_KEY[K_PAIRS + pair];
        expected += (a * c + b * d) * std::cos(angle) + (a * d - b * c) * std::sin(angle);
    }
    EXPECT_NEAR(two_apart, expected, K_TOLERANCE);

    // Another distance gives another score for these two vectors.
    EXPECT_GT(std::abs(score(queries, 3, keys, 0) - two_apart), 1.0);
}
