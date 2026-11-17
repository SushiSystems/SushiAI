/**************************************************************************/
/* test_sequence_pool.cpp                                                 */
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
 * @file test_sequence_pool.cpp
 * @brief Tests SequencePool: its trace and refusals, and its mean and gradient on an engine.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/sequence_pool.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/gradient_probe.hpp>
#include <support/tolerance.hpp>
#include "blocks_test_support.hpp"

using namespace SushiAI;
using Language::Blocks::SequencePool;
using NN::Trace;
// SaTest stays reachable inside a TEST_F body; see tests/language/blocks/README.md.
namespace SaTest = SushiAI::Test;
using Graph::GraphBuilder;
using Graph::ValueID;

namespace
{
    /** @brief Holds N, the number of texts of every case. */
    constexpr int64_t K_TEXTS = 2;

    /** @brief Holds T, the positions of every text. */
    constexpr int64_t K_POSITIONS = 5;

    /** @brief Holds the width of every position. */
    constexpr int64_t K_DIM = 4;

    /** @brief Holds the length of the first text; the second fills its sequence. */
    constexpr int64_t K_FIRST_LENGTH = 3;

    /** @brief Names a row by its name field. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return info.param.name;
    }

    /** @brief Returns the [2, 5, 4] values every numeric case reads, as dyadic eighths. */
    std::vector<double> sequence_values()
    {
        return SaTest::eighths(K_TEXTS * K_POSITIONS * K_DIM, 3);
    }

    /**
     * @brief Returns the mean of the first @p length positions of text @p text, worked on the
     *        host in double precision.
     */
    std::vector<double> mean_on_host(const std::vector<double>& values, int64_t text,
                                     int64_t length)
    {
        std::vector<double> mean(static_cast<std::size_t>(K_DIM), 0.0);
        for (int64_t position = 0; position < length; ++position)
        {
            for (int64_t channel = 0; channel < K_DIM; ++channel)
                mean[static_cast<std::size_t>(channel)] += values[static_cast<std::size_t>(
                    (text * K_POSITIONS + position) * K_DIM + channel)];
        }

        for (double& element : mean)
            element /= static_cast<double>(length);

        return mean;
    }

    /** @brief Holds an input the trace refuses and the words of its message. */
    struct InputRow
    {
        std::string name;    /**< The row name. */
        Shape sequence;      /**< The shape of the sequence value. */
        DType sequence_type; /**< The element type of the sequence value. */
        Shape lengths;       /**< The shape of the lengths. */
        std::string first;   /**< A word of the message. */
        std::string second;  /**< A word of the message. */
    };

    /** @brief Runs one input the trace refuses. */
    class Small_BlockSequencePoolInput : public ::testing::TestWithParam<InputRow>
    {

    };

    /** @brief Provides the Engine every numeric case runs on. */
    class Medium_BlockSequencePool : public SaTest::EngineTest
    {

    };

    /**
     * @brief Holds the leaf and the value of the pool graph that a probe builds around a
     *        [2, 5, 4] variable.
     *
     * The probe's loss is the cross-entropy of the pooled [2, 4] rows against targets whose row r
     * names class r.
     */
    struct PoolGraph
    {
        ValueID lengths = Graph::K_NO_VALUE; /**< The [2] lengths of the texts. */
        ValueID pooled = Graph::K_NO_VALUE;  /**< The pooled [2, 4] rows. */

        /** @brief Pools @p x with the lengths and returns the log-softmax of the pooled rows. */
        Graph::Traced build(const GraphBuilder& builder, ValueID x)
        {
            lengths = builder.input(Shape{K_TEXTS}, "lengths", DType::Float64);

            SequencePool block;
            const Trace trace = Trace::evaluation(builder);
            pooled = block.forward_with_lengths(trace, x, lengths);
            SA_EXPECT_OK(trace.status());

            return builder.log_softmax(pooled);
        }
    };

    /** @brief Returns the shape of the variable every probe differentiates at. */
    Shape sequence_shape()
    {
        return Shape{K_TEXTS, K_POSITIONS, K_DIM};
    }

    /** @brief Expects every element of @p computed within the bound of the mean of @p length. */
    void expect_mean_close(const std::vector<double>& computed,
                           const std::vector<double>& reference, int64_t length)
    {
        ASSERT_EQ(computed.size(), reference.size());
        for (std::size_t i = 0; i < computed.size(); ++i)
        {
            // Every eighth is at most 5/8, so the sum of the absolute products is below 1.
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<double>(
                computed[i], reference[i], 1.0, static_cast<std::size_t>(length) + 1)))
                << "at element " << i;
        }
    }
} // namespace

/** @brief Tests that the pool of a [N, T, dim] value is a [N, dim] value and records no failure. */
TEST(Small_BlockSequencePool, ATraceDropsTheSequenceAxis)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_TEXTS, K_POSITIONS, K_DIM}, "x");
    const Graph::ValueID lengths = builder.input(Shape{K_TEXTS}, "lengths");
    SequencePool block;

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID pooled = block.forward_with_lengths(trace, x, lengths);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(pooled).shape, (Shape{K_TEXTS, K_DIM}));
}

/** @brief Tests that an input that does not fit fails and hands back the sequence. */
TEST_P(Small_BlockSequencePoolInput, AnInputThatDoesNotFitFails)
{
    const InputRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(row.sequence, "x", row.sequence_type);
    const Graph::ValueID lengths = builder.input(row.lengths, "lengths");
    SequencePool block;

    const Trace trace = Trace::evaluation(builder);
    EXPECT_EQ(block.forward_with_lengths(trace, x, lengths), x);

    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockSequencePoolInput,
    ::testing::Values(
        InputRow{"RankTwoSequence", Shape{K_POSITIONS, K_DIM}, DType::Float32, Shape{K_TEXTS},
                 "rank-3 [N, T, dim] input", "[5, 4]"},
        InputRow{"IntegerSequence", Shape{K_TEXTS, K_POSITIONS, K_DIM}, DType::Int32,
                 Shape{K_TEXTS}, "half, float32 or float64", "int32"},
        InputRow{"OneLengthForTwoTexts", Shape{K_TEXTS, K_POSITIONS, K_DIM}, DType::Float32,
                 Shape{1}, "one real floating length", "[1]"}),
    row_name<InputRow>);

/** @brief Tests that the per-sample shape of the result is the width of a position. */
TEST(Small_BlockSequencePool, TheResultShapeIsTheWidthOfAPosition)
{
    const SequencePool block{};

    SA_ASSERT_OK_AND_ASSIGN(const Shape pooled, block.output_shape(Shape{K_POSITIONS, K_DIM}));
    EXPECT_EQ(pooled, (Shape{K_DIM}));
    SA_EXPECT_FAILURE(block.output_shape(Shape{K_DIM}), StatusCode::InvalidDescription,
                      "per-sample [T, dim] input", "[4]");
}

/**
 * @brief Tests that row n of the pool is the mean of the first lengths[n] positions of text n.
 *
 * The oracle is the mean worked on the host in double precision from the same eighths. The first
 * text has two padded positions, which must neither add to the sum nor count in the divisor.
 */
TEST_F(Medium_BlockSequencePool, ARowIsTheMeanOverTheValidPositionsOfItsText)
{
    PoolGraph graph;
    SaTest::GradientProbe probe(backend(), sequence_shape(),
                                [&graph](const GraphBuilder& builder, ValueID x)
    {
        return graph.build(builder, x);
    });
    probe.fill(graph.lengths, {static_cast<double>(K_FIRST_LENGTH),
                               static_cast<double>(K_POSITIONS)});
    const std::vector<double> values = sequence_values();

    probe.run_at(values);

    const std::vector<double> pooled = probe.read(graph.pooled);

    ASSERT_EQ(pooled.size(), static_cast<std::size_t>(K_TEXTS * K_DIM));
    expect_mean_close(std::vector<double>(pooled.begin(), pooled.begin() + K_DIM),
                      mean_on_host(values, 0, K_FIRST_LENGTH), K_FIRST_LENGTH);
    expect_mean_close(std::vector<double>(pooled.begin() + K_DIM, pooled.end()),
                      mean_on_host(values, 1, K_POSITIONS), K_POSITIONS);
}

/** @brief Tests that a length of 0 gives a row of zeros and leaves the other row alone. */
TEST_F(Medium_BlockSequencePool, ALengthOfZeroGivesARowOfZeros)
{
    PoolGraph graph;
    SaTest::GradientProbe probe(backend(), sequence_shape(),
                                [&graph](const GraphBuilder& builder, ValueID x)
    {
        return graph.build(builder, x);
    });
    probe.fill(graph.lengths, {0.0, static_cast<double>(K_POSITIONS)});
    const std::vector<double> values = sequence_values();

    probe.run_at(values);

    const std::vector<double> pooled = probe.read(graph.pooled);

    ASSERT_EQ(pooled.size(), static_cast<std::size_t>(K_TEXTS * K_DIM));
    // A zero weight times a finite value is exactly zero, so the row compares exactly.
    EXPECT_EQ(std::vector<double>(pooled.begin(), pooled.begin() + K_DIM),
              std::vector<double>(static_cast<std::size_t>(K_DIM), 0.0));
    expect_mean_close(std::vector<double>(pooled.begin() + K_DIM, pooled.end()),
                      mean_on_host(values, 1, K_POSITIONS), K_POSITIONS);
}

/** @brief Tests that the gradient of the pool at its sequence matches central differences. */
TEST_F(Medium_BlockSequencePool, ThePoolDifferentiatesAtItsSequence)
{
    PoolGraph graph;
    SaTest::GradientProbe probe(backend(), sequence_shape(),
                                [&graph](const GraphBuilder& builder, ValueID x)
    {
        return graph.build(builder, x);
    });
    probe.fill(graph.lengths, {static_cast<double>(K_FIRST_LENGTH),
                               static_cast<double>(K_POSITIONS)});
    std::vector<double> point = sequence_values();

    SaTest::expect_gradient_matches(probe, point);
}
