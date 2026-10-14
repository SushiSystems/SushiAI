/**************************************************************************/
/* test_causal_attention.cpp                                              */
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
 * @file test_causal_attention.cpp
 * @brief Checks CAUSAL_MASK on an engine, and that a masked attention and a transformer block
 *        keep a position's output when a later position's input changes.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using Language::Blocks::AttentionMask;
using Language::Blocks::CausalSelfAttention;
using Language::Blocks::TransformerBlock;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /** @brief Provides the Engine every causal case records on. */
    class Integration_CausalAttention : public SaTest::EngineTest
    {

    };

    /** @brief Bounds a value that must be the same number in both runs. */
    constexpr double K_EXACT_TOLERANCE = 0.0;

    /** @brief Holds the value the mask cases write above the diagonal. */
    constexpr double K_FILL = -1.0e9;

    /** @brief Holds T, the number of positions of the sequence the block cases read. */
    constexpr std::size_t K_LENGTH = 5;

    /** @brief Holds the width of the sequence the block cases read. */
    constexpr std::size_t K_DIM = 8;

    /** @brief Holds the head count of the block cases. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Holds the first position whose input the block cases change. */
    constexpr std::size_t K_CHANGED_POSITION = 3;

    /** @brief Returns @p count values of the deterministic host stream seeded with @p seed. */
    std::vector<double> noise(std::size_t count, std::uint64_t seed)
    {
        std::vector<double> values(count);
        SaTest::HostNoise(seed).fill(values, 1.0);
        return values;
    }

    /** @brief Returns @p scores, [matrices, length, length], with the fill above each diagonal. */
    std::vector<double> masked_on_host(std::vector<double> scores, std::size_t length,
                                       double fill)
    {
        for (std::size_t i = 0; i < scores.size(); ++i)
        {
            const std::size_t row = (i / length) % length;
            const std::size_t column = i % length;
            if (column > row)
                scores[i] = fill;
        }

        return scores;
    }

    /** @brief Runs the mask over a [2, 4, 4] value of @p dtype and compares with the host. */
    void expect_mask_matches_host(SushiBLAS::Engine& engine, DType dtype)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID scores = builder.input(Shape{2, 4, 4}, "scores", dtype);
        const Graph::Traced masked = builder.causal_mask(scores, K_FILL);
        SA_EXPECT_OK(masked);
        ASSERT_TRUE(masked.has_value());

        // Dyadic values survive the FLOAT32 round trip, so the kept entries compare exactly.
        std::vector<double> values(2 * 4 * 4);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = 0.25 * static_cast<double>(i) - 3.0;

        TensorPool pool(engine);
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        write_tensor(pool.tensor(scores), values);

        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(engine, graph, pool));
        SA_EXPECT_OK(step->run());
        SA_EXPECT_OK(step->run());

        const std::vector<double> got = read_tensor(pool.tensor(masked.value()));
        const std::vector<double> expected = masked_on_host(values, 4, K_FILL);
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(got[i], expected[i], K_EXACT_TOLERANCE) << "at element " << i;
    }

    /** @brief Holds what two runs of one block gave: before and after a later input changed. */
    struct TwoRuns
    {
        std::vector<double> before; /**< The [K_LENGTH, K_DIM] output of the first input. */
        std::vector<double> after;  /**< The output once positions from 3 on were changed. */
    };

    /**
     * @brief Traces @p block over a [1, K_LENGTH, K_DIM] input, fills every parameter from
     *        the host stream, and runs it on an input and on one changed from position 3 on.
     */
    template<typename Block>
    void run_twice(SushiBLAS::Engine& engine, Block& block, TwoRuns& runs)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID x = builder.input(
            Shape{1, static_cast<int64_t>(K_LENGTH), static_cast<int64_t>(K_DIM)}, "x");
        const NN::Trace trace = NN::Trace::evaluation(builder);
        const ValueID y = block.forward(trace, x);
        SA_EXPECT_OK(trace.status());
        ASSERT_FALSE(graph.has_refusal());

        TensorPool pool(engine);
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        for (ValueID id = 0; id < graph.value_count(); ++id)
        {
            if (graph.value(id).lifetime != Graph::Lifetime::Parameter)
                continue;

            const std::size_t count =
                static_cast<std::size_t>(graph.value(id).shape.num_elements());
            write_tensor(pool.tensor(id), noise(count, 6100 + id));
        }

        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(engine, graph, pool));

        std::vector<double> input = noise(K_LENGTH * K_DIM, 6001);
        write_tensor(pool.tensor(x), input);
        SA_EXPECT_OK(step->run());
        runs.before = read_tensor(pool.tensor(y));

        const std::vector<double> later = noise(K_LENGTH * K_DIM, 6002);
        for (std::size_t i = K_CHANGED_POSITION * K_DIM; i < input.size(); ++i)
            input[i] = later[i];

        write_tensor(pool.tensor(x), input);
        SA_EXPECT_OK(step->run());
        runs.after = read_tensor(pool.tensor(y));
    }

    /** @brief Counts the elements of positions [first, last) at which the two runs differ. */
    std::size_t changed_elements(const TwoRuns& runs, std::size_t first, std::size_t last)
    {
        std::size_t changed = 0;
        for (std::size_t i = first * K_DIM; i < last * K_DIM; ++i)
        {
            if (runs.before[i] != runs.after[i])
                ++changed;
        }

        return changed;
    }

    /** @brief Expects the two runs to hold K_LENGTH * K_DIM finite elements each. */
    void expect_whole_and_finite(const TwoRuns& runs)
    {
        ASSERT_EQ(runs.before.size(), K_LENGTH * K_DIM);
        ASSERT_EQ(runs.after.size(), K_LENGTH * K_DIM);
        for (const double value : runs.before)
            ASSERT_TRUE(std::isfinite(value));
    }
}

/** @brief Tests that the mask keeps the lower triangle and writes the fill above, in FLOAT32. */
TEST_F(Integration_CausalAttention, TheMaskReplacesTheEntriesAboveTheDiagonal)
{
    ASSERT_NO_FATAL_FAILURE(expect_mask_matches_host(*engine, DType::Float32));
}

/** @brief Tests the same in FLOAT64, where the mask vectors are converted from FLOAT32. */
TEST_F(Integration_CausalAttention, TheMaskReplacesTheEntriesAboveTheDiagonalInFloat64)
{
    ASSERT_NO_FATAL_FAILURE(expect_mask_matches_host(*engine, DType::Float64));
}

/**
 * @brief Tests that the gradient of the scores is the incoming gradient on and below the
 *        diagonal and zero above it.
 *
 * The loss is the cross-entropy of the masked [4, 4] rows, so the incoming gradient is
 * (softmax - target) / 4, which the host computes from the masked scores.
 */
TEST_F(Integration_CausalAttention, TheMaskPassesItsGradientThroughTheKeptEntriesAlone)
{
    constexpr std::size_t K_SIDE = 4;

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID scores = builder.parameter(Shape{4, 4}, "scores", DType::Float64);
    const ValueID targets = builder.input(Shape{4, 4}, "targets", DType::Float64);
    const Graph::Traced masked = builder.causal_mask(scores, K_FILL);
    ASSERT_TRUE(masked.has_value());
    const ValueID loss = builder.nll_loss(builder.log_softmax(masked.value()), targets);

    const std::vector<ValueID> wrt = {scores};
    const std::vector<ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));
    ASSERT_NE(gradients.at(0), Graph::K_NO_VALUE);

    // Row r targets class 0, which every row keeps.
    const std::vector<double> values = noise(K_SIDE * K_SIDE, 6201);
    std::vector<double> one_hot(K_SIDE * K_SIDE, 0.0);
    for (std::size_t row = 0; row < K_SIDE; ++row)
        one_hot[row * K_SIDE] = 1.0;

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(scores), values);
    write_tensor(pool.tensor(targets), one_hot);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));
    SA_EXPECT_OK(step->run());

    const std::vector<double> got = read_tensor(pool.tensor(gradients[0]));
    ASSERT_EQ(got.size(), K_SIDE * K_SIDE);
    for (std::size_t row = 0; row < K_SIDE; ++row)
    {
        double total = 0.0;
        for (std::size_t column = 0; column <= row; ++column)
            total += std::exp(values[row * K_SIDE + column]);

        for (std::size_t column = 0; column < K_SIDE; ++column)
        {
            const std::size_t i = row * K_SIDE + column;
            const double kept = (std::exp(values[i]) / total - one_hot[i]) / 4.0;
            const double expected = column <= row ? kept : 0.0;
            const double bound = column <= row ? 1e-12 : K_EXACT_TOLERANCE;
            EXPECT_NEAR(got[i], expected, bound) << "at row " << row << " column " << column;
        }
    }
}

/** @brief Tests that positions 0 to 2 of a masked attention keep their output exactly. */
TEST_F(Integration_CausalAttention, AnEarlierPositionIgnoresALaterInput)
{
    CausalSelfAttention block(static_cast<int64_t>(K_DIM), K_HEADS, AttentionMask::Causal);
    TwoRuns runs;
    ASSERT_NO_FATAL_FAILURE(run_twice(*engine, block, runs));
    ASSERT_NO_FATAL_FAILURE(expect_whole_and_finite(runs));

    EXPECT_EQ(changed_elements(runs, 0, K_CHANGED_POSITION), 0u);
    EXPECT_GT(changed_elements(runs, K_CHANGED_POSITION, K_LENGTH), 0u)
        << "the changed positions gave the output they gave before";
}

/** @brief Tests that without the mask an earlier position does read the later input. */
TEST_F(Integration_CausalAttention, WithoutTheMaskAnEarlierPositionReadsALaterInput)
{
    CausalSelfAttention block(static_cast<int64_t>(K_DIM), K_HEADS, AttentionMask::None);
    TwoRuns runs;
    ASSERT_NO_FATAL_FAILURE(run_twice(*engine, block, runs));
    ASSERT_NO_FATAL_FAILURE(expect_whole_and_finite(runs));

    EXPECT_GT(changed_elements(runs, 0, K_CHANGED_POSITION), 0u);
}

/** @brief Tests that a whole causal transformer block keeps the same property. */
TEST_F(Integration_CausalAttention, ATransformerBlockKeepsEarlierPositions)
{
    TransformerBlock block(static_cast<int64_t>(K_DIM), K_HEADS, 4 * static_cast<int64_t>(K_DIM),
                           AttentionMask::Causal, NN::K_LAYER_NORM_EPSILON);
    TwoRuns runs;
    ASSERT_NO_FATAL_FAILURE(run_twice(*engine, block, runs));
    ASSERT_NO_FATAL_FAILURE(expect_whole_and_finite(runs));

    EXPECT_EQ(changed_elements(runs, 0, K_CHANGED_POSITION), 0u);
    EXPECT_GT(changed_elements(runs, K_CHANGED_POSITION, K_LENGTH), 0u);
}
