/**************************************************************************/
/* test_attention_operations.cpp                                          */
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
 * @file test_attention_operations.cpp
 * @brief Tests the rank-3 product, the softmax and the product's gradient on a device.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueId;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /** @brief Provides the Engine every attention case records on. */
    class Integration_AttentionOperations : public SaTest::EngineTest
    {

    };

    /** @brief Bounds a FLOAT32 result computed on the device against a double host value. */
    constexpr double K_ABSOLUTE_TOLERANCE = 1e-6;

    /**
     * @brief Returns @p count values of a fixed pattern, scale * ((i * stride) % 11 - 5).
     *
     * The pattern is deterministic, so no seed is needed, and it is symmetric about zero.
     */
    std::vector<double> patterned(std::size_t count, std::size_t stride, double scale)
    {
        std::vector<double> values(count);
        for (std::size_t i = 0; i < count; ++i)
            values[i] = scale * (static_cast<double>((i * stride) % 11) - 5.0);
        return values;
    }

    /** @brief Compares every element of @p got with @p expected to within @p absolute. */
    void expect_elements(const std::vector<double>& got, const std::vector<double>& expected,
                         double absolute)
    {
        ASSERT_EQ(got.size(), expected.size());

        for (std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(got[i], expected[i], absolute) << "at " << i;
    }

    /**
     * @brief Lowers @p graph into a step over @p pool and runs it once.
     *
     * The leaves of @p graph must already be materialised in @p pool and written.
     */
    void compile_and_run(SushiBLAS::Engine& engine, const Graph::Graph& graph, TensorPool& pool)
    {
        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(engine, graph, pool));
        SA_EXPECT_OK(step->run());
    }

    /**
     * @brief Returns the softmax over each row of @p x, taken on the host in double.
     * @param last The extent of the last axis; @p x holds whole rows of that length.
     */
    std::vector<double> host_softmax_rows(const std::vector<double>& x, std::size_t last)
    {
        std::vector<double> out(x.size());
        for (std::size_t base = 0; base < x.size(); base += last)
        {
            const double top = *std::max_element(x.data() + base, x.data() + base + last);
            double total = 0.0;
            for (std::size_t c = 0; c < last; ++c)
            {
                out[base + c] = std::exp(x[base + c] - top);
                total += out[base + c];
            }
            for (std::size_t c = 0; c < last; ++c)
                out[base + c] /= total;
        }
        return out;
    }
}

/** @brief Tests that a rank-3 product with a transposed right operand matches the host product. */
TEST_F(Integration_AttentionOperations, BatchedProductWithTransposedRightOperandMatchesHost)
{
    // [B, m, k] by [B, n, k] transposed gives [B, m, n] with B = 2, m = 3, k = 4, n = 5.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.input(Shape{2, 3, 4}, "a");
    const ValueId b = builder.input(Shape{2, 5, 4}, "b");
    const ValueId y = builder.matmul(a, b, false, true);

    const std::vector<double> left = patterned(2 * 3 * 4, 7, 0.1);
    const std::vector<double> right = patterned(2 * 5 * 4, 5, 0.05);

    std::vector<double> expected(2 * 3 * 5, 0.0);
    for (std::size_t batch = 0; batch < 2; ++batch)
    {
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t j = 0; j < 5; ++j)
            {
                double sum = 0.0;
                for (std::size_t k = 0; k < 4; ++k)
                    sum += left[(batch * 3 + i) * 4 + k] * right[(batch * 5 + j) * 4 + k];
                expected[(batch * 3 + i) * 5 + j] = sum;
            }
        }
    }

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(a), left);
    write_tensor(pool.tensor(b), right);

    ASSERT_NO_FATAL_FAILURE(compile_and_run(*engine, graph, pool));

    expect_elements(read_tensor(pool.tensor(y)), expected, K_ABSOLUTE_TOLERANCE);
}

/** @brief Tests that the softmax over a [2, 3, 3] value matches the host softmax in double. */
TEST_F(Integration_AttentionOperations, SoftmaxOverTheLastAxisMatchesHost)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId x = builder.input(Shape{2, 3, 3}, "x");
    const ValueId y = builder.softmax(x);

    const std::vector<double> logits = patterned(2 * 3 * 3, 3, 0.4);
    const std::vector<double> expected = host_softmax_rows(logits, 3);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), logits);

    ASSERT_NO_FATAL_FAILURE(compile_and_run(*engine, graph, pool));

    expect_elements(read_tensor(pool.tensor(y)), expected, K_ABSOLUTE_TOLERANCE);
}

/**
 * @brief Tests that the gradient of a rank-3 product is rank 3 and matches the host.
 *
 * The loss is the sum of every product element, so dloss/da[b,i,k] is the sum over j of
 * b[b,j,k], and dloss/db[b,j,k] is the sum over i of a[b,i,k].
 */
TEST_F(Integration_AttentionOperations, BatchedProductGradientEmitsRankThreeProducts)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId a = builder.parameter(Shape{2, 3, 4}, "a");
    const ValueId b = builder.parameter(Shape{2, 5, 4}, "b");
    const ValueId product = builder.matmul(a, b, false, true);
    const ValueId loss = builder.sum_rows(builder.reshape(product, Shape{30, 1}));

    const std::array<ValueId, 2> wrt = {a, b};
    const std::vector<ValueId> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

    ASSERT_EQ(gradients.size(), 2u);
    ASSERT_NE(gradients[0], Graph::K_NO_VALUE);
    ASSERT_NE(gradients[1], Graph::K_NO_VALUE);
    EXPECT_EQ(graph.value(gradients[0]).shape, Shape({2, 3, 4}));
    EXPECT_EQ(graph.value(gradients[1]).shape, Shape({2, 5, 4}));

    // The forward product and the two products matmul_backward emits are all rank 3.
    std::size_t products = 0;
    for (const Graph::Node& node : graph.nodes())
    {
        if (node.op != Ops::K_MATMUL.id)
            continue;
        ++products;
        EXPECT_EQ(graph.value(node.outputs[0]).shape.rank(), 3);
    }
    EXPECT_EQ(products, 3u);

    const std::vector<double> left = patterned(2 * 3 * 4, 7, 0.1);
    const std::vector<double> right = patterned(2 * 5 * 4, 5, 0.05);

    std::vector<double> expected_a(2 * 3 * 4, 0.0);
    std::vector<double> expected_b(2 * 5 * 4, 0.0);
    for (std::size_t batch = 0; batch < 2; ++batch)
    {
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t k = 0; k < 4; ++k)
            {
                for (std::size_t j = 0; j < 5; ++j)
                    expected_a[(batch * 3 + i) * 4 + k] += right[(batch * 5 + j) * 4 + k];
            }
        }
        for (std::size_t j = 0; j < 5; ++j)
        {
            for (std::size_t k = 0; k < 4; ++k)
            {
                for (std::size_t i = 0; i < 3; ++i)
                    expected_b[(batch * 5 + j) * 4 + k] += left[(batch * 3 + i) * 4 + k];
            }
        }
    }

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(a), left);
    write_tensor(pool.tensor(b), right);

    ASSERT_NO_FATAL_FAILURE(compile_and_run(*engine, graph, pool));

    expect_elements(read_tensor(pool.tensor(gradients[0])), expected_a, K_ABSOLUTE_TOLERANCE);
    expect_elements(read_tensor(pool.tensor(gradients[1])), expected_b, K_ABSOLUTE_TOLERANCE);
}
