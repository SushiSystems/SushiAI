/**************************************************************************/
/* test_linalg.cpp                                                        */
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
 * @file test_linalg.cpp
 * @brief Tests the linear algebra lowerings on a device: the rank-3 product, the softmax and the
 *        product's gradient.
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
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/graph/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /** @brief Provides the Engine every attention case records on. */
    class Medium_AttentionOperations : public SaTest::EngineTest
    {

    };

    /**
     * @brief Bounds a FLOAT32 result computed on the device against a double host value; 32 unit
     * roundoffs of float, the smallest power of two that covers 1e-6.
     */
    constexpr double K_ABSOLUTE_TOLERANCE = 32.0 * TestSupport::unit_roundoff<float>();

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
    void compile_and_run(SushiAI::ExecutionBackend& backend, const Graph::Graph& graph,
                         TensorPool& pool)
    {
        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(backend, graph, pool));
        SA_EXPECT_OK(step->run());
    }

    /**
     * @brief Returns the [B, m, n] product of @p left [B, m, k] and the transpose of @p right
     *        [B, n, k], taken on the host in double.
     */
    std::vector<double> host_product_with_transposed_right(const std::vector<double>& left,
                                                           const std::vector<double>& right,
                                                           std::size_t batches, std::size_t m,
                                                           std::size_t n, std::size_t k)
    {
        std::vector<double> product(batches * m * n, 0.0);
        for (std::size_t batch = 0; batch < batches; ++batch)
        {
            for (std::size_t i = 0; i < m; ++i)
            {
                for (std::size_t j = 0; j < n; ++j)
                {
                    double sum = 0.0;
                    for (std::size_t p = 0; p < k; ++p)
                        sum += left[(batch * m + i) * k + p] * right[(batch * n + j) * k + p];
                    product[(batch * m + i) * n + j] = sum;
                }
            }
        }
        return product;
    }

    /**
     * @brief Returns dloss/dleft of the sum of every element of the product of @p left [B, m, k]
     *        and the transpose of @p right [B, n, k]: the sum of @p right over its rows.
     */
    std::vector<double> host_left_gradient_of_the_summed_product(
        const std::vector<double>& right, std::size_t batches, std::size_t m, std::size_t n,
        std::size_t k)
    {
        std::vector<double> gradient(batches * m * k, 0.0);
        for (std::size_t batch = 0; batch < batches; ++batch)
        {
            for (std::size_t i = 0; i < m; ++i)
            {
                for (std::size_t p = 0; p < k; ++p)
                {
                    for (std::size_t j = 0; j < n; ++j)
                        gradient[(batch * m + i) * k + p] += right[(batch * n + j) * k + p];
                }
            }
        }
        return gradient;
    }

    /**
     * @brief Returns dloss/dright of the sum of every element of the product of @p left [B, m, k]
     *        and the transpose of @p right [B, n, k]: the sum of @p left over its rows.
     */
    std::vector<double> host_right_gradient_of_the_summed_product(
        const std::vector<double>& left, std::size_t batches, std::size_t m, std::size_t n,
        std::size_t k)
    {
        std::vector<double> gradient(batches * n * k, 0.0);
        for (std::size_t batch = 0; batch < batches; ++batch)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                for (std::size_t p = 0; p < k; ++p)
                {
                    for (std::size_t i = 0; i < m; ++i)
                        gradient[(batch * n + j) * k + p] += left[(batch * m + i) * k + p];
                }
            }
        }
        return gradient;
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
TEST_F(Medium_AttentionOperations, BatchedProductWithTransposedRightOperandMatchesHost)
{
    // [B, m, k] by [B, n, k] transposed gives [B, m, n] with B = 2, m = 3, k = 4, n = 5.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{2, 3, 4}, "a");
    const ValueID b = builder.input(Shape{2, 5, 4}, "b");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.matmul(a, b, false, true));

    const std::vector<double> left = patterned(2 * 3 * 4, 7, 0.1);
    const std::vector<double> right = patterned(2 * 5 * 4, 5, 0.05);

    const std::vector<double> expected = host_product_with_transposed_right(left, right, 2, 3, 5, 4);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(a), left);
    write_tensor(pool.tensor(b), right);

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    expect_elements(read_tensor(pool.tensor(y)), expected, K_ABSOLUTE_TOLERANCE);
}

/** @brief Tests that the softmax over a [2, 3, 3] value matches the host softmax in double. */
TEST_F(Medium_AttentionOperations, SoftmaxOverTheLastAxisMatchesHost)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, 3, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.softmax(x));

    const std::vector<double> logits = patterned(2 * 3 * 3, 3, 0.4);
    const std::vector<double> expected = host_softmax_rows(logits, 3);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), logits);

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    expect_elements(read_tensor(pool.tensor(y)), expected, K_ABSOLUTE_TOLERANCE);
}

/**
 * @brief Tests that the gradients of a rank-3 product have the operands' shapes and match the host.
 *
 * The loss is the sum of every product element, so dloss/da[b,i,k] is the sum over j of
 * b[b,j,k], and dloss/db[b,j,k] is the sum over i of a[b,i,k].
 */
TEST_F(Medium_AttentionOperations, BatchedProductGradientsHaveTheOperandShapesAndMatchTheHost)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.parameter(Shape{2, 3, 4}, "a");
    const ValueID b = builder.parameter(Shape{2, 5, 4}, "b");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID product, builder.matmul(a, b, false, true));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            builder.sum_rows(builder.reshape(product, Shape{30, 1})));

    const std::array<ValueID, 2> wrt = {a, b};
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<ValueID> gradients,
                            Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())));

    ASSERT_EQ(gradients.size(), 2u);
    ASSERT_NE(gradients[0], Graph::K_NO_VALUE);
    ASSERT_NE(gradients[1], Graph::K_NO_VALUE);
    EXPECT_EQ(graph.value(gradients[0]).shape, Shape({2, 3, 4}));
    EXPECT_EQ(graph.value(gradients[1]).shape, Shape({2, 5, 4}));

    const std::vector<double> left = patterned(2 * 3 * 4, 7, 0.1);
    const std::vector<double> right = patterned(2 * 5 * 4, 5, 0.05);

    const std::vector<double> expected_a =
        host_left_gradient_of_the_summed_product(right, 2, 3, 5, 4);
    const std::vector<double> expected_b =
        host_right_gradient_of_the_summed_product(left, 2, 3, 5, 4);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(a), left);
    write_tensor(pool.tensor(b), right);

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    expect_elements(read_tensor(pool.tensor(gradients[0])), expected_a, K_ABSOLUTE_TOLERANCE);
    expect_elements(read_tensor(pool.tensor(gradients[1])), expected_b, K_ABSOLUTE_TOLERANCE);
}
