/**************************************************************************/
/* test_activation.cpp                                                    */
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
 * @file test_activation.cpp
 * @brief Tests the activation lowering table and the gradient rows that read a forward output.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/activation.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::LoweringRule;
using Graph::TensorPool;
using Graph::ValueID;

namespace
{
    /** @brief Provides the Engine every activation gradient case records on. */
    class Medium_ActivationLowerings : public SaTest::EngineTest
    {

    };

    /**
     * @brief Bounds a FLOAT32 result computed on the device against a double host value; 32 unit
     * roundoffs of float, the smallest power of two that covers 1e-6.
     */
    constexpr double K_ABSOLUTE_TOLERANCE = 32.0 * TestSupport::unit_roundoff<float>();

    /** @brief Compares every element of @p got with @p expected to within @p absolute. */
    void expect_elements(const std::vector<double>& got, const std::vector<double>& expected,
                         double absolute)
    {
        ASSERT_EQ(got.size(), expected.size());

        for (std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(got[i], expected[i], absolute) << "at " << i;
    }

    /** @brief Returns dy * (1 - y * y) element by element, the tanh gradient given its output. */
    std::vector<double> host_tanh_gradient(const std::vector<double>& upstream,
                                           const std::vector<double>& output)
    {
        std::vector<double> gradient(upstream.size());
        for (std::size_t i = 0; i < upstream.size(); ++i)
            gradient[i] = upstream[i] * (1.0 - output[i] * output[i]);
        return gradient;
    }

    /** @brief Returns dy * y * (1 - y) element by element, the sigmoid gradient given its output. */
    std::vector<double> host_sigmoid_gradient(const std::vector<double>& upstream,
                                              const std::vector<double>& output)
    {
        std::vector<double> gradient(upstream.size());
        for (std::size_t i = 0; i < upstream.size(); ++i)
            gradient[i] = upstream[i] * output[i] * (1.0 - output[i]);
        return gradient;
    }

    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        std::size_t index; /**< The position of the row in the table. */
        OpID op;          /**< The operation. */
        LoweringRule rule; /**< The row that lowers it. */
    };

    /** @brief Runs once per row the header names for the activation table. */
    class Small_ActivationRow : public ::testing::TestWithParam<ExpectedRow>
    {

    };

    /** @brief Lists the rows of the activation table in the order the header writes them. */
    const std::array<ExpectedRow, 14> K_EXPECTED_ROWS = {{
        {0, Ops::K_RELU.id, &Graph::lower_relu},
        {1, Ops::K_GELU.id, &Graph::lower_gelu},
        {2, Ops::K_SIGMOID.id, &Graph::lower_sigmoid},
        {3, Ops::K_TANH.id, &Graph::lower_tanh},
        {4, Ops::K_SILU.id, &Graph::lower_silu},
        {5, Ops::K_SOFTPLUS.id, &Graph::lower_softplus},
        {6, Ops::K_LEAKY_RELU.id, &Graph::lower_leaky_relu},
        {7, Ops::K_RELU_BACKWARD.id, &Graph::lower_relu_backward},
        {8, Ops::K_GELU_BACKWARD.id, &Graph::lower_gelu_backward},
        {9, Ops::K_SIGMOID_BACKWARD.id, &Graph::lower_sigmoid_backward},
        {10, Ops::K_TANH_BACKWARD.id, &Graph::lower_tanh_backward},
        {11, Ops::K_SILU_BACKWARD.id, &Graph::lower_silu_backward},
        {12, Ops::K_SOFTPLUS_BACKWARD.id, &Graph::lower_softplus_backward},
        {13, Ops::K_LEAKY_RELU_BACKWARD.id, &Graph::lower_leaky_relu_backward},
    }};
} // namespace

/** @brief Tests that the table holds one row for each row the header names, and no other. */
TEST(Small_ActivationLowerings, TheTableHoldsExactlyTheRowsTheHeaderNames)
{
    EXPECT_EQ(Graph::K_ACTIVATION_LOWERINGS.size(), K_EXPECTED_ROWS.size());
}

/** @brief Tests that a row pairs its operation with the function the header names. */
TEST_P(Small_ActivationRow, EachRowPairsItsOperationWithItsLowering)
{
    const ExpectedRow row = GetParam();
    const std::string name(Ops::name_of(row.op));
    ASSERT_LT(row.index, Graph::K_ACTIVATION_LOWERINGS.size());
    const Graph::LoweringEntry& entry = Graph::K_ACTIVATION_LOWERINGS[row.index];

    EXPECT_EQ(entry.op, row.op) << name;
    EXPECT_EQ(entry.rule, row.rule) << name;
    EXPECT_EQ(entry.scratch, nullptr) << name;
    EXPECT_EQ(Graph::find_lowering(row.op), row.rule) << name;
}

INSTANTIATE_TEST_SUITE_P(, Small_ActivationRow, ::testing::ValuesIn(K_EXPECTED_ROWS),
                         [](const ::testing::TestParamInfo<ExpectedRow>& info)
                         {
                             std::string name(Ops::name_of(info.param.op));
                             name.replace(0, name.find('.') + 1, "");
                             return ::SushiAI::Test::parameter_name(name);
                         });

/**
 * @brief Tests that the tanh and sigmoid gradients read the forward output, not the input.
 *
 * Given the output y, the tanh gradient is dy * (1 - y * y) and the sigmoid gradient is
 * dy * y * (1 - y).
 */
TEST_F(Medium_ActivationLowerings, TanhAndSigmoidGradientsReadTheForwardOutput)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID dy = builder.input(Shape{6}, "dy");
    const ValueID y = builder.input(Shape{6}, "y");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID tanh_dx,
                            builder.activation_backward(Ops::K_TANH_BACKWARD.id, dy, y));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID sigmoid_dx,
                            builder.activation_backward(Ops::K_SIGMOID_BACKWARD.id, dy, y));

    const std::vector<double> upstream = {1.0, -2.0, 0.5, 3.0, -0.25, 1.5};
    const std::vector<double> output = {0.1, 0.25, 0.5, 0.75, 0.9, 0.05};

    const std::vector<double> expected_tanh = host_tanh_gradient(upstream, output);
    const std::vector<double> expected_sigmoid = host_sigmoid_gradient(upstream, output);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(dy), upstream);
    SaTest::write_tensor(pool.tensor(y), output);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(backend(), graph, pool));
    SA_EXPECT_OK(step->run());

    expect_elements(SaTest::read_tensor(pool.tensor(tanh_dx)), expected_tanh,
                    K_ABSOLUTE_TOLERANCE);
    expect_elements(SaTest::read_tensor(pool.tensor(sigmoid_dx)), expected_sigmoid,
                    K_ABSOLUTE_TOLERANCE);
}
