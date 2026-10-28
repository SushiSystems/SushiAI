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
 * @file test_activation.cpp
 * @brief Tests the activation lowering table and the gradient rows that read a forward output.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <memory>
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

    /** @brief Bounds a FLOAT32 result computed on the device against a double host value. */
    constexpr double K_ABSOLUTE_TOLERANCE = 1e-6;

    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        OpID op;          /**< The operation. */
        LoweringRule rule; /**< The row that lowers it. */
    };
} // namespace

/** @brief Tests that each row pairs its operation with the function the header names. */
TEST(Small_ActivationLowerings, EachRowPairsItsOperationWithItsLowering)
{
    const std::array<ExpectedRow, 14> expected = {{
        {Ops::K_RELU.id, &Graph::lower_relu},
        {Ops::K_GELU.id, &Graph::lower_gelu},
        {Ops::K_SIGMOID.id, &Graph::lower_sigmoid},
        {Ops::K_TANH.id, &Graph::lower_tanh},
        {Ops::K_SILU.id, &Graph::lower_silu},
        {Ops::K_SOFTPLUS.id, &Graph::lower_softplus},
        {Ops::K_LEAKY_RELU.id, &Graph::lower_leaky_relu},
        {Ops::K_RELU_BACKWARD.id, &Graph::lower_relu_backward},
        {Ops::K_GELU_BACKWARD.id, &Graph::lower_gelu_backward},
        {Ops::K_SIGMOID_BACKWARD.id, &Graph::lower_sigmoid_backward},
        {Ops::K_TANH_BACKWARD.id, &Graph::lower_tanh_backward},
        {Ops::K_SILU_BACKWARD.id, &Graph::lower_silu_backward},
        {Ops::K_SOFTPLUS_BACKWARD.id, &Graph::lower_softplus_backward},
        {Ops::K_LEAKY_RELU_BACKWARD.id, &Graph::lower_leaky_relu_backward},
    }};

    ASSERT_EQ(Graph::K_ACTIVATION_LOWERINGS.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const Graph::LoweringEntry& row = Graph::K_ACTIVATION_LOWERINGS[i];
        EXPECT_EQ(row.op, expected[i].op) << Ops::name_of(expected[i].op);
        EXPECT_EQ(row.rule, expected[i].rule) << Ops::name_of(expected[i].op);
        EXPECT_EQ(row.scratch, nullptr) << Ops::name_of(expected[i].op);
        EXPECT_EQ(Graph::find_lowering(expected[i].op), expected[i].rule)
            << Ops::name_of(expected[i].op);
    }
}

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

    std::vector<double> expected_tanh(6);
    std::vector<double> expected_sigmoid(6);
    for (std::size_t i = 0; i < 6; ++i)
    {
        expected_tanh[i] = upstream[i] * (1.0 - output[i] * output[i]);
        expected_sigmoid[i] = upstream[i] * output[i] * (1.0 - output[i]);
    }

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(dy), upstream);
    SaTest::write_tensor(pool.tensor(y), output);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));
    SA_EXPECT_OK(step->run());

    const std::vector<double> got_tanh = SaTest::read_tensor(pool.tensor(tanh_dx));
    const std::vector<double> got_sigmoid = SaTest::read_tensor(pool.tensor(sigmoid_dx));
    ASSERT_EQ(got_tanh.size(), 6u);
    ASSERT_EQ(got_sigmoid.size(), 6u);
    for (std::size_t i = 0; i < 6; ++i)
    {
        EXPECT_NEAR(got_tanh[i], expected_tanh[i], K_ABSOLUTE_TOLERANCE) << "at " << i;
        EXPECT_NEAR(got_sigmoid[i], expected_sigmoid[i], K_ABSOLUTE_TOLERANCE) << "at " << i;
    }
}
