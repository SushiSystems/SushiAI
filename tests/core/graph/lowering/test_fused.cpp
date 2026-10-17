/**************************************************************************/
/* test_fused.cpp                                                         */
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
 * @file test_fused.cpp
 * @brief Tests the fused lowering table and the GEMM row with a bias and a ReLU epilogue.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/fused.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::FusedForm;
using Graph::GraphBuilder;
using Graph::LoweringRule;
using Graph::TensorPool;
using Graph::ValueID;

namespace
{
    /** @brief Provides the Engine every fused case records on. */
    class Medium_FusedLowerings : public SaTest::EngineTest
    {

    };

    /** @brief Bounds a FLOAT32 result computed on the device against a double host value. */
    constexpr double K_ABSOLUTE_TOLERANCE = 1e-5;

    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        OpID op;           /**< The operation. */
        LoweringRule rule; /**< The row that lowers it. */
    };

    /** @brief Takes every match whole, so the fused row is the one that records. */
    class TakeTheMatch final : public Graph::FusionSelector
    {
        public:
            /** @brief Returns the matched form with a fixed reason. */
            Graph::FusionChoice choose(const Graph::FusionQuery& query) const override
            {
                return Graph::FusionChoice{query.matched, "the test takes every match"};
            }
    };
} // namespace

/** @brief Tests that each row pairs its operation with the function the header names. */
TEST(Small_FusedLowerings, EachRowPairsItsOperationWithItsLowering)
{
    const std::array<ExpectedRow, 6> expected = {{
        {Ops::K_GEMM_BIAS.id, &Graph::lower_gemm_bias},
        {Ops::K_GEMM_BIAS_RELU.id, &Graph::lower_gemm_bias_relu},
        {Ops::K_GEMM_BIAS_SIGMOID.id, &Graph::lower_gemm_bias_sigmoid},
        {Ops::K_GEMM_BIAS_TANH.id, &Graph::lower_gemm_bias_tanh},
        {Ops::K_GEMM_BIAS_GELU.id, &Graph::lower_gemm_bias_gelu},
        {Ops::K_EXP_SUB_SCALE.id, &Graph::lower_exp_sub_scale},
    }};

    ASSERT_EQ(Graph::K_FUSED_LOWERINGS.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const Graph::LoweringEntry& row = Graph::K_FUSED_LOWERINGS[i];
        EXPECT_EQ(row.op, expected[i].op) << Ops::name_of(expected[i].op);
        EXPECT_EQ(row.rule, expected[i].rule) << Ops::name_of(expected[i].op);
        EXPECT_EQ(row.scratch, nullptr) << Ops::name_of(expected[i].op);
        EXPECT_EQ(Graph::find_lowering(expected[i].op), expected[i].rule)
            << Ops::name_of(expected[i].op);
    }
}

/**
 * @brief Tests that GEMM_BIAS_RELU, taken whole, writes relu(x * w^T + b).
 *
 * x is [2, 3], w is [4, 3] read transposed and b is [4]; the pattern puts entries on both
 * sides of zero, so the ReLU clips some of them.
 */
TEST_F(Medium_FusedLowerings, TheGEMMWithBiasAndReLUMatchesTheHost)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{2, 3}, "x");
    const ValueID w = builder.parameter(Shape{4, 3}, "w");
    const ValueID b = builder.parameter(Shape{4}, "b");
    const ValueID y = builder.relu(builder.bias_add(builder.matmul(x, w, false, true), b));

    const std::vector<ValueID> observed{y};
    Graph::FusionOptions options;
    options.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());
    const Graph::FusionReport report = Graph::fuse(graph, options);
    ASSERT_EQ(report.count(FusedForm::GEMMBiasReLU), 1u);

    const std::vector<double> left = {0.5, -1.0, 2.0, 1.5, 0.25, -0.5};
    const std::vector<double> weight = {1.0, 0.5, -0.25, -1.0, 2.0, 0.5,
                                        0.25, 0.25, 0.25, -0.5, -0.5, 1.0};
    const std::vector<double> bias = {0.1, -0.2, 0.3, -0.4};

    std::vector<double> expected(2 * 4);
    for (std::size_t row = 0; row < 2; ++row)
    {
        for (std::size_t column = 0; column < 4; ++column)
        {
            double sum = bias[column];
            for (std::size_t k = 0; k < 3; ++k)
                sum += left[row * 3 + k] * weight[column * 3 + k];
            expected[row * 4 + column] = std::max(sum, 0.0);
        }
    }

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(x), left);
    SaTest::write_tensor(pool.tensor(w), weight);
    SaTest::write_tensor(pool.tensor(b), bias);

    const TakeTheMatch selector;
    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step,
                            CompiledStep::compile(*engine, graph, pool, nullptr, &selector));
    SA_EXPECT_OK(step->run());

    ASSERT_EQ(step->selections().size(), 1u);
    EXPECT_EQ(step->selections()[0].chosen, FusedForm::GEMMBiasReLU);

    const std::vector<double> got = SaTest::read_tensor(pool.tensor(y));
    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(got[i], expected[i], K_ABSOLUTE_TOLERANCE) << "at " << i;
}
