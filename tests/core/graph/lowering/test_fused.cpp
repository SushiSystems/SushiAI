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
 * @file test_fused.cpp
 * @brief Tests the fused lowering table and the GEMM row with a bias and a ReLU epilogue.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <string>
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
#include <support/tolerance.hpp>

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

    /**
     * @brief Bounds a FLOAT32 result computed on the device against a double host value; 256 unit
     * roundoffs of float, the smallest power of two that covers 1e-5.
     */
    constexpr double K_ABSOLUTE_TOLERANCE = 256.0 * TestSupport::unit_roundoff<float>();

    /** @brief Compares every element of @p got with @p expected to within @p absolute. */
    void expect_elements(const std::vector<double>& got, const std::vector<double>& expected,
                         double absolute)
    {
        ASSERT_EQ(got.size(), expected.size());

        for (std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(got[i], expected[i], absolute) << "at " << i;
    }

    /**
     * @brief Returns relu(x * w^T + b) taken on the host in double.
     * @param x The [2, 3] input, row major.
     * @param w The [4, 3] weight, row major, read transposed.
     * @param b The [4] bias.
     */
    std::vector<double> host_gemm_bias_relu(const std::vector<double>& x,
                                            const std::vector<double>& w,
                                            const std::vector<double>& b)
    {
        std::vector<double> result(2 * 4);
        for (std::size_t row = 0; row < 2; ++row)
        {
            for (std::size_t column = 0; column < 4; ++column)
            {
                double sum = b[column];
                for (std::size_t k = 0; k < 3; ++k)
                    sum += x[row * 3 + k] * w[column * 3 + k];
                result[row * 4 + column] = std::max(sum, 0.0);
            }
        }
        return result;
    }

    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        std::size_t index; /**< The position of the row in the table. */
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

    /** @brief Runs once per row the header names for the fused table. */
    class Small_FusedRow : public ::testing::TestWithParam<ExpectedRow>
    {

    };

    /** @brief Lists the rows of the fused table in the order the header writes them. */
    const std::array<ExpectedRow, 6> K_EXPECTED_ROWS = {{
        {0, Ops::K_GEMM_BIAS.id, &Graph::lower_gemm_bias},
        {1, Ops::K_GEMM_BIAS_RELU.id, &Graph::lower_gemm_bias_relu},
        {2, Ops::K_GEMM_BIAS_SIGMOID.id, &Graph::lower_gemm_bias_sigmoid},
        {3, Ops::K_GEMM_BIAS_TANH.id, &Graph::lower_gemm_bias_tanh},
        {4, Ops::K_GEMM_BIAS_GELU.id, &Graph::lower_gemm_bias_gelu},
        {5, Ops::K_EXP_SUB_SCALE.id, &Graph::lower_exp_sub_scale},
    }};
} // namespace

/** @brief Tests that the table holds one row for each row the header names, and no other. */
TEST(Small_FusedLowerings, TheTableHoldsExactlyTheRowsTheHeaderNames)
{
    EXPECT_EQ(Graph::K_FUSED_LOWERINGS.size(), K_EXPECTED_ROWS.size());
}

/** @brief Tests that a row pairs its operation with the function the header names. */
TEST_P(Small_FusedRow, EachRowPairsItsOperationWithItsLowering)
{
    const ExpectedRow row = GetParam();
    const std::string name(Ops::name_of(row.op));
    ASSERT_LT(row.index, Graph::K_FUSED_LOWERINGS.size());
    const Graph::LoweringEntry& entry = Graph::K_FUSED_LOWERINGS[row.index];

    EXPECT_EQ(entry.op, row.op) << name;
    EXPECT_EQ(entry.rule, row.rule) << name;
    EXPECT_EQ(entry.scratch, nullptr) << name;
    EXPECT_EQ(Graph::find_lowering(row.op), row.rule) << name;
}

INSTANTIATE_TEST_SUITE_P(, Small_FusedRow, ::testing::ValuesIn(K_EXPECTED_ROWS),
                         [](const ::testing::TestParamInfo<ExpectedRow>& info)
                         {
                             std::string name(Ops::name_of(info.param.op));
                             name.replace(0, name.find('.') + 1, "");
                             return name;
                         });

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
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y,
                            builder.relu(builder.bias_add(builder.matmul(x, w, false, true), b)));

    const std::vector<ValueID> observed{y};
    Graph::FusionOptions options;
    options.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());
    const Graph::FusionReport report = Graph::fuse(graph, options);
    ASSERT_EQ(report.count(FusedForm::GEMMBiasReLU), 1u);

    const std::vector<double> left = {0.5, -1.0, 2.0, 1.5, 0.25, -0.5};
    const std::vector<double> weight = {1.0, 0.5, -0.25, -1.0, 2.0, 0.5,
                                        0.25, 0.25, 0.25, -0.5, -0.5, 1.0};
    const std::vector<double> bias = {0.1, -0.2, 0.3, -0.4};

    const std::vector<double> expected = host_gemm_bias_relu(left, weight, bias);

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

    expect_elements(SaTest::read_tensor(pool.tensor(y)), expected, K_ABSOLUTE_TOLERANCE);
}
