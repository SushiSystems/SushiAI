/**************************************************************************/
/* test_elementwise.cpp                                                   */
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
 * @file test_elementwise.cpp
 * @brief Tests the elementwise lowering table, the run-time factor of DYNAMIC_SCALE and CAST.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/elementwise.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::LoweringRule;
using Graph::TensorPool;
using Graph::ValueID;

namespace
{
    /** @brief Provides the Engine every elementwise case records on. */
    class Medium_ElementwiseLowerings : public SaTest::EngineTest
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

    /** @brief Returns every element of @p values times @p factor. */
    std::vector<double> scaled(const std::vector<double>& values, double factor)
    {
        std::vector<double> result(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            result[i] = values[i] * factor;
        return result;
    }

    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        std::size_t index; /**< The position of the row in the table. */
        OpID op;           /**< The operation. */
        LoweringRule rule; /**< The row that lowers it. */
    };

    /** @brief Runs once per row the header names for the elementwise table. */
    class Small_ElementwiseRow : public ::testing::TestWithParam<ExpectedRow>
    {

    };

    /** @brief Lists the rows of the elementwise table in the order the header writes them. */
    const std::array<ExpectedRow, 17> K_EXPECTED_ROWS = {{
        {0, Ops::K_ADD.id, &Graph::lower_add},
        {1, Ops::K_SUB.id, &Graph::lower_sub},
        {2, Ops::K_MUL.id, &Graph::lower_mul},
        {3, Ops::K_DIV.id, &Graph::lower_div},
        {4, Ops::K_MINIMUM.id, &Graph::lower_minimum},
        {5, Ops::K_MAXIMUM.id, &Graph::lower_maximum},
        {6, Ops::K_SCALE.id, &Graph::lower_scale},
        {7, Ops::K_DYNAMIC_SCALE.id, &Graph::lower_dynamic_scale},
        {8, Ops::K_EXP.id, &Graph::lower_exp},
        {9, Ops::K_ABS.id, &Graph::lower_abs},
        {10, Ops::K_ATAN.id, &Graph::lower_atan},
        {11, Ops::K_LESS_EQUAL.id, &Graph::lower_less_equal},
        {12, Ops::K_SELECT.id, &Graph::lower_select},
        {13, Ops::K_ZERO.id, &Graph::lower_zero},
        {14, Ops::K_CONSTANT.id, &Graph::lower_constant},
        {15, Ops::K_ACCUMULATE.id, &Graph::lower_accumulate},
        {16, Ops::K_CAST.id, &Graph::lower_cast},
    }};
} // namespace

/** @brief Tests that the table holds one row for each row the header names, and no other. */
TEST(Small_ElementwiseLowerings, TheTableHoldsExactlyTheRowsTheHeaderNames)
{
    EXPECT_EQ(Graph::K_ELEMENTWISE_LOWERINGS.size(), K_EXPECTED_ROWS.size());
}

/** @brief Tests that a row pairs its operation with the function the header names. */
TEST_P(Small_ElementwiseRow, EachRowPairsItsOperationWithItsLowering)
{
    const ExpectedRow row = GetParam();
    const std::string name(Ops::name_of(row.op));
    ASSERT_LT(row.index, Graph::K_ELEMENTWISE_LOWERINGS.size());
    const Graph::LoweringEntry& entry = Graph::K_ELEMENTWISE_LOWERINGS[row.index];

    EXPECT_EQ(entry.op, row.op) << name;
    EXPECT_EQ(entry.rule, row.rule) << name;
    EXPECT_EQ(entry.scratch, nullptr) << name;
    EXPECT_EQ(Graph::find_lowering(row.op), row.rule) << name;
}

INSTANTIATE_TEST_SUITE_P(, Small_ElementwiseRow, ::testing::ValuesIn(K_EXPECTED_ROWS),
                         [](const ::testing::TestParamInfo<ExpectedRow>& info)
                         {
                             std::string name(Ops::name_of(info.param.op));
                             name.replace(0, name.find('.') + 1, "");
                             return name;
                         });

/** @brief Tests that DYNAMIC_SCALE reads its factor when the step runs, not when it is recorded. */
TEST_F(Medium_ElementwiseLowerings, DynamicScaleReadsItsFactorOnEveryRun)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4}, "x");
    const ValueID factor = builder.input(Shape{1}, "factor");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.dynamic_scale(x, factor));

    const std::vector<double> values = {1.0, -2.0, 0.5, 4.0};

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(x), values);
    SaTest::write_tensor(pool.tensor(factor), std::vector<double>{2.0});

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(backend(), graph, pool));

    SaTest::write_tensor(pool.tensor(factor), std::vector<double>{2.0});
    SA_EXPECT_OK(step->run());
    expect_elements(SaTest::read_tensor(pool.tensor(y)), scaled(values, 2.0),
                    K_ABSOLUTE_TOLERANCE);

    SaTest::write_tensor(pool.tensor(factor), std::vector<double>{-0.5});
    SA_EXPECT_OK(step->run());
    expect_elements(SaTest::read_tensor(pool.tensor(y)), scaled(values, -0.5),
                    K_ABSOLUTE_TOLERANCE);
}

/** @brief Tests that CAST writes each FLOAT32 operand element into a FLOAT64 result. */
TEST_F(Medium_ElementwiseLowerings, CastConvertsEachElementToTheResultType)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID wide, builder.cast(x, DType::Float64));

    const std::vector<double> values = {0.25, -1.5, 3.0};

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(x), values);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(backend(), graph, pool));
    SA_EXPECT_OK(step->run());

    EXPECT_EQ(pool.tensor(wide).dtype, DType::Float64);
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(wide)), values);
}
