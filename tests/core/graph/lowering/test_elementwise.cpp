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
 * @file test_elementwise.cpp
 * @brief Tests the elementwise lowering table, the run-time factor of DYNAMIC_SCALE and CAST.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <memory>
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

    /** @brief Bounds a FLOAT32 result computed on the device against a double host value. */
    constexpr double K_ABSOLUTE_TOLERANCE = 1e-6;

    /** @brief Bounds a conversion of values that FLOAT32 holds exactly; it loses nothing. */
    constexpr double K_EXACT = 0.0;

    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        OpID op;           /**< The operation. */
        LoweringRule rule; /**< The row that lowers it. */
    };
} // namespace

/** @brief Tests that each row pairs its operation with the function the header names. */
TEST(Small_ElementwiseLowerings, EachRowPairsItsOperationWithItsLowering)
{
    const std::array<ExpectedRow, 17> expected = {{
        {Ops::K_ADD.id, &Graph::lower_add},
        {Ops::K_SUB.id, &Graph::lower_sub},
        {Ops::K_MUL.id, &Graph::lower_mul},
        {Ops::K_DIV.id, &Graph::lower_div},
        {Ops::K_MINIMUM.id, &Graph::lower_minimum},
        {Ops::K_MAXIMUM.id, &Graph::lower_maximum},
        {Ops::K_SCALE.id, &Graph::lower_scale},
        {Ops::K_DYNAMIC_SCALE.id, &Graph::lower_dynamic_scale},
        {Ops::K_EXP.id, &Graph::lower_exp},
        {Ops::K_ABS.id, &Graph::lower_abs},
        {Ops::K_ATAN.id, &Graph::lower_atan},
        {Ops::K_LESS_EQUAL.id, &Graph::lower_less_equal},
        {Ops::K_SELECT.id, &Graph::lower_select},
        {Ops::K_ZERO.id, &Graph::lower_zero},
        {Ops::K_CONSTANT.id, &Graph::lower_constant},
        {Ops::K_ACCUMULATE.id, &Graph::lower_accumulate},
        {Ops::K_CAST.id, &Graph::lower_cast},
    }};

    ASSERT_EQ(Graph::K_ELEMENTWISE_LOWERINGS.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const Graph::LoweringEntry& row = Graph::K_ELEMENTWISE_LOWERINGS[i];
        EXPECT_EQ(row.op, expected[i].op) << Ops::name_of(expected[i].op);
        EXPECT_EQ(row.rule, expected[i].rule) << Ops::name_of(expected[i].op);
        EXPECT_EQ(row.scratch, nullptr) << Ops::name_of(expected[i].op);
        EXPECT_EQ(Graph::find_lowering(expected[i].op), expected[i].rule)
            << Ops::name_of(expected[i].op);
    }
}

/** @brief Tests that DYNAMIC_SCALE reads its factor when the step runs, not when it is recorded. */
TEST_F(Medium_ElementwiseLowerings, DynamicScaleReadsItsFactorOnEveryRun)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{4}, "x");
    const ValueID factor = builder.input(Shape{1}, "factor");
    const ValueID y = builder.dynamic_scale(x, factor);

    const std::vector<double> values = {1.0, -2.0, 0.5, 4.0};

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(x), values);
    SaTest::write_tensor(pool.tensor(factor), std::vector<double>{2.0});

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));

    for (const double scale : {2.0, -0.5})
    {
        SaTest::write_tensor(pool.tensor(factor), std::vector<double>{scale});
        SA_EXPECT_OK(step->run());

        const std::vector<double> got = SaTest::read_tensor(pool.tensor(y));
        ASSERT_EQ(got.size(), values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            EXPECT_NEAR(got[i], values[i] * scale, K_ABSOLUTE_TOLERANCE) << scale << " at " << i;
    }
}

/** @brief Tests that CAST writes each FLOAT32 operand element into a FLOAT64 result. */
TEST_F(Medium_ElementwiseLowerings, CastConvertsEachElementToTheResultType)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{3}, "x");
    const ValueID wide = builder.cast(x, DType::Float64);

    const std::vector<double> values = {0.25, -1.5, 3.0};

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SaTest::write_tensor(pool.tensor(x), values);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));
    SA_EXPECT_OK(step->run());

    EXPECT_EQ(pool.tensor(wide).dtype, DType::Float64);
    const std::vector<double> got = SaTest::read_tensor(pool.tensor(wide));
    ASSERT_EQ(got.size(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        EXPECT_NEAR(got[i], values[i], K_EXACT) << "at " << i;
}
