/**************************************************************************/
/* test_spatial.cpp                                                       */
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
 * @file test_spatial.cpp
 * @brief Tests the spatial lowering table and the gradient rows of upsampling and the spatial mean.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/spatial.hpp>
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
    /** @brief Provides the Engine every spatial gradient case records on. */
    class Medium_SpatialLowerings : public SaTest::EngineTest
    {

    };

    /**
     * @brief Bounds a FLOAT32 result computed on the device against a double host value; 32 unit
     * roundoffs of float, the smallest power of two that covers 1e-6.
     */
    constexpr double K_ABSOLUTE_TOLERANCE = 32.0 * TestSupport::unit_roundoff<float>();

    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        std::size_t index; /**< The position of the row in the table. */
        OpID op;           /**< The operation. */
        LoweringRule rule; /**< The row that lowers it. */
    };

    /** @brief Compares every element of @p got with @p expected to within @p absolute. */
    void expect_elements(const std::vector<double>& got, const std::vector<double>& expected,
                         double absolute)
    {
        ASSERT_EQ(got.size(), expected.size());

        for (std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(got[i], expected[i], absolute) << "at " << i;
    }

    /** @brief Writes @p values into @p leaf, runs @p graph once and returns @p out. */
    std::vector<double> run_one(SushiBLAS::Engine& engine, const Graph::Graph& graph, ValueID leaf,
                                const std::vector<double>& values, ValueID out)
    {
        TensorPool pool(engine);
        EXPECT_TRUE(pool.materialise_leaves(graph).is_ok());
        SaTest::write_tensor(pool.tensor(leaf), values);

        Result<std::unique_ptr<CompiledStep>> step = CompiledStep::compile(engine, graph, pool);
        EXPECT_TRUE(step.has_value());
        if (!step.has_value())
            return {};
        EXPECT_TRUE(step.value()->run().is_ok());
        return SaTest::read_tensor(pool.tensor(out));
    }

    /** @brief Runs once per row the header names for the spatial table. */
    class Small_SpatialRow : public ::testing::TestWithParam<ExpectedRow>
    {

    };

    /** @brief Lists the rows of the spatial table in the order the header writes them. */
    const std::array<ExpectedRow, 15> K_EXPECTED_ROWS = {{
        {0, Ops::K_IM2COL.id, &Graph::lower_im2col},
        {1, Ops::K_COL2IM.id, &Graph::lower_col2im},
        {2, Ops::K_MAXPOOL2D.id, &Graph::lower_maxpool2d},
        {3, Ops::K_MAXPOOL2D_BACKWARD.id, &Graph::lower_maxpool2d_backward},
        {4, Ops::K_AVGPOOL2D.id, &Graph::lower_avgpool2d},
        {5, Ops::K_AVGPOOL2D_BACKWARD.id, &Graph::lower_avgpool2d_backward},
        {6, Ops::K_UPSAMPLE2D.id, &Graph::lower_upsample2d},
        {7, Ops::K_UPSAMPLE2D_BACKWARD.id, &Graph::lower_upsample2d_backward},
        {8, Ops::K_SPATIAL_MEAN.id, &Graph::lower_spatial_mean},
        {9, Ops::K_SPATIAL_MEAN_BACKWARD.id, &Graph::lower_spatial_mean_backward},
        {10, Ops::K_CONCAT_CHANNELS.id, &Graph::lower_concat_channels},
        {11, Ops::K_SLICE_CHANNELS.id, &Graph::lower_slice_channels},
        {12, Ops::K_DEPTHWISE_CONV2D.id, &Graph::lower_depthwise_conv2d},
        {13, Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT.id,
         &Graph::lower_depthwise_conv2d_backward_input},
        {14, Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT.id,
         &Graph::lower_depthwise_conv2d_backward_weight},
    }};
} // namespace

/** @brief Tests that the table holds one row for each row the header names, and no other. */
TEST(Small_SpatialLowerings, TheTableHoldsExactlyTheRowsTheHeaderNames)
{
    EXPECT_EQ(Graph::K_SPATIAL_LOWERINGS.size(), K_EXPECTED_ROWS.size());
}

/** @brief Tests that a row pairs its operation with the function the header names. */
TEST_P(Small_SpatialRow, EachRowPairsItsOperationWithItsLowering)
{
    const ExpectedRow row = GetParam();
    const std::string name(Ops::name_of(row.op));
    ASSERT_LT(row.index, Graph::K_SPATIAL_LOWERINGS.size());
    const Graph::LoweringEntry& entry = Graph::K_SPATIAL_LOWERINGS[row.index];

    EXPECT_EQ(entry.op, row.op) << name;
    EXPECT_EQ(entry.rule, row.rule) << name;
    EXPECT_EQ(entry.scratch, nullptr) << name;
    EXPECT_EQ(Graph::find_lowering(row.op), row.rule) << name;
}

INSTANTIATE_TEST_SUITE_P(, Small_SpatialRow, ::testing::ValuesIn(K_EXPECTED_ROWS),
                         [](const ::testing::TestParamInfo<ExpectedRow>& info)
                         {
                             std::string name(Ops::name_of(info.param.op));
                             name.replace(0, name.find('.') + 1, "");
                             return name;
                         });

/** @brief Tests that the upsampling gradient sums each 2 x 2 block of dy onto its pixel. */
TEST_F(Medium_SpatialLowerings, UpsampleGradientSumsEachBlockOntoItsPixel)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID dy = builder.input(Shape{1, 4, 4, 1}, "dy");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID dx,
                            builder.upsample2d_backward(dy, Shape{1, 2, 2, 1}, 2, 2));

    // Each pixel of dx sums its 2 x 2 block of the values 0 to 15: 0 + 1 + 4 + 5, and so on.
    const std::vector<double> upstream = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                                          8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    const std::vector<double> expected = {10.0, 18.0, 42.0, 50.0};

    expect_elements(run_one(*engine, graph, dy, upstream, dx), expected, K_ABSOLUTE_TOLERANCE);
}

/** @brief Tests that the spatial mean gradient writes dy[n, c] / (H * W) at every pixel. */
TEST_F(Medium_SpatialLowerings, SpatialMeanGradientSpreadsEvenlyOverThePlane)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID dy = builder.input(Shape{1, 2}, "dy");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID dx, builder.spatial_mean_backward(dy, Shape{1, 2, 2, 2}));

    const std::vector<double> upstream = {4.0, -2.0};

    // Each of the four pixels holds (4 / 4, -2 / 4), channels interleaved.
    const std::vector<double> expected = {1.0, -0.5, 1.0, -0.5, 1.0, -0.5, 1.0, -0.5};

    expect_elements(run_one(*engine, graph, dy, upstream, dx), expected, K_ABSOLUTE_TOLERANCE);
}
