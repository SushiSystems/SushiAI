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
 * @file test_spatial.cpp
 * @brief Tests the spatial lowering table and the gradient rows of upsampling and the spatial mean.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <memory>
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

    /** @brief Bounds a FLOAT32 result computed on the device against a double host value. */
    constexpr double K_ABSOLUTE_TOLERANCE = 1e-6;

    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        OpID op;           /**< The operation. */
        LoweringRule rule; /**< The row that lowers it. */
    };

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
} // namespace

/** @brief Tests that each row pairs its operation with the function the header names. */
TEST(Small_SpatialLowerings, EachRowPairsItsOperationWithItsLowering)
{
    const std::array<ExpectedRow, 15> expected = {{
        {Ops::K_IM2COL.id, &Graph::lower_im2col},
        {Ops::K_COL2IM.id, &Graph::lower_col2im},
        {Ops::K_MAXPOOL2D.id, &Graph::lower_maxpool2d},
        {Ops::K_MAXPOOL2D_BACKWARD.id, &Graph::lower_maxpool2d_backward},
        {Ops::K_AVGPOOL2D.id, &Graph::lower_avgpool2d},
        {Ops::K_AVGPOOL2D_BACKWARD.id, &Graph::lower_avgpool2d_backward},
        {Ops::K_UPSAMPLE2D.id, &Graph::lower_upsample2d},
        {Ops::K_UPSAMPLE2D_BACKWARD.id, &Graph::lower_upsample2d_backward},
        {Ops::K_SPATIAL_MEAN.id, &Graph::lower_spatial_mean},
        {Ops::K_SPATIAL_MEAN_BACKWARD.id, &Graph::lower_spatial_mean_backward},
        {Ops::K_CONCAT_CHANNELS.id, &Graph::lower_concat_channels},
        {Ops::K_SLICE_CHANNELS.id, &Graph::lower_slice_channels},
        {Ops::K_DEPTHWISE_CONV2D.id, &Graph::lower_depthwise_conv2d},
        {Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT.id, &Graph::lower_depthwise_conv2d_backward_input},
        {Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT.id,
         &Graph::lower_depthwise_conv2d_backward_weight},
    }};

    ASSERT_EQ(Graph::K_SPATIAL_LOWERINGS.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const Graph::LoweringEntry& row = Graph::K_SPATIAL_LOWERINGS[i];
        EXPECT_EQ(row.op, expected[i].op) << Ops::name_of(expected[i].op);
        EXPECT_EQ(row.rule, expected[i].rule) << Ops::name_of(expected[i].op);
        EXPECT_EQ(row.scratch, nullptr) << Ops::name_of(expected[i].op);
        EXPECT_EQ(Graph::find_lowering(expected[i].op), expected[i].rule)
            << Ops::name_of(expected[i].op);
    }
}

/** @brief Tests that the upsampling gradient sums each 2 x 2 block of dy onto its pixel. */
TEST_F(Medium_SpatialLowerings, UpsampleGradientSumsEachBlockOntoItsPixel)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID dy = builder.input(Shape{1, 4, 4, 1}, "dy");
    const ValueID dx = builder.upsample2d_backward(dy, Shape{1, 2, 2, 1}, 2, 2);

    std::vector<double> upstream(16);
    for (std::size_t i = 0; i < upstream.size(); ++i)
        upstream[i] = static_cast<double>(i);

    std::vector<double> expected(4, 0.0);
    for (std::size_t h = 0; h < 4; ++h)
    {
        for (std::size_t w = 0; w < 4; ++w)
            expected[(h / 2) * 2 + w / 2] += upstream[h * 4 + w];
    }

    const std::vector<double> got = run_one(*engine, graph, dy, upstream, dx);
    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(got[i], expected[i], K_ABSOLUTE_TOLERANCE) << "at " << i;
}

/** @brief Tests that the spatial mean gradient writes dy[n, c] / (H * W) at every pixel. */
TEST_F(Medium_SpatialLowerings, SpatialMeanGradientSpreadsEvenlyOverThePlane)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID dy = builder.input(Shape{1, 2}, "dy");
    const ValueID dx = builder.spatial_mean_backward(dy, Shape{1, 2, 2, 2});

    const std::vector<double> upstream = {4.0, -2.0};

    const std::vector<double> got = run_one(*engine, graph, dy, upstream, dx);
    ASSERT_EQ(got.size(), 8u);
    for (std::size_t pixel = 0; pixel < 4; ++pixel)
    {
        EXPECT_NEAR(got[pixel * 2 + 0], 1.0, K_ABSOLUTE_TOLERANCE) << "pixel " << pixel;
        EXPECT_NEAR(got[pixel * 2 + 1], -0.5, K_ABSOLUTE_TOLERANCE) << "pixel " << pixel;
    }
}
