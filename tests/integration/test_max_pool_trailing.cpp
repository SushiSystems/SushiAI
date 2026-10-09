/**************************************************************************/
/* test_max_pool_trailing.cpp                                             */
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
 * @file test_max_pool_trailing.cpp
 * @brief Tests the engine's max pool with a trailing pad against the truncated windows' maxima.
 * @author Mustafa Garip
 */

#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/max_pool2d.hpp>
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
    class Integration_MaxPoolTrailing : public SaTest::EngineTest {};

    /** @brief Builds a 2x2 window at stride one with one trailing row and one trailing column. */
    NN::MaxPool2d trailing_window()
    {
        NN::MaxPool2d layer;
        layer.geometry.kernel_h = 2;
        layer.geometry.kernel_w = 2;
        layer.geometry.stride_h = 1;
        layer.geometry.stride_w = 1;
        layer.trailing_pad_h = 1;
        layer.trailing_pad_w = 1;
        return layer;
    }

    /**
     * @brief Pools a 3x3 single-channel input with trailing_window() and reads the 3x3 result.
     * @param input The 9 input values, row-major.
     * @param pooled Receives the 9 pooled values, row-major.
     */
    void pool_three_by_three(SushiBLAS::Engine& engine, const std::vector<double>& input,
                             std::vector<double>& pooled)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueId x = builder.input(Shape{1, 3, 3, 1}, "x");
        const ValueId y = trailing_window().forward(builder, x);

        TensorPool pool(engine);
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        write_tensor(pool.tensor(x), input);

        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(engine, graph, pool));
        SA_EXPECT_OK(step->run());

        pooled = read_tensor(pool.tensor(y));
    }
} // namespace

TEST_F(Integration_MaxPoolTrailing, TrailingWindowsTakeTheMaximaOfTheirTruncatedTaps)
{
    // Input [[5, -1, 2], [-3, 8, 0], [4, -7, 6]]; expected [[8, 8, 2], [8, 8, 6], [4, 6, 6]].
    const std::vector<double> input = {5, -1, 2, -3, 8, 0, 4, -7, 6};
    const std::vector<double> expected = {8, 8, 2, 8, 8, 6, 4, 6, 6};

    std::vector<double> pooled;
    ASSERT_NO_FATAL_FAILURE(pool_three_by_three(*engine, input, pooled));

    ASSERT_EQ(pooled.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(pooled[i], expected[i], 1e-6) << "at " << i;
}

TEST_F(Integration_MaxPoolTrailing, AllNegativeTrailingWindowsKeepTheirTrueMaximum)
{
    // Input [[-18, -17, -16], [-15, -14, -13], [-12, -11, -10]]; a zero pad would show as 0.
    const std::vector<double> input = {-18, -17, -16, -15, -14, -13, -12, -11, -10};
    const std::vector<double> expected = {-14, -13, -13, -11, -10, -10, -11, -10, -10};

    std::vector<double> pooled;
    ASSERT_NO_FATAL_FAILURE(pool_three_by_three(*engine, input, pooled));

    ASSERT_EQ(pooled.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(pooled[i], expected[i], 1e-6) << "at " << i;
}
