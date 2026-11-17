/**************************************************************************/
/* test_gradients.cpp                                               */
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
 * @file test_gradients.cpp
 * @brief Tests the backward rules of matmul, bias add, sum rows, gather, scatter, take and put
 *        against central differences in FLOAT64.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/gradient_probe.hpp>
#include "../rule_gradient_support.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/autograd/README.md.
namespace SaTest = SushiAI::Test;
using Graph::GraphBuilder;
using Graph::ValueID;
using SaTest::GradientProbe;
using SaTest::expect_gradient_matches;

TEST_F(Medium_Autograd, MatmulWithoutTransposes)
{
    ValueID other = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{4, 5}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{5, 3}, "b", DType::Float64);
        return builder.matmul(x, other, false, false);
    });

    probe.fill_noise(other, 11, 1.0);
    probe.fill_noise(probe.targets(), 12, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(13).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, MatmulWithATransposedRightOperand)
{
    // The shape Linear actually emits: X [N, in] against W [out, in]^T.
    ValueID weight = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{4, 5}, [&weight](const GraphBuilder& builder, ValueID x)
    {
        weight = builder.input(Shape{3, 5}, "w", DType::Float64);
        return builder.matmul(x, weight, false, true);
    });

    probe.fill_noise(weight, 21, 1.0);
    probe.fill_noise(probe.targets(), 22, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(23).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, MatmulWithATransposedLeftOperand)
{
    ValueID other = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{5, 4}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{5, 3}, "b", DType::Float64);
        return builder.matmul(x, other, true, false);
    });

    probe.fill_noise(other, 31, 1.0);
    probe.fill_noise(probe.targets(), 32, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(33).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, MatmulWithBothOperandsTransposed)
{
    ValueID other = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{5, 4}, [&other](const GraphBuilder& builder, ValueID x)
    {
        other = builder.input(Shape{3, 5}, "b", DType::Float64);
        return builder.matmul(x, other, true, true);
    });

    probe.fill_noise(other, 41, 1.0);
    probe.fill_noise(probe.targets(), 42, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(43).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, MatmulGradientOfTheRightOperand)
{
    // The weight is the variable, covering dB; see tests/core/autograd/README.md.
    ValueID activations = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{3, 5},
                        [&activations](const GraphBuilder& builder, ValueID w)
    {
        activations = builder.input(Shape{4, 5}, "x", DType::Float64);
        return builder.matmul(activations, w, false, true);
    });

    probe.fill_noise(activations, 51, 1.0);
    probe.fill_noise(probe.targets(), 52, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(53).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, BiasAddPassesTheMatrixGradientThrough)
{
    ValueID bias = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{4, 3}, [&bias](const GraphBuilder& builder, ValueID x)
    {
        bias = builder.input(Shape{3}, "b", DType::Float64);
        return builder.bias_add(x, bias);
    });

    probe.fill_noise(bias, 61, 1.0);
    probe.fill_noise(probe.targets(), 62, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(63).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, BiasAddSumsOverTheBroadcastAxis)
{
    ValueID base = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{3}, [&base](const GraphBuilder& builder, ValueID bias)
    {
        base = builder.input(Shape{4, 3}, "x", DType::Float64);
        return builder.bias_add(base, bias);
    });

    probe.fill_noise(base, 71, 1.0);
    probe.fill_noise(probe.targets(), 72, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(73).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, SumRowsBroadcastsItsGradientBack)
{
    GradientProbe probe(backend(), Shape{4, 3}, [](const GraphBuilder& builder, ValueID x)
    {
        // [4, 3] -> [3] -> [1, 3], so the loss can consume it as a one-row batch.
        const Graph::Traced sums = builder.sum_rows(x);
        return builder.bias_add(builder.zeros(Shape{1, 3}, DType::Float64), sums);
    });

    probe.fill_noise(probe.targets(), 191, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(192).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, GatherRowsScatterAddsARepeatedIndexIntoItsTable)
{
    ValueID indices = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{5, 3}, [&indices](const GraphBuilder& builder, ValueID x)
    {
        indices = builder.input(Shape{6}, "indices", DType::Int32);
        return builder.gather_rows(x, indices);
    });

    // Row 3 is read three times and row 1 twice; rows 2 and 4 are read by no index.
    probe.fill_indices(indices, {3, 1, 3, 3, 0, 1});
    probe.fill_noise(probe.targets(), 193, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(194).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, GatherRowsOverARankTwoIndexDifferentiatesThroughItsReshape)
{
    ValueID indices = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{4, 3}, [&indices](const GraphBuilder& builder, ValueID x)
    {
        // [4, 3] -> [2, 3, 3] -> [6, 3], so the loss can consume the tokens as rows.
        indices = builder.input(Shape{2, 3}, "indices", DType::Int32);
        return builder.reshape(builder.gather_rows(x, indices), Shape{6, 3});
    });

    probe.fill_indices(indices, {2, 2, 0, 3, 2, 1});
    probe.fill_noise(probe.targets(), 195, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(196).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ScatterAddRowsGathersARepeatedIndexBackOntoItsRows)
{
    ValueID indices = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{6, 3}, [&indices](const GraphBuilder& builder, ValueID x)
    {
        indices = builder.input(Shape{6}, "indices", DType::Int32);
        return builder.scatter_add_rows(x, indices, 5);
    });

    // Row 3 is written three times and row 1 twice; rows 2 and 4 are written by no index.
    probe.fill_indices(indices, {3, 1, 3, 3, 0, 1});
    probe.fill_noise(probe.targets(), 197, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(198).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ScatterAddRowsOverARankTwoIndexGathersInTheRowsOwnShape)
{
    ValueID indices = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{2, 3, 3},
                        [&indices](const GraphBuilder& builder, ValueID x)
    {
        // [2, 3, 3] rows over a [2, 3] index sum into the [4, 3] the loss reads as rows.
        indices = builder.input(Shape{2, 3}, "indices", DType::Int32);
        return builder.scatter_add_rows(x, indices, 4);
    });

    probe.fill_indices(indices, {2, 2, 0, 3, 2, 1});
    probe.fill_noise(probe.targets(), 199, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(200).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, WholeTwoLayerModelUnderRandomDirections)
{
    // Sixteen random directions cost 2k runs, not 2n; see tests/core/autograd/README.md.
    ValueID hidden_weight = Graph::K_NO_VALUE;
    ValueID output_weight = Graph::K_NO_VALUE;
    ValueID inputs = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{6}, [&](const GraphBuilder& builder, ValueID bias)
    {
        inputs = builder.input(Shape{4, 5}, "x", DType::Float64);
        hidden_weight = builder.input(Shape{6, 5}, "w0", DType::Float64);
        output_weight = builder.input(Shape{3, 6}, "w1", DType::Float64);

        const Graph::Traced hidden = builder.relu(
            builder.bias_add(builder.matmul(inputs, hidden_weight, false, true), bias));

        return builder.log_softmax(builder.matmul(hidden, output_weight, false, true));
    });

    probe.fill_noise(inputs, 221, 1.0);
    probe.fill_noise(hidden_weight, 222, 0.6);
    probe.fill_noise(output_weight, 223, 0.6);
    probe.fill_one_hot(4, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(224).fill(point, 0.5);
    Autograd::push_off_kinks(point);

    const Autograd::GradCheckResult result =
        Autograd::check_directional(probe.value_fn(), probe.grad_fn(), point, 16);

    EXPECT_TRUE(result.passed) << result.to_string();
}

/** @brief Tests that the pick of one element per row sends its gradient to the picked column. */
TEST_F(Medium_Autograd, TakeAlongPutsItsGradientAtThePickedColumn)
{
    ValueID index = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{5, 4}, [&index](const GraphBuilder& b, ValueID x)
    {
        // Column 3 is picked by two rows, and column 2 by one.
        index = b.input(Shape{5}, "index", DType::Int32);
        return b.reshape(b.take_along(x, index), Shape{5, 1});
    });

    probe.fill_indices(index, {0, 3, 3, 1, 2});
    probe.fill_noise(probe.targets(), 741, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(742).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

/** @brief Tests that the write of one element per row takes its gradient from the put column. */
TEST_F(Medium_Autograd, PutAlongTakesItsGradientFromThePutColumn)
{
    ValueID index = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{5}, [&index](const GraphBuilder& b, ValueID x)
    {
        index = b.input(Shape{5}, "index", DType::Int32);
        return b.put_along(x, index, 4);
    });

    probe.fill_indices(index, {0, 3, 3, 1, 2});
    probe.fill_noise(probe.targets(), 743, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(744).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}
