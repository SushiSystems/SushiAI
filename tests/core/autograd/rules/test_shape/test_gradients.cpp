/**************************************************************************/
/* test_gradients.cpp                                                */
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
 * @brief Tests the backward rules of reshape, transpose and layout convert against central differences in
 *        FLOAT64.
 * @author Mustafa Garip
 */

#include <array>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
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

TEST_F(Medium_Autograd, ReshapeMovesItsGradientBackUnchanged)
{
    // The gradient returns in the input shape; see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape{3, 8}, [](const GraphBuilder& builder, ValueID x)
    {
        return builder.log_softmax(builder.reshape(builder.reshape(x, Shape{4, 6}), Shape{3, 8}));
    });

    // The fused NLL rule needs one-hot targets; see tests/core/autograd/README.md.
    probe.fill_one_hot(3, 8);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(232).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, TransposeMovesItsGradientBackThroughTheInversePermutation)
{
    // The cycle (1, 2, 0) differs from its inverse (2, 0, 1); see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape{2, 3, 4, 5}, [](const GraphBuilder& builder, ValueID x)
    {
        const std::array<int64_t, 3> cycle = {1, 2, 0};
        const Graph::Traced moved = builder.transpose(
            x, SushiRuntime::span<const int64_t>(cycle.data(), cycle.size()));
        return builder.log_softmax(builder.reshape(moved, Shape{2, 60}));
    });

    probe.fill_one_hot(2, 60);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(317).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, LayoutConvertsGradientIsThePermutedGradient)
{
    // A single layout_convert, so the order matters; see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape{2, 3, 4, 5}, [](const GraphBuilder& builder, ValueID x)
    {
        builder.graph().value(x).layout = Graph::TensorLayout::NCHW;
        const Graph::Traced nhwc = builder.layout_convert(x, Graph::TensorLayout::NHWC);
        return builder.log_softmax(builder.reshape(nhwc, Shape{2, 60}));
    });

    probe.fill_one_hot(2, 60);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(291).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, LayoutConvertRoundTripComposesBothDirections)
{
    // The round trip is a secondary check only; see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape{2, 3, 4, 5}, [](const GraphBuilder& builder, ValueID x)
    {
        const Graph::Traced nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);
        const Graph::Traced back = builder.layout_convert(nchw, Graph::TensorLayout::NHWC);
        return builder.log_softmax(builder.reshape(back, Shape{2, 60}));
    });

    probe.fill_one_hot(2, 60);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(242).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}
