/**************************************************************************/
/* test_gradients.cpp                                                 */
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
 * @brief Tests the backward rules of batch norm and channel affine against central differences
 *        in FLOAT64.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
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

TEST_F(Medium_Autograd, BatchNormsGradientAccountsForTheStatisticsItComputed)
{
    // dx has three terms, not one; see tests/core/autograd/README.md.
    ValueID gamma = Graph::K_NO_VALUE;
    ValueID beta = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape({4, 3, 3, 5}),
                        [&gamma, &beta](const GraphBuilder& b, ValueID x) -> Graph::Traced
    {
        gamma = b.parameter(Shape{5}, "gamma", DType::Float64);
        beta = b.parameter(Shape{5}, "beta", DType::Float64);
        SA_TRY_ASSIGN(const Graph::BatchNormResult normalised, b.batchnorm2d(x, gamma, beta, 1e-5));
        return b.log_softmax(b.spatial_mean(normalised.output));
    });

    probe.fill_noise(gamma, 471, 1.0);
    probe.fill_noise(beta, 472, 1.0);
    probe.fill_one_hot(4, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(473).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, BatchNormsScaleGradientIsTheSumOfDyTimesTheNormalisedInput)
{
    // The same node, differentiated at gamma; see tests/core/autograd/README.md.
    ValueID x = Graph::K_NO_VALUE;
    ValueID beta = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{5},
                        [&x, &beta](const GraphBuilder& b, ValueID gamma) -> Graph::Traced
    {
        x = b.input(Shape({4, 3, 3, 5}), "x", DType::Float64);
        beta = b.parameter(Shape{5}, "beta", DType::Float64);
        SA_TRY_ASSIGN(const Graph::BatchNormResult normalised, b.batchnorm2d(x, gamma, beta, 1e-5));
        return b.log_softmax(b.spatial_mean(normalised.output));
    });

    probe.fill_noise(x, 481, 1.0);
    probe.fill_noise(beta, 482, 1.0);
    probe.fill_one_hot(4, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(483).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, BatchNormsShiftGradientIsThePlainSumOfDy)
{
    ValueID x = Graph::K_NO_VALUE;
    ValueID gamma = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{5},
                        [&x, &gamma](const GraphBuilder& b, ValueID beta) -> Graph::Traced
    {
        x = b.input(Shape({4, 3, 3, 5}), "x", DType::Float64);
        gamma = b.parameter(Shape{5}, "gamma", DType::Float64);
        SA_TRY_ASSIGN(const Graph::BatchNormResult normalised, b.batchnorm2d(x, gamma, beta, 1e-5));
        return b.log_softmax(b.spatial_mean(normalised.output));
    });

    probe.fill_noise(x, 491, 1.0);
    probe.fill_noise(gamma, 492, 1.0);
    probe.fill_one_hot(4, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(493).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ChannelAffineDifferentiatesAtItsInput)
{
    ValueID gamma = Graph::K_NO_VALUE;
    ValueID beta = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape({2, 4, 4, 6}),
                        [&gamma, &beta](const GraphBuilder& b, ValueID x)
    {
        gamma = b.parameter(Shape{6}, "gamma", DType::Float64);
        beta = b.parameter(Shape{6}, "beta", DType::Float64);
        return b.log_softmax(b.spatial_mean(b.channel_affine(x, gamma, beta)));
    });

    probe.fill_noise(gamma, 501, 1.0);
    probe.fill_noise(beta, 502, 1.0);
    probe.fill_one_hot(2, 6);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(503).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ChannelAffinesScaleGradientSumsOverNHAndW)
{
    // dgamma sums over N, H and W at once; see tests/core/autograd/README.md.
    ValueID x = Graph::K_NO_VALUE;
    ValueID beta = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{6}, [&x, &beta](const GraphBuilder& b, ValueID gamma)
    {
        x = b.input(Shape({2, 4, 4, 6}), "x", DType::Float64);
        beta = b.parameter(Shape{6}, "beta", DType::Float64);
        return b.log_softmax(b.spatial_mean(b.channel_affine(x, gamma, beta)));
    });

    probe.fill_noise(x, 511, 1.0);
    probe.fill_noise(beta, 512, 1.0);
    probe.fill_one_hot(2, 6);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(513).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ChannelAffinesShiftGradientSumsOverNHAndW)
{
    ValueID x = Graph::K_NO_VALUE;
    ValueID gamma = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{6}, [&x, &gamma](const GraphBuilder& b, ValueID beta)
    {
        x = b.input(Shape({2, 4, 4, 6}), "x", DType::Float64);
        gamma = b.parameter(Shape{6}, "gamma", DType::Float64);
        return b.log_softmax(b.spatial_mean(b.channel_affine(x, gamma, beta)));
    });

    probe.fill_noise(x, 521, 1.0);
    probe.fill_noise(gamma, 522, 1.0);
    probe.fill_one_hot(2, 6);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(523).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}
