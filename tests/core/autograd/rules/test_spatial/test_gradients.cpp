/**************************************************************************/
/* test_gradients.cpp                                              */
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
 * @brief Tests the backward rules of im2col, col2im, pooling, upsampling, concat, slice and
 *        depthwise convolution against central differences in FLOAT64.
 * @author Mustafa Garip
 */

#include <array>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/gradient_probe.hpp>
#include <support/test_names.hpp>
#include "../rule_gradient_support.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/autograd/README.md.
namespace SaTest = SushiAI::Test;
using Graph::GraphBuilder;
using Graph::ValueID;
using SaTest::GradientProbe;
using SaTest::expect_gradient_matches;

namespace
{
    /** @brief Names one scale of the bilinear upsample. */
    struct ScaleCase
    {
        const char* name; /**< The case name. */
        int64_t scale;    /**< The scale on both axes. */
    };

    /** @brief Runs the bilinear case once per scale. */
    class Medium_AutogradBilinear : public SaTest::EngineTest,
                                    public ::testing::WithParamInterface<ScaleCase>
    {
    };
} // namespace

TEST_F(Medium_Autograd, AWholeConvolutionDifferentiatesThroughIm2colAndMatmul)
{
    // Single hop: im2col feeds matmul, never col2im; see tests/core/autograd/README.md.
    ValueID weights = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{2, 5, 5, 3},
                        [&weights](const GraphBuilder& builder, ValueID x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const Graph::Traced cols = builder.im2col(x, g);                      // [2*25, 27]
        weights = builder.parameter(Shape{27, 8}, "w", DType::Float64); // 8 output channels
        return builder.log_softmax(builder.matmul(cols, weights));      // [2*25, 8]
    });

    probe.fill_noise(weights, 301, 1.0);
    // A log_softmax probe needs one-hot targets; see tests/core/autograd/README.md.
    probe.fill_one_hot(50, 8);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(302).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, Im2colsGradientAccumulatesEveryWindowThatReadAPixel)
{
    // Nine windows read each interior pixel; see tests/core/autograd/README.md.
    ValueID weights = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{1, 4, 4, 2},
                        [&weights](const GraphBuilder& builder, ValueID x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const Graph::Traced cols = builder.im2col(x, g);
        weights = builder.parameter(Shape{18, 4}, "w", DType::Float64);
        return builder.log_softmax(builder.matmul(cols, weights)); // [16, 4]
    });

    probe.fill_noise(weights, 311, 1.0);
    probe.fill_one_hot(16, 4);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(312).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, AnAsymmetricallyPaddedConvolutionDifferentiatesOverItsRealWindowGrid)
{
    // The real grid is 5x4; a symmetric guess is 4x5; see tests/core/autograd/README.md.
    ValueID weights = Graph::K_NO_VALUE;

    GradientProbe probe(backend(), Shape{1, 4, 5, 1},
                        [&weights](const GraphBuilder& builder, ValueID x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const Graph::Traced cols = builder.im2col_with_extents(x, g, 5, 4);  // [20, 9]
        weights = builder.parameter(Shape{9, 4}, "w", DType::Float64);
        return builder.log_softmax(builder.matmul(cols, weights));     // [20, 4]
    });

    probe.fill_noise(weights, 331, 1.0);
    probe.fill_one_hot(20, 4);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(332).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, Col2imsGradientGathersEveryWindowThatWroteAPixel)
{
    // The variable is the column matrix itself; see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape{9, 18}, [](const GraphBuilder& builder, ValueID x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const Graph::Traced image = builder.col2im(x, Shape{1, 3, 3, 2}, g); // [1, 3, 3, 2]
        return builder.log_softmax(builder.reshape(image, Shape{1, 18}));
    });

    probe.fill_one_hot(1, 18);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(321).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, MaxPoolRoutesEachGradientToTheElementThatWon)
{
    // push_off_kinks leaves no ties in a window; see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape({2, 4, 4, 3}),
                        [](const GraphBuilder& b, ValueID x) -> Graph::Traced
    {
        Graph::ConvGeometry g;
        g.kernel_h = 2; g.kernel_w = 2;
        g.stride_h = 2; g.stride_w = 2;
        SA_TRY_ASSIGN(const Graph::PoolResult pooled, b.maxpool2d(x, g));
        return b.log_softmax(b.spatial_mean(pooled.output));
    });

    probe.fill_one_hot(2, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(401).fill(point, 1.0);
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, OverlappingMaxPoolWindowsAccumulateOnADominantPixel)
{
    // At stride 1 a pixel can win several windows; see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape({1, 5, 5, 2}),
                        [](const GraphBuilder& b, ValueID x) -> Graph::Traced
    {
        Graph::ConvGeometry g;
        g.kernel_h = 2; g.kernel_w = 2;
        SA_TRY_ASSIGN(const Graph::PoolResult pooled, b.maxpool2d(x, g));
        return b.log_softmax(b.spatial_mean(pooled.output));
    });

    probe.fill_one_hot(1, 2);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(411).fill(point, 1.0);
    Autograd::push_off_kinks(point);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, AvgPoolSpreadsItsGradientOverTheWholeWindow)
{
    // One batch row keeps this under 200 inputs; see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape({1, 6, 6, 3}), [](const GraphBuilder& b, ValueID x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.stride_h = 3; g.stride_w = 3;
        return b.log_softmax(b.spatial_mean(b.avgpool2d(x, g)));
    });

    probe.fill_one_hot(1, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(421).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, UpsampleSumsEachSourcePixelsReplicas)
{
    // Scale 2 makes each gradient a sum of four; see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape({2, 3, 3, 4}), [](const GraphBuilder& b, ValueID x)
    {
        return b.log_softmax(b.spatial_mean(b.upsample2d(x, 2, 2)));
    });

    probe.fill_one_hot(2, 4);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(431).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, SpatialMeanBroadcastsScaledByOneOverHW)
{
    // The broadcast is per (n, c); see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape({2, 4, 4, 5}), [](const GraphBuilder& b, ValueID x)
    {
        return b.log_softmax(b.spatial_mean(x));
    });

    probe.fill_one_hot(2, 5);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(441).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, ConcatGivesEachOperandItsOwnChannelRange)
{
    // Operands are reordered against their slices; see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape({2, 4, 4, 6}),
                        [](const GraphBuilder& b, ValueID x) -> Graph::Traced
    {
        SA_TRY_ASSIGN(const ValueID left, b.slice_channels(x, 0, 2));
        SA_TRY_ASSIGN(const ValueID right, b.slice_channels(x, 2, 4));
        const std::array<ValueID, 2> parts = {right, left};   // deliberately reordered
        return b.log_softmax(b.spatial_mean(b.concat_channels(
            SushiRuntime::span<const ValueID>(parts.data(), parts.size()))));
    });

    probe.fill_one_hot(2, 6);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(451).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, SliceScattersItsGradientBackIntoZeros)
{
    // Channels outside the slice receive zero; see tests/core/autograd/README.md.
    GradientProbe probe(backend(), Shape({1, 3, 3, 8}), [](const GraphBuilder& b, ValueID x)
    {
        return b.log_softmax(b.spatial_mean(b.slice_channels(x, 3, 2)));
    });

    probe.fill_one_hot(1, 2);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(461).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

TEST_F(Medium_Autograd, DepthwiseConvolutionDifferentiatesOverItsInputAndItsTaps)
{
    // The image and the taps are slices of one flat variable, so one check covers both.
    constexpr int64_t K_IMAGE_ELEMENTS = 2 * 5 * 5 * 3;
    constexpr int64_t K_TAP_ELEMENTS = 3 * 3 * 3;

    GradientProbe probe(backend(), Shape{1, 1, 1, K_IMAGE_ELEMENTS + K_TAP_ELEMENTS},
                        [](const GraphBuilder& b, ValueID flat)
    {
        const Graph::Traced image = b.reshape(b.slice_channels(flat, 0, K_IMAGE_ELEMENTS),
                                              Shape{2, 5, 5, 3});
        const Graph::Traced taps =
            b.reshape(b.slice_channels(flat, K_IMAGE_ELEMENTS, K_TAP_ELEMENTS), Shape{3, 3, 3});

        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.stride_h = 2; g.stride_w = 2;
        g.pad_top = 1;  g.pad_left = 1;

        const Graph::Traced windows = b.depthwise_conv2d(image, taps, g); // [2, 3, 3, 3]
        return b.log_softmax(b.spatial_mean(windows));                    // [2, 3]
    });

    probe.fill_one_hot(2, 3);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(601).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

/** @brief Tests that the bilinear upsample differentiates against central differences. */
TEST_P(Medium_AutogradBilinear, TheBilinearUpsampleDifferentiatesAtEachScale)
{
    const ScaleCase& row = GetParam();

    // [2, 3, 4, 2] grows to [2, 3s, 4s, 2]; the loss reads its pixels as rows of two channels.
    constexpr int64_t K_BATCH = 2;
    constexpr int64_t K_HEIGHT = 3;
    constexpr int64_t K_WIDTH = 4;
    constexpr int64_t K_CHANNELS = 2;
    const int64_t rows = K_BATCH * K_HEIGHT * row.scale * K_WIDTH * row.scale;

    GradientProbe probe(backend(), Shape{K_BATCH, K_HEIGHT, K_WIDTH, K_CHANNELS},
                        [&row, rows](const GraphBuilder& b, ValueID x)
    {
        return b.reshape(b.upsample2d_bilinear(x, row.scale, row.scale),
                         Shape{rows, K_CHANNELS});
    });

    probe.fill_noise(probe.targets(), 711, 1.0);

    std::vector<double> point(probe.size());
    SaTest::HostNoise(712).fill(point, 1.0);

    expect_gradient_matches(probe, point);
}

INSTANTIATE_TEST_SUITE_P(Scales, Medium_AutogradBilinear,
                         ::testing::Values(ScaleCase{"ScaleTwo", 2}, ScaleCase{"ScaleThree", 3}),
                         [](const ::testing::TestParamInfo<ScaleCase>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });
