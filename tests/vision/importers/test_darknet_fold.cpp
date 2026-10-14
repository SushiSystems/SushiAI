/**************************************************************************/
/* test_darknet_fold.cpp                                                  */
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
 * @file test_darknet_fold.cpp
 * @brief Tests the batch-norm fold against the formula, its agreement with the unfolded convolution
 *        on the host, and its refusals.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/importers/darknet_fold.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

namespace
{
    using SushiAI::Vision::Importers::DarknetConvolution;
    using SushiAI::Vision::Importers::fold_darknet_normalisation;

    /** @brief Holds the epsilon the batch-norm formula adds under the variance. */
    constexpr long double K_EPSILON = 0.00001L;

    /** @brief Holds the seed of every random draw in this file. */
    constexpr std::uint32_t K_SEED = 20261009u;

    /** @brief Holds the geometry of the host convolution: filters, channels, kernel side. */
    constexpr std::size_t K_FILTERS = 3;
    constexpr std::size_t K_CHANNELS = 2;
    constexpr std::size_t K_SIDE = 3;

    /** @brief Holds the input height and width; the convolution keeps them with pad 1. */
    constexpr std::size_t K_HEIGHT = 5;
    constexpr std::size_t K_WIDTH = 5;

    /** @brief Draws @p count values uniformly from [low, high) with @p engine. */
    std::vector<float> uniform_values(std::mt19937& engine, std::size_t count, float low,
                                      float high)
    {
        std::uniform_real_distribution<float> distribution(low, high);
        std::vector<float> values(count);
        for (float& value : values)
            value = distribution(engine);
        return values;
    }

    /**
     * @brief Returns the 3x3, pad 1, stride 1 convolution of @p input at one output pixel.
     *
     * The weights are [filter, channel, kh, kw] and the input is [channel, height, width]; the
     * sum is taken in double so the test's own rounding stays far below its tolerance.
     */
    double convolve_at(const std::vector<float>& weights, const std::vector<float>& input,
                       std::size_t filter, std::size_t y, std::size_t x)
    {
        double sum = 0.0;
        for (std::size_t c = 0; c < K_CHANNELS; ++c)
        {
            for (std::size_t ky = 0; ky < K_SIDE; ++ky)
            {
                for (std::size_t kx = 0; kx < K_SIDE; ++kx)
                {
                    // The input index is shifted by the pad of one; a tap outside it adds nothing.
                    const std::int64_t in_y = static_cast<std::int64_t>(y + ky) - 1;
                    const std::int64_t in_x = static_cast<std::int64_t>(x + kx) - 1;
                    if (in_y < 0 || in_x < 0 || in_y >= static_cast<std::int64_t>(K_HEIGHT)
                        || in_x >= static_cast<std::int64_t>(K_WIDTH))
                        continue;

                    const std::size_t w = ((filter * K_CHANNELS + c) * K_SIDE + ky) * K_SIDE + kx;
                    const std::size_t p = (c * K_HEIGHT + static_cast<std::size_t>(in_y)) * K_WIDTH
                                          + static_cast<std::size_t>(in_x);
                    sum += static_cast<double>(weights[w]) * static_cast<double>(input[p]);
                }
            }
        }
        return sum;
    }

} // namespace

/**
 * @brief Tests that a one-filter fold equals the formula evaluated in long double.
 *
 * The tolerance is 1e-6 relative plus 1e-9 absolute, one float32 rounding of slack.
 */
TEST(Unit_DarknetFold, OneFilterFoldMatchesTheFormulaInLongDouble)
{
    const std::vector<float> original_weights = {0.5f, -1.25f, 2.0f, 0.125f,
                                                 -0.75f, 1.0f, 0.3f, -2.5f};
    DarknetConvolution convolution;
    convolution.biases = {0.75f};
    convolution.scales = {1.5f};
    convolution.rolling_mean = {0.25f};
    convolution.rolling_variance = {0.5f};
    convolution.weights = original_weights;

    const long double root = std::sqrt(0.5L + K_EPSILON);
    const long double factor = 1.5L / root;
    const long double expected_bias = 0.75L - 1.5L * 0.25L / root;

    fold_darknet_normalisation(convolution);

    ASSERT_EQ(convolution.weights.size(), original_weights.size());
    ASSERT_EQ(convolution.biases.size(), 1u);
    const double bias_tolerance = 1e-6 * std::fabs(static_cast<double>(expected_bias)) + 1e-9;
    EXPECT_NEAR(convolution.biases[0], static_cast<double>(expected_bias), bias_tolerance);

    for (std::size_t j = 0; j < original_weights.size(); ++j)
    {
        const auto expected = static_cast<double>(
            static_cast<long double>(original_weights[j]) * factor);
        const double tolerance = 1e-6 * std::fabs(expected) + 1e-9;
        EXPECT_NEAR(convolution.weights[j], expected, tolerance) << "weight " << j;
    }

    EXPECT_TRUE(convolution.scales.empty());
    EXPECT_TRUE(convolution.rolling_mean.empty());
    EXPECT_TRUE(convolution.rolling_variance.empty());
}

/**
 * @brief Tests that the folded convolution and the unfolded one agree on one input on the host.
 *
 * The seed is fixed at 20261009; the tolerance is 1e-5 absolute on every output pixel.
 */
TEST(Unit_DarknetFold, FoldedAndUnfoldedConvolutionsAgreeOnTheHost)
{
    std::mt19937 engine(K_SEED);
    const std::vector<float> weights =
        uniform_values(engine, K_FILTERS * K_CHANNELS * K_SIDE * K_SIDE, -1.0f, 1.0f);
    const std::vector<float> input = uniform_values(engine, K_CHANNELS * K_HEIGHT * K_WIDTH,
                                                    -1.0f, 1.0f);

    DarknetConvolution unfolded;
    unfolded.biases = uniform_values(engine, K_FILTERS, -0.5f, 0.5f);
    unfolded.scales = uniform_values(engine, K_FILTERS, 0.5f, 1.5f);
    unfolded.rolling_mean = uniform_values(engine, K_FILTERS, -0.5f, 0.5f);
    unfolded.rolling_variance = uniform_values(engine, K_FILTERS, 0.25f, 2.0f);
    unfolded.weights = weights;

    DarknetConvolution folded = unfolded;
    fold_darknet_normalisation(folded);

    double largest = 0.0;
    for (std::size_t f = 0; f < K_FILTERS; ++f)
    {
        const double scale = unfolded.scales[f];
        const double mean = unfolded.rolling_mean[f];
        const double root = std::sqrt(static_cast<double>(unfolded.rolling_variance[f]) + 0.00001);
        for (std::size_t y = 0; y < K_HEIGHT; ++y)
        {
            for (std::size_t x = 0; x < K_WIDTH; ++x)
            {
                const double raw = convolve_at(weights, input, f, y, x);
                const double expected = scale * (raw - mean) / root + unfolded.biases[f];
                const double actual =
                    convolve_at(folded.weights, input, f, y, x) + folded.biases[f];
                largest = std::max(largest, std::fabs(expected - actual));
            }
        }
    }
    EXPECT_LE(largest, 1e-5) << "largest difference " << largest;
}

/**
 * @brief Tests that folding a convolution twice, which is a misuse, stops the process.
 */
TEST(Unit_DarknetFold, FoldingTwiceStopsTheProcess)
{
    DarknetConvolution convolution;
    convolution.biases = {0.5f};
    convolution.scales = {1.0f};
    convolution.rolling_mean = {0.0f};
    convolution.rolling_variance = {1.0f};
    convolution.weights = {1.0f, 2.0f};
    fold_darknet_normalisation(convolution);

    SA_EXPECT_FATAL(fold_darknet_normalisation(convolution), "not normalised");
}

/**
 * @brief Tests that a variance whose epsilon-shifted value is not positive stops the process.
 */
TEST(Unit_DarknetFold, NonPositiveVarianceStopsTheProcess)
{
    DarknetConvolution convolution;
    convolution.biases = {0.5f};
    convolution.scales = {1.0f};
    convolution.rolling_mean = {0.0f};
    convolution.rolling_variance = {-1.0f};
    convolution.weights = {1.0f, 2.0f};

    SA_EXPECT_FATAL(fold_darknet_normalisation(convolution), "positive variance");
}
