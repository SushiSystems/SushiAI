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
#include <support/tolerance.hpp>

namespace
{
    using SushiAI::Vision::Importers::DarknetConvolution;
    using SushiAI::Vision::Importers::fold_darknet_normalisation;

    /** @brief Holds the epsilon the batch-norm formula adds under the variance, an input value. */
    constexpr long double K_EPSILON = 0.00001L;

    /**
     * @brief Holds the absolute tolerance of the folded output against the unfolded one: 256
     *        times the float unit roundoff. The fold rounds each weight and bias to float once
     *        and a pixel sums 18 products of values below 1, so the bound is far above the
     *        rounding and far below any wrong fold.
     */
    constexpr double K_FOLD_TOLERANCE = 256.0 * TestSupport::unit_roundoff<float>();

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

    /** @brief Returns each value multiplied by @p factor in long double. */
    std::vector<long double> scaled_by(const std::vector<float>& values, long double factor)
    {
        std::vector<long double> scaled;
        for (const float value : values)
            scaled.push_back(static_cast<long double>(value) * factor);

        return scaled;
    }

    /**
     * @brief Expects each value to be its reference rounded once to float.
     *
     * The fold computes each value in double and stores it as float, so one rounding of the
     * float type is the whole error: TestSupport::measure_relative_error with one operation.
     */
    ::testing::AssertionResult each_rounded_once(const std::vector<float>& computed,
                                                 const std::vector<long double>& reference)
    {
        if (computed.size() != reference.size())
            return ::testing::AssertionFailure()
                   << computed.size() << " values for " << reference.size() << " references";

        for (std::size_t i = 0; i < computed.size(); ++i)
        {
            const TestSupport::ErrorRatio measured = TestSupport::measure_relative_error<float>(
                computed[i], static_cast<double>(reference[i]), 1);
            if (!measured.is_within())
                return ::testing::AssertionFailure()
                       << "value " << i << ": " << measured.describe();
        }

        return ::testing::AssertionSuccess();
    }

    /**
     * @brief Returns the largest difference between the unfolded convolution followed by the
     *        normalisation and the folded convolution, over every filter and output pixel.
     */
    double largest_fold_difference(const DarknetConvolution& unfolded,
                                   const DarknetConvolution& folded,
                                   const std::vector<float>& weights,
                                   const std::vector<float>& input)
    {
        double largest = 0.0;
        for (std::size_t f = 0; f < K_FILTERS; ++f)
        {
            const double scale = unfolded.scales[f];
            const double mean = unfolded.rolling_mean[f];
            const double root = std::sqrt(static_cast<double>(unfolded.rolling_variance[f]) +
                                          static_cast<double>(K_EPSILON));
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

        return largest;
    }
} // namespace

/**
 * @brief Tests that a one-filter fold equals the formula evaluated in long double.
 *
 * Each value is the formula's value rounded once to float; the oracle is in long double.
 */
TEST(Small_DarknetFold, OneFilterFoldMatchesTheFormulaInLongDouble)
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

    EXPECT_TRUE(each_rounded_once(convolution.weights, scaled_by(original_weights, factor)));
    EXPECT_TRUE(each_rounded_once(convolution.biases, {expected_bias}));
    EXPECT_TRUE(convolution.scales.empty());
    EXPECT_TRUE(convolution.rolling_mean.empty());
    EXPECT_TRUE(convolution.rolling_variance.empty());
}

/**
 * @brief Tests that the folded convolution and the unfolded one agree on one input on the host.
 *
 * The seed is fixed at 20261009; the tolerance is K_FOLD_TOLERANCE on every output pixel.
 */
TEST(Small_DarknetFold, FoldedAndUnfoldedConvolutionsAgreeOnTheHost)
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

    const double largest = largest_fold_difference(unfolded, folded, weights, input);
    EXPECT_LE(largest, K_FOLD_TOLERANCE) << "largest difference " << largest;
}

/**
 * @brief Tests that folding a convolution twice, which is a misuse, stops the process.
 *
 * The case is Medium_: a death test runs a child process.
 */
TEST(Medium_DarknetFoldFatal, FoldingTwiceStopsTheProcess)
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
TEST(Medium_DarknetFoldFatal, NonPositiveVarianceStopsTheProcess)
{
    DarknetConvolution convolution;
    convolution.biases = {0.5f};
    convolution.scales = {1.0f};
    convolution.rolling_mean = {0.0f};
    convolution.rolling_variance = {-1.0f};
    convolution.weights = {1.0f, 2.0f};

    SA_EXPECT_FATAL(fold_darknet_normalisation(convolution), "positive variance");
}
