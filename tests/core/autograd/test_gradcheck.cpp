/**************************************************************************/
/* test_gradcheck.cpp                                                     */
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
 * @file test_gradcheck.cpp
 * @brief Tests the gradient checker against closed-form functions and a wrong gradient.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI::Autograd;

namespace
{
    /**
     * @brief Bounds the relative error of an exact gradient under central differences in double; 2^30
     * unit roundoffs of double, the smallest power of two that covers the 1e-7 truncation of the
     * difference.
     */
    constexpr double K_EXACT_GRADIENT_ERROR = 1073741824.0 * TestSupport::unit_roundoff<double>();

    /**
     * @brief Bounds from below the relative error a gradient wrong by a tenth of a percent must
     * show; 2^40 unit roundoffs of double, the smallest power of two that reaches 1e-4, a detection
     * floor a decade under the 1e-3 error and far above the double noise.
     */
    constexpr double K_WRONG_GRADIENT_ERROR_FLOOR =
        1099511627776.0 * TestSupport::unit_roundoff<double>();

    /** @brief f(x) = sum(x_i^3) - sum(x_i^2) / 2, a smooth non-quadratic. */
    double cubic(SushiRuntime::span<const double> x)
    {
        double total = 0.0;
        for (double v : x) total += v * v * v - 0.5 * v * v;
        return total;
    }

    void cubic_gradient(SushiRuntime::span<const double> x, SushiRuntime::span<double> g)
    {
        for (std::size_t i = 0; i < x.size(); ++i) g[i] = 3.0 * x[i] * x[i] - x[i];
    }

    /** @brief The same gradient with one coordinate 0.1% out. */
    void slightly_wrong_gradient(SushiRuntime::span<const double> x, SushiRuntime::span<double> g)
    {
        cubic_gradient(x, g);
        g[2] *= 1.001;
    }

    /** @brief f(x) = sum(relu(x_i)^2), differentiable away from zero. */
    double rectified(SushiRuntime::span<const double> x)
    {
        double total = 0.0;
        for (double v : x) if (v > 0.0) total += v * v;
        return total;
    }

    void rectified_gradient(SushiRuntime::span<const double> x, SushiRuntime::span<double> g)
    {
        for (std::size_t i = 0; i < x.size(); ++i) g[i] = x[i] > 0.0 ? 2.0 * x[i] : 0.0;
    }

    /** @brief Expects @p v to sit at least the kink margin of a minimum step away from zero. */
    void expect_off_the_kink(double v, const GradCheckOptions& options)
    {
        EXPECT_GE(std::abs(v), options.kink_margin * options.min_step * 0.999);
    }
}

TEST(Small_GradCheck, AcceptsAnExactGradient)
{
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45, 1.9, 0.6};
    const GradCheckResult result = check_full_jacobian(cubic, cubic_gradient, x);

    EXPECT_TRUE(result.passed) << result.to_string();
    EXPECT_LT(result.max_relative_error, K_EXACT_GRADIENT_ERROR) << result.to_string();
    EXPECT_EQ(result.probes, x.size());
}

TEST(Small_GradCheck, RejectsAGradientThatIsWrongByATenthOfAPercent)
{
    // A 1e-3 relative error is above the FLOAT64 noise floor; see tests/core/autograd/README.md.
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45, 1.9, 0.6};
    const GradCheckResult result = check_full_jacobian(cubic, slightly_wrong_gradient, x);

    EXPECT_FALSE(result.passed) << result.to_string();
    EXPECT_EQ(result.worst_index, 2u);
    EXPECT_GT(result.max_relative_error, K_WRONG_GRADIENT_ERROR_FLOOR);
}

TEST(Small_GradCheck, TheFullJacobianLeavesTheProbePointUnchanged)
{
    std::vector<double> x = {0.7, -1.3, 2.1};
    const std::vector<double> before = x;

    (void)check_full_jacobian(cubic, cubic_gradient, x);

    EXPECT_EQ(x, before);
}

TEST(Small_GradCheck, TheDirectionalModeLeavesTheProbePointUnchanged)
{
    std::vector<double> x = {0.7, -1.3, 2.1};
    const std::vector<double> before = x;

    (void)check_directional(cubic, cubic_gradient, x, 4);

    EXPECT_EQ(x, before);
}

TEST(Small_GradCheck, DirectionalModePassesAnExactGradient)
{
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45, 1.9, 0.6, 3.3, -2.7};

    const GradCheckResult directional = check_directional(cubic, cubic_gradient, x, 12);

    EXPECT_TRUE(directional.passed) << directional.to_string();
}

TEST(Small_GradCheck, DirectionalModeReportsTheRequestedProbeCount)
{
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45, 1.9, 0.6, 3.3, -2.7};

    const GradCheckResult directional = check_directional(cubic, cubic_gradient, x, 12);

    EXPECT_EQ(directional.probes, 12u);
}

TEST(Small_GradCheck, DirectionalModeCatchesAWrongGradientToo)
{
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45, 1.9, 0.6, 3.3, -2.7};
    const GradCheckResult result = check_directional(cubic, slightly_wrong_gradient, x, 16);
    EXPECT_FALSE(result.passed) << result.to_string();
}

TEST(Small_GradCheck, DirectionalModeIsReproducibleForAGivenSeed)
{
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45};

    GradCheckOptions options;
    options.seed = 12345;

    const GradCheckResult first = check_directional(cubic, cubic_gradient, x, 5, options);
    const GradCheckResult second = check_directional(cubic, cubic_gradient, x, 5, options);

    EXPECT_EQ(first.max_relative_error, second.max_relative_error);
    EXPECT_EQ(first.worst_index, second.worst_index);
}

TEST(Small_GradCheck, PushOffKinksMovesKinkCoordinatesOffZeroKeepingTheirSign)
{
    GradCheckOptions options;
    std::vector<double> x = {0.0, 1e-12, -1e-12};

    push_off_kinks(x, options);

    EXPECT_GT(std::abs(x[0]), 0.0);
    EXPECT_GT(x[1], 0.0);
    EXPECT_LT(x[2], 0.0);
    expect_off_the_kink(x[0], options);
    expect_off_the_kink(x[1], options);
    expect_off_the_kink(x[2], options);
}

TEST(Small_GradCheck, PushOffKinksLeavesCoordinatesAwayFromZeroUnchanged)
{
    GradCheckOptions options;
    std::vector<double> x = {5.0, -5.0};

    push_off_kinks(x, options);

    EXPECT_DOUBLE_EQ(x[0], 5.0);
    EXPECT_DOUBLE_EQ(x[1], -5.0);
}

TEST(Small_GradCheck, AQuotientAcrossTheKinkOfAReLUFamilyFunctionIsRejected)
{
    // A quotient across the kink averages both derivatives; see tests/core/autograd/README.md.
    GradCheckOptions options;
    options.min_step = 1e-4;

    std::vector<double> unshifted = {0.0, 1e-14, -1e-14, 0.8, -1.4, 2.2};
    const GradCheckResult straddling =
        check_full_jacobian(rectified, rectified_gradient, unshifted, options);

    EXPECT_FALSE(straddling.passed) << straddling.to_string();
}

TEST(Small_GradCheck, AReLUFamilyFunctionPassesOnceSampledOffTheKink)
{
    // The step keeps the pushed coordinates at 1e-3; see tests/core/autograd/README.md.
    GradCheckOptions options;
    options.min_step = 1e-4;

    std::vector<double> shifted = {0.0, 1e-14, -1e-14, 0.8, -1.4, 2.2};
    push_off_kinks(shifted, options);
    const GradCheckResult clear =
        check_full_jacobian(rectified, rectified_gradient, shifted, options);

    EXPECT_TRUE(clear.passed) << clear.to_string();
}

TEST(Medium_GradCheckFatal, AnEmptyProbePointIsFatal)
{
    std::vector<double> empty;

    SA_EXPECT_FATAL((void)check_full_jacobian(cubic, cubic_gradient, empty), "at least one input");
}

TEST(Medium_GradCheckFatal, ZeroDirectionsIsFatal)
{
    std::vector<double> x = {1.0};

    SA_EXPECT_FATAL((void)check_directional(cubic, cubic_gradient, x, 0), "at least one direction");
}
