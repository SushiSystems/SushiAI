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
 * @file test_gradcheck.cpp
 * @brief Tests the gradient checker against closed-form functions and a wrong gradient.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/gradcheck.hpp>

using namespace SushiAI::Autograd;

namespace
{
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
}

TEST(Small_GradCheck, AcceptsAnExactGradient)
{
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45, 1.9, 0.6};
    const GradCheckResult result = check_full_jacobian(cubic, cubic_gradient, x);

    EXPECT_TRUE(result.passed) << result.to_string();
    EXPECT_LT(result.max_relative_error, 1e-7) << result.to_string();
    EXPECT_EQ(result.probes, x.size());
}

TEST(Small_GradCheck, RejectsAGradientThatIsWrongByATenthOfAPercent)
{
    // A 1e-3 relative error is above the FLOAT64 noise floor; see tests/core/autograd/README.md.
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45, 1.9, 0.6};
    const GradCheckResult result = check_full_jacobian(cubic, slightly_wrong_gradient, x);

    EXPECT_FALSE(result.passed) << result.to_string();
    EXPECT_EQ(result.worst_index, 2u);
    EXPECT_GT(result.max_relative_error, 1e-4);
}

TEST(Small_GradCheck, LeavesTheProbePointUnchanged)
{
    std::vector<double> x = {0.7, -1.3, 2.1};
    const std::vector<double> before = x;

    (void)check_full_jacobian(cubic, cubic_gradient, x);
    EXPECT_EQ(x, before);

    (void)check_directional(cubic, cubic_gradient, x, 4);
    EXPECT_EQ(x, before);
}

TEST(Small_GradCheck, DirectionalModeAgreesWithTheFullJacobian)
{
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45, 1.9, 0.6, 3.3, -2.7};

    const GradCheckResult full = check_full_jacobian(cubic, cubic_gradient, x);
    const GradCheckResult directional = check_directional(cubic, cubic_gradient, x, 12);

    EXPECT_TRUE(full.passed) << full.to_string();
    EXPECT_TRUE(directional.passed) << directional.to_string();
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

TEST(Small_GradCheck, PushOffKinksMovesCoordinatesAwayFromZeroAndKeepsTheirSign)
{
    GradCheckOptions options;
    std::vector<double> x = {0.0, 1e-12, -1e-12, 5.0, -5.0};

    push_off_kinks(x, options);

    EXPECT_GT(std::abs(x[0]), 0.0);
    EXPECT_GT(x[1], 0.0);
    EXPECT_LT(x[2], 0.0);
    EXPECT_DOUBLE_EQ(x[3], 5.0);
    EXPECT_DOUBLE_EQ(x[4], -5.0);

    for (double v : x)
        EXPECT_GE(std::abs(v), options.kink_margin * options.min_step * 0.999);
}

TEST(Small_GradCheck, AReLUFamilyFunctionNeedsToBeSampledOffTheKink)
{
    // The step keeps the pushed coordinates at 1e-3; see tests/core/autograd/README.md.
    GradCheckOptions options;
    options.min_step = 1e-4;

    const std::vector<double> at_the_kink = {0.0, 1e-14, -1e-14, 0.8, -1.4, 2.2};

    // A quotient across the kink averages both derivatives; see tests/core/autograd/README.md.
    std::vector<double> unshifted = at_the_kink;
    const GradCheckResult straddling =
        check_full_jacobian(rectified, rectified_gradient, unshifted, options);
    EXPECT_FALSE(straddling.passed) << straddling.to_string();

    std::vector<double> shifted = at_the_kink;
    push_off_kinks(shifted, options);
    const GradCheckResult clear =
        check_full_jacobian(rectified, rectified_gradient, shifted, options);
    EXPECT_TRUE(clear.passed) << clear.to_string();
}

TEST(Small_GradCheck, AnEmptyProbePointIsRejected)
{
    std::vector<double> empty;
    EXPECT_THROW((void)check_full_jacobian(cubic, cubic_gradient, empty), SushiAI::Error);

    std::vector<double> x = {1.0};
    EXPECT_THROW((void)check_directional(cubic, cubic_gradient, x, 0), SushiAI::Error);
}
