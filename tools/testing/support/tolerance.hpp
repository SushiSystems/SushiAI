/**************************************************************************/
/* tolerance.hpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                               SushiSkills                              */
/*               https://github.com/SushiSystems/SushiSkills              */
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
 * @file tolerance.hpp
 * @brief Derives floating-point tolerances from rounding bounds, so no test types an epsilon.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>

#if __has_include(<gtest/gtest.h>)
#include <gtest/gtest.h>
#define TEST_SUPPORT_HAS_GTEST 1
#endif

/**
 * @brief Tolerance helpers shared by the tests of every repository that copies the tools.
 */
namespace TestSupport
{

/**
 * @brief The ratio above which a result fails, as the reference BLAS testers use it.
 */
inline constexpr double K_DEFAULT_THRESHOLD = 16.0;

/**
 * @brief Holds a measured error, its rounding bound and how far the error uses the bound.
 */
struct ErrorRatio
{
    /** @brief Absolute difference between the computed and the reference value. */
    double error = 0.0;
    /** @brief Rounding bound the error is measured against. */
    double bound = 0.0;
    /** @brief error divided by bound; infinite when the bound is zero and the error is not. */
    double ratio = 0.0;
    /** @brief The threshold the ratio is held to. */
    double threshold = K_DEFAULT_THRESHOLD;

    /**
     * @brief Returns whether the ratio is within the threshold and both values are numbers.
     */
    [[nodiscard]] bool is_within() const noexcept
    {
        return !std::isnan(ratio) && ratio <= threshold;
    }

    /**
     * @brief Returns one line naming the error, the bound, the ratio and the threshold.
     */
    [[nodiscard]] std::string describe() const
    {
        std::ostringstream text;
        text << "error " << error << ", bound " << bound << ", ratio " << ratio << " of " << threshold;
        return text.str();
    }
};

/**
 * @brief Returns the unit roundoff u of a floating-point type: half its machine epsilon.
 */
template <typename Real>
[[nodiscard]] constexpr double unit_roundoff() noexcept
{
    static_assert(std::numeric_limits<Real>::is_iec559, "unit_roundoff needs an IEEE 754 type");
    return static_cast<double>(std::numeric_limits<Real>::epsilon()) / 2.0;
}

/**
 * @brief Measures an error against a bound the caller derived.
 * @param computed The value the code under test produced.
 * @param reference The value the oracle gives.
 * @param bound The rounding bound; zero demands an exact result.
 * @param threshold The ratio above which the result fails.
 */
[[nodiscard]] inline ErrorRatio measure_error(double computed, double reference, double bound,
                                              double threshold = K_DEFAULT_THRESHOLD) noexcept
{
    ErrorRatio result;
    result.error = std::fabs(computed - reference);
    result.bound = bound;
    result.threshold = threshold;
    if (std::isnan(computed) != std::isnan(reference))
    {
        result.ratio = std::numeric_limits<double>::quiet_NaN();
    }
    else if (std::isnan(computed) || result.error == 0.0 || computed == reference)
    {
        result.error = 0.0;
        result.ratio = 0.0;
    }
    else if (bound == 0.0)
    {
        result.ratio = std::numeric_limits<double>::infinity();
    }
    else
    {
        result.ratio = result.error / bound;
    }
    return result;
}

/**
 * @brief Measures a sum of n products against n · u · Σ|aᵢbᵢ| (Higham, Accuracy and Stability, §3.1).
 * @param computed The value the code under test produced, in Real.
 * @param reference The value a wider-precision reference produced.
 * @param magnitude The sum of the absolute products, Σ|aᵢbᵢ|.
 * @param length The number of products n.
 * @param threshold The ratio above which the result fails.
 */
template <typename Real>
[[nodiscard]] ErrorRatio measure_sum_error(double computed, double reference, double magnitude, std::size_t length,
                                           double threshold = K_DEFAULT_THRESHOLD) noexcept
{
    double const bound = static_cast<double>(length) * unit_roundoff<Real>() * std::fabs(magnitude);
    return measure_error(computed, reference, bound, threshold);
}

/**
 * @brief Measures a value made by a fixed number of rounded operations against operations · u · |reference|.
 * @param computed The value the code under test produced, in Real.
 * @param reference The value a wider-precision reference produced.
 * @param operations The number of rounded operations on the longest path to the value.
 * @param threshold The ratio above which the result fails.
 */
template <typename Real>
[[nodiscard]] ErrorRatio measure_relative_error(double computed, double reference, std::size_t operations,
                                                double threshold = K_DEFAULT_THRESHOLD) noexcept
{
    double const bound = static_cast<double>(operations) * unit_roundoff<Real>() * std::fabs(reference);
    return measure_error(computed, reference, bound, threshold);
}

#ifdef TEST_SUPPORT_HAS_GTEST
/**
 * @brief Returns a GoogleTest result for a measured error, for EXPECT_TRUE and ASSERT_TRUE.
 */
[[nodiscard]] inline ::testing::AssertionResult is_within(ErrorRatio const& measured)
{
    if (measured.is_within())
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << measured.describe();
}
#endif

} // namespace TestSupport
