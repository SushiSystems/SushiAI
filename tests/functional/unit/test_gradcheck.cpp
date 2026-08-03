/**************************************************************************/
/* test_gradcheck.cpp                                                     */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free. The gradient checker is the instrument the integration tier
// measures with, so it is calibrated here against closed-form functions where
// the right answer is known - including a deliberately wrong gradient, to
// prove the check can actually fail.

#include <cmath>
#include <span>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/autograd/gradcheck.hpp>

using namespace SushiAI::Autograd;

namespace
{
    /** @brief f(x) = sum(x_i^3) - sum(x_i^2) / 2, a smooth non-quadratic. */
    double cubic(std::span<const double> x)
    {
        double total = 0.0;
        for (double v : x) total += v * v * v - 0.5 * v * v;
        return total;
    }

    void cubic_gradient(std::span<const double> x, std::span<double> g)
    {
        for (std::size_t i = 0; i < x.size(); ++i) g[i] = 3.0 * x[i] * x[i] - x[i];
    }

    /** @brief The same gradient with one coordinate 0.1% out. */
    void slightly_wrong_gradient(std::span<const double> x, std::span<double> g)
    {
        cubic_gradient(x, g);
        g[2] *= 1.001;
    }

    /** @brief f(x) = sum(relu(x_i)^2), differentiable away from zero. */
    double rectified(std::span<const double> x)
    {
        double total = 0.0;
        for (double v : x) if (v > 0.0) total += v * v;
        return total;
    }

    void rectified_gradient(std::span<const double> x, std::span<double> g)
    {
        for (std::size_t i = 0; i < x.size(); ++i) g[i] = x[i] > 0.0 ? 2.0 * x[i] : 0.0;
    }
}

TEST(Unit_GradCheck, AcceptsAnExactGradient)
{
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45, 1.9, 0.6};
    const GradCheckResult result = check_full_jacobian(cubic, cubic_gradient, x);

    EXPECT_TRUE(result.passed) << result.to_string();
    EXPECT_LT(result.max_relative_error, 1e-7) << result.to_string();
    EXPECT_EQ(result.probes, x.size());
}

TEST(Unit_GradCheck, RejectsAGradientThatIsWrongByATenthOfAPercent)
{
    // The point of insisting on FLOAT64: a 1e-3 relative error is far above
    // the 1e-9-ish noise floor here, but it would sit inside the ~2.4e-5 floor
    // an fp32 difference quotient bottoms out at, and would pass unnoticed.
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45, 1.9, 0.6};
    const GradCheckResult result = check_full_jacobian(cubic, slightly_wrong_gradient, x);

    EXPECT_FALSE(result.passed) << result.to_string();
    EXPECT_EQ(result.worst_index, 2u);
    EXPECT_GT(result.max_relative_error, 1e-4);
}

TEST(Unit_GradCheck, LeavesTheProbePointUnchanged)
{
    std::vector<double> x = {0.7, -1.3, 2.1};
    const std::vector<double> before = x;

    (void)check_full_jacobian(cubic, cubic_gradient, x);
    EXPECT_EQ(x, before);

    (void)check_directional(cubic, cubic_gradient, x, 4);
    EXPECT_EQ(x, before);
}

TEST(Unit_GradCheck, DirectionalModeAgreesWithTheFullJacobian)
{
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45, 1.9, 0.6, 3.3, -2.7};

    const GradCheckResult full = check_full_jacobian(cubic, cubic_gradient, x);
    const GradCheckResult directional = check_directional(cubic, cubic_gradient, x, 12);

    EXPECT_TRUE(full.passed) << full.to_string();
    EXPECT_TRUE(directional.passed) << directional.to_string();
    EXPECT_EQ(directional.probes, 12u);
}

TEST(Unit_GradCheck, DirectionalModeCatchesAWrongGradientToo)
{
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45, 1.9, 0.6, 3.3, -2.7};
    const GradCheckResult result = check_directional(cubic, slightly_wrong_gradient, x, 16);
    EXPECT_FALSE(result.passed) << result.to_string();
}

TEST(Unit_GradCheck, DirectionalModeIsReproducibleForAGivenSeed)
{
    std::vector<double> x = {0.7, -1.3, 2.1, -0.45};

    GradCheckOptions options;
    options.seed = 12345;

    const GradCheckResult first = check_directional(cubic, cubic_gradient, x, 5, options);
    const GradCheckResult second = check_directional(cubic, cubic_gradient, x, 5, options);

    EXPECT_EQ(first.max_relative_error, second.max_relative_error);
    EXPECT_EQ(first.worst_index, second.worst_index);
}

TEST(Unit_GradCheck, PushOffKinksMovesCoordinatesAwayFromZeroAndKeepsTheirSign)
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

TEST(Unit_GradCheck, AReluFamilyFunctionNeedsToBeSampledOffTheKink)
{
    // A step size large enough that the difference survives the cancellation
    // of summing it against the O(1) terms; 1e-4 keeps the pushed coordinates
    // at 1e-3, where a central difference of x^2 is still exact to 1e-9.
    GradCheckOptions options;
    options.min_step = 1e-4;

    const std::vector<double> at_the_kink = {0.0, 1e-14, -1e-14, 0.8, -1.4, 2.2};

    // Straddling the kink, the quotient reports the average of the two
    // one-sided derivatives, which no correct rule can match.
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

TEST(Unit_GradCheck, AnEmptyProbePointIsRejected)
{
    std::vector<double> empty;
    EXPECT_THROW((void)check_full_jacobian(cubic, cubic_gradient, empty), SushiAI::Error);

    std::vector<double> x = {1.0};
    EXPECT_THROW((void)check_directional(cubic, cubic_gradient, x, 0), SushiAI::Error);
}
