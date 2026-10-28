/**************************************************************************/
/* test_distribution_expectation.cpp                                      */
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

/**
 * @file test_distribution_expectation.cpp
 * @brief Tests the expected distances of bin logits against hand-worked distributions.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/free/distribution_expectation.hpp>

namespace
{
    using SushiAI::Vision::Detect::Free::expect_distances;
    using SushiAI::Vision::Detect::Free::fill_distribution_projection;

    /** @brief Holds the bins per side of the hand cases. */
    constexpr std::int64_t K_BINS = 4;

    /** @brief Holds the tolerance of an expectation below 4: eight float32 steps at 4. */
    constexpr float K_EXPECTATION_TOLERANCE = 4e-6f;

    /** @brief Returns the logits of one anchor whose four sides hold the hand distributions. */
    std::vector<float> hand_logits()
    {
        const auto log = [](double value) { return static_cast<float>(std::log(value)); };
        return {
            0.0f,   0.0f,   0.0f,   0.0f,   // left: uniform, 1.5
            log(1), log(2), log(3), log(4), // top: (0 + 2 + 6 + 12) / 10 = 2
            log(4), log(3), log(2), log(1), // right: (0 + 3 + 4 + 3) / 10 = 1
            5.0f,   5.0f,   5.0f,   5.0f,   // bottom: uniform after a shift, 1.5
        };
    }
} // namespace

/** @brief Tests that each side's expectation is the hand-worked mean of its bin weights. */
TEST(Small_DetectDistributionExpectation, EachSideIsTheMeanOfItsBins)
{
    const std::vector<float> logits = hand_logits();
    std::vector<float> distances(4, -1.0f);

    SA_EXPECT_OK(expect_distances(logits, K_BINS, distances));

    EXPECT_NEAR(distances[0], 1.5f, K_EXPECTATION_TOLERANCE);
    EXPECT_NEAR(distances[1], 2.0f, K_EXPECTATION_TOLERANCE);
    EXPECT_NEAR(distances[2], 1.0f, K_EXPECTATION_TOLERANCE);
    EXPECT_NEAR(distances[3], 1.5f, K_EXPECTATION_TOLERANCE);
}

/** @brief Tests that one bin per side copies the logits unchanged. */
TEST(Small_DetectDistributionExpectation, OneBinCopiesTheLogits)
{
    const std::vector<float> logits = {-2.5f, 0.0f, 3.25f, 7.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> distances(8, 0.0f);

    SA_EXPECT_OK(expect_distances(logits, 1, distances));

    EXPECT_EQ(distances, logits);
}

/** @brief Tests that logits of another size are refused, naming the bins and both sizes. */
TEST(Small_DetectDistributionExpectation, LogitsOfAnotherSizeAreRefused)
{
    const std::vector<float> logits(15, 0.0f);
    std::vector<float> distances(4, 0.0f);

    SA_EXPECT_FAILURE(expect_distances(logits, K_BINS, distances),
                      SushiAI::StatusCode::InvalidDescription, "4 * 4", "15", "4 values");
}

/** @brief Tests that the projection holds 0, 1, ..., reg_max - 1. */
TEST(Small_DetectDistributionExpectation, TheProjectionCountsTheBins)
{
    std::vector<float> projection(4, -1.0f);

    fill_distribution_projection(projection);

    EXPECT_EQ(projection, (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f}));
}

/** @brief Tests that a non-finite logit is refused with its index. */
TEST(Small_DetectDistributionExpectation, RefusesANonFiniteLogitWithItsIndex)
{
    std::vector<float> logits = hand_logits();
    logits[6] = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> distances(4, -1.0f);

    SA_EXPECT_FAILURE(expect_distances(logits, K_BINS, distances),
                      SushiAI::StatusCode::InvalidDescription, "bin logit 6", "finite");
}
