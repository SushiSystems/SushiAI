/**************************************************************************/
/* test_distribution_targets.cpp                                          */
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
 * @file test_distribution_targets.cpp
 * @brief Tests the bin weights of the distribution focal loss against hand-worked targets.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/vision/detect/free/distribution_targets.hpp>

namespace
{
    using SushiAI::Vision::Detect::Free::fill_distribution_weights;

    /** @brief Holds the bins per side of the hand cases. */
    constexpr std::int64_t K_BINS = 4;

    /** @brief Holds the tolerance of a weight below 1: eight float32 steps at 1. */
    constexpr float K_WEIGHT_TOLERANCE = 1e-6f;

    /** @brief Holds a foreground anchor's normalized weight; a quarter of it is 0.2. */
    constexpr float K_FOREGROUND_WEIGHT = 0.8f;
} // namespace

/** @brief Tests the two bins around each target of a foreground anchor, and a background row. */
TEST(Small_DetectDistributionTargets, EachSideWeighsTheTwoBinsAroundItsTarget)
{
    // Sides: between bins 1 and 2, on bin 0, past the ceiling, below zero.
    const std::vector<float> targets = {1.25f, 0.0f, 5.0f, -1.0f, 1.5f, 1.5f, 1.5f, 1.5f};
    const std::vector<float> normalized = {K_FOREGROUND_WEIGHT, 0.0f};
    std::vector<float> weights(2 * 4 * K_BINS, -1.0f);

    fill_distribution_weights(targets, normalized, K_BINS, weights);

    const std::vector<float> expected = {
        0.0f, 0.75f * 0.2f, 0.25f * 0.2f, 0.0f,  // 1.25: bins 1 and 2
        1.0f * 0.2f, 0.0f, 0.0f, 0.0f,           // 0: bin 0 alone
        0.0f, 0.0f, 0.01f * 0.2f, 0.99f * 0.2f,  // 5 clamps to 2.99: bins 2 and 3
        1.0f * 0.2f, 0.0f, 0.0f, 0.0f,           // -1 clamps to 0
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };
    ASSERT_EQ(weights.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(weights[i], expected[i], K_WEIGHT_TOLERANCE) << "weight " << i;
}

/** @brief Tests that one bin per side throws, naming the bin count. */
TEST(Small_DetectDistributionTargets, OneBinThrows)
{
    const std::vector<float> targets(4, 0.0f);
    const std::vector<float> normalized(1, 1.0f);
    std::vector<float> weights(4, 0.0f);

    SA_EXPECT_ERROR(fill_distribution_weights(targets, normalized, 1, weights), "at least 2",
                    "got 1");
}

/** @brief Tests that weights of another size throw, naming every size. */
TEST(Small_DetectDistributionTargets, WeightsOfAnotherSizeThrow)
{
    const std::vector<float> targets(4, 0.0f);
    const std::vector<float> normalized(1, 1.0f);
    std::vector<float> weights(15, 0.0f);

    SA_EXPECT_ERROR(fill_distribution_weights(targets, normalized, K_BINS, weights), "4, 1",
                    "15 values");
}

/** @brief Tests that a non-finite target distance is refused with its index. */
TEST(Small_DetectDistributionTargets, RefusesANonFiniteTargetWithItsIndex)
{
    std::vector<float> targets = {1.0f, 1.0f, 1.0f, 1.0f};
    targets[2] = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> normalized = {K_FOREGROUND_WEIGHT};
    std::vector<float> weights(4 * K_BINS, 0.0f);

    SA_EXPECT_ERROR(fill_distribution_weights(targets, normalized, K_BINS, weights),
                    "target distance 2", "finite");
}
