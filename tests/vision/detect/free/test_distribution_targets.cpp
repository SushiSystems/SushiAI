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

/**
 * @file test_distribution_targets.cpp
 * @brief Tests the bin weights of the distribution focal loss against hand-worked targets.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/vision/detect/free/distribution_targets.hpp>
#include <support/tolerance.hpp>

namespace
{
    using SushiAI::Test::near_reference;
    using SushiAI::Vision::Detect::Free::fill_distribution_weights;

    /** @brief Holds the bins per side of the hand cases. */
    constexpr std::int64_t K_BINS = 4;

    /**
     * @brief Holds the absolute the tolerance of a weight below 1: eight float32 steps at 1: 32
     *        times the float unit roundoff, the smallest power of two not tighter than the earlier
     *        1e-6f.
     */
    constexpr float K_WEIGHT_TOLERANCE =
        static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());

    /** @brief Holds a foreground anchor's normalized weight; a quarter of it is 0.2. */
    constexpr float K_FOREGROUND_WEIGHT = 0.8f;

    /** @brief Returns the weights of one side's bins written once for each of the four sides. */
    std::vector<float> on_four_sides(const std::vector<float>& bin_weights)
    {
        std::vector<float> weights;
        for (std::size_t side = 0; side < 4; ++side)
            weights.insert(weights.end(), bin_weights.begin(), bin_weights.end());

        return weights;
    }
} // namespace

/** @brief Names one target and the weights of the four bins around it, a quarter of 0.8 each. */
struct TargetRow
{
    const char* name;               /**< Names the row. */
    float target;                   /**< The distance of every side. */
    std::vector<float> bin_weights; /**< The weights of the four bins of one side. */
};

/** @brief Runs a row on a foreground anchor whose four sides all hold the row's target. */
class Small_DetectDistributionTargetsRows : public ::testing::TestWithParam<TargetRow>
{

};

/** @brief Tests the two bins around the target of every side of a foreground anchor. */
TEST_P(Small_DetectDistributionTargetsRows, EachSideWeighsTheTwoBinsAroundItsTarget)
{
    const std::vector<float> targets(4, GetParam().target);
    const std::vector<float> normalized = {K_FOREGROUND_WEIGHT};
    std::vector<float> weights(4 * K_BINS, -1.0f);

    SA_EXPECT_OK(fill_distribution_weights(targets, normalized, K_BINS, weights));

    EXPECT_TRUE(near_reference(weights, on_four_sides(GetParam().bin_weights),
                               K_WEIGHT_TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DetectDistributionTargetsRows,
    ::testing::Values(TargetRow{"BetweenBins1And2", 1.25f,
                                {0.0f, 0.75f * 0.2f, 0.25f * 0.2f, 0.0f}},
                      TargetRow{"OnBin0", 0.0f, {1.0f * 0.2f, 0.0f, 0.0f, 0.0f}},
                      TargetRow{"PastTheCeilingClampsTo2p99", 5.0f,
                                {0.0f, 0.0f, 0.01f * 0.2f, 0.99f * 0.2f}},
                      TargetRow{"BelowZeroClampsTo0", -1.0f, {1.0f * 0.2f, 0.0f, 0.0f, 0.0f}}),
    [](const ::testing::TestParamInfo<TargetRow>& info) { return std::string(info.param.name); });

/** @brief Tests that a background anchor, whose normalized weight is 0, weighs no bin. */
TEST(Small_DetectDistributionTargets, ABackgroundAnchorWeighsNoBin)
{
    const std::vector<float> targets(4, 1.5f);
    const std::vector<float> normalized = {0.0f};
    std::vector<float> weights(4 * K_BINS, -1.0f);

    SA_EXPECT_OK(fill_distribution_weights(targets, normalized, K_BINS, weights));

    EXPECT_EQ(weights, std::vector<float>(4 * K_BINS, 0.0f));
}

/** @brief Tests that one bin per side is refused, naming the bin count. */
TEST(Small_DetectDistributionTargets, OneBinIsRefused)
{
    const std::vector<float> targets(4, 0.0f);
    const std::vector<float> normalized(1, 1.0f);
    std::vector<float> weights(4, 0.0f);

    SA_EXPECT_FAILURE(fill_distribution_weights(targets, normalized, 1, weights),
                      SushiAI::StatusCode::InvalidDescription, "at least 2", "got 1");
}

/** @brief Tests that weights of another size are refused, naming every size. */
TEST(Small_DetectDistributionTargets, WeightsOfAnotherSizeAreRefused)
{
    const std::vector<float> targets(4, 0.0f);
    const std::vector<float> normalized(1, 1.0f);
    std::vector<float> weights(15, 0.0f);

    SA_EXPECT_FAILURE(fill_distribution_weights(targets, normalized, K_BINS, weights),
                      SushiAI::StatusCode::InvalidDescription, "4, 1", "15 values");
}

/** @brief Tests that a non-finite target distance is refused with its index. */
TEST(Small_DetectDistributionTargets, RefusesANonFiniteTargetWithItsIndex)
{
    std::vector<float> targets = {1.0f, 1.0f, 1.0f, 1.0f};
    targets[2] = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> normalized = {K_FOREGROUND_WEIGHT};
    std::vector<float> weights(4 * K_BINS, 0.0f);

    SA_EXPECT_FAILURE(fill_distribution_weights(targets, normalized, K_BINS, weights),
                      SushiAI::StatusCode::InvalidDescription, "target distance 2", "finite");
}

/** @brief Tests that a non-finite normalized weight is refused with its index. */
TEST(Small_DetectDistributionTargets, RefusesANonFiniteNormalizedWeightWithItsIndex)
{
    const std::vector<float> targets = {1.0f, 1.0f, 1.0f, 1.0f};
    const std::vector<float> normalized = {std::numeric_limits<float>::infinity()};
    std::vector<float> weights(4 * K_BINS, 0.0f);

    SA_EXPECT_FAILURE(fill_distribution_weights(targets, normalized, K_BINS, weights),
                      SushiAI::StatusCode::InvalidDescription, "normalized weight 0", "finite");
}
