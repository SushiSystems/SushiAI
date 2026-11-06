/**************************************************************************/
/* test_host_noise.cpp                                                    */
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
 * @file test_host_noise.cpp
 * @brief Tests that HostNoise is a reproducible stream bounded by its scale.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>

namespace
{
    using SushiAI::Test::HostNoise;

    /** @brief Holds how many values a case draws. */
    constexpr std::size_t K_DRAWS = 1000;

    /** @brief Returns the first @p count values of the stream seeded with @p seed. */
    std::vector<double> draw(std::uint64_t seed, std::size_t count, double scale)
    {
        HostNoise noise(seed);
        std::vector<double> values(count);
        for (double& value : values)
            value = noise.next(scale);
        return values;
    }
} // namespace

/** @brief Tests that one seed gives one stream and another seed another. */
TEST(Small_HostNoise, TheSeedAloneFixesTheStream)
{
    EXPECT_EQ(draw(42, K_DRAWS, 1.0), draw(42, K_DRAWS, 1.0));
    EXPECT_NE(draw(42, K_DRAWS, 1.0), draw(43, K_DRAWS, 1.0));
}

/** @brief Tests that every value lies in [-scale, scale) and both signs occur. */
TEST(Small_HostNoise, ValuesStayWithinTheScale)
{
    constexpr double K_SCALE = 0.25;
    const std::vector<double> values = draw(7, K_DRAWS, K_SCALE);
    const auto [lowest, highest] = std::minmax_element(values.begin(), values.end());
    const auto negatives = static_cast<std::size_t>(
        std::count_if(values.begin(), values.end(), [](double value) { return value < 0.0; }));

    EXPECT_GE(*lowest, -K_SCALE);
    EXPECT_LT(*highest, K_SCALE);
    EXPECT_GT(negatives, 0u);
    EXPECT_LT(negatives, K_DRAWS);
}

/** @brief Tests that fill writes the values next would return, in order. */
TEST(Small_HostNoise, FillWritesTheValuesNextReturns)
{
    std::vector<double> filled(16);
    HostNoise(11).fill(SushiRuntime::span<double>(filled.data(), filled.size()), 2.0);

    EXPECT_EQ(filled, draw(11, filled.size(), 2.0));
}
