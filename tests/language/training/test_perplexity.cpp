/**************************************************************************/
/* test_perplexity.cpp                                                    */
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
 * @file test_perplexity.cpp
 * @brief Tests the perplexity of a loss, its product rule, its metrics and a loss out of range.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/language/training/perplexity.hpp>
#include <SushiAI/train/metrics.hpp>
#include <support/tolerance.hpp>

using SushiAI::Language::Training::next_token_metrics;
using SushiAI::Language::Training::perplexity;
using SushiAI::Train::Metric;

namespace
{
    /**
     * @brief Counts the rounded operations between a loss of order one and its perplexity: the
     *        loss itself, the exponential and the rounding of the exponent, which scales with the
     *        loss, so six cover the losses these cases use (below ln 64).
     */
    constexpr std::size_t K_PERPLEXITY_OPERATIONS = 6;

    /** @brief Counts the rounded operations of a product of two perplexities and the sum before. */
    constexpr std::size_t K_PRODUCT_OPERATIONS = 6;

    /** @brief Provides the losses too large for a double's exponential. */
    class Small_LanguageTrainingPerplexityOverflow : public ::testing::TestWithParam<double>
    {

    };
} // namespace

/** @brief Tests that the perplexity of a loss of 0 is exactly 1. */
TEST(Small_LanguageTrainingPerplexity, ALossOfZeroHasAPerplexityOfOne)
{
    EXPECT_EQ(perplexity(0.0), 1.0);
}

/** @brief Tests that the perplexity of ln 64, a uniform guess over 64 tokens, is 64. */
TEST(Small_LanguageTrainingPerplexity, ThePerplexityIsTheExponentialOfTheLoss)
{
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        perplexity(std::log(64.0)), 64.0, K_PERPLEXITY_OPERATIONS)));
}

/** @brief Tests that the metrics are named loss and perplexity and read as bare numbers. */
TEST(Small_LanguageTrainingPerplexity, TheMetricsAreTheLossAndThePerplexity)
{
    const std::array<Metric, 2> metrics = next_token_metrics(std::log(8.0));

    EXPECT_EQ(metrics[0].name, "loss");
    EXPECT_EQ(metrics[0].value, std::log(8.0));
    EXPECT_FALSE(metrics[0].is_fraction);
    EXPECT_EQ(metrics[1].name, "perplexity");
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        metrics[1].value, 8.0, K_PERPLEXITY_OPERATIONS)));
    EXPECT_FALSE(metrics[1].is_fraction);
}

/** @brief Tests that the perplexity of a sum of losses is the product of their perplexities. */
TEST(Small_LanguageTrainingPerplexity, ASumOfLossesMultipliesThePerplexities)
{
    const double first = std::log(2.0);
    const double second = std::log(3.0);

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        perplexity(first + second), perplexity(first) * perplexity(second),
        K_PRODUCT_OPERATIONS)));
}

/** @brief Tests that a larger loss has a larger perplexity. */
TEST(Small_LanguageTrainingPerplexity, ALargerLossHasALargerPerplexity)
{
    EXPECT_LT(perplexity(std::log(2.0)), perplexity(std::log(3.0)));
}

/** @brief Tests that a loss too large for a double keeps its value and an infinite perplexity. */
TEST_P(Small_LanguageTrainingPerplexityOverflow, AnOverflowingLossKeepsItsValue)
{
    const double loss = GetParam();

    const std::array<Metric, 2> metrics = next_token_metrics(loss);

    EXPECT_EQ(metrics[0].value, loss);
    EXPECT_EQ(metrics[1].value, std::numeric_limits<double>::infinity());
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LanguageTrainingPerplexityOverflow,
                         ::testing::Values(710.0, 1.0e6, 1.0e300));
