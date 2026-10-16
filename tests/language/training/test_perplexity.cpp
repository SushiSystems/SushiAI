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
 * @file test_perplexity.cpp
 * @brief Tests the product rule of the perplexity and the metrics of a loss out of range.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <SushiAI/language/training/perplexity.hpp>
#include <SushiAI/train/metrics.hpp>

using SushiAI::Language::Training::next_token_metrics;
using SushiAI::Language::Training::perplexity;
using SushiAI::Train::Metric;

namespace
{
    /** @brief Bounds the relative error of a perplexity against its exact value. */
    constexpr double K_RELATIVE_TOLERANCE = 1.0e-12;
} // namespace

/** @brief Tests that the perplexity of a sum of losses is the product of their perplexities. */
TEST(Small_LanguageTrainingPerplexity, ASumOfLossesMultipliesThePerplexities)
{
    const double first = std::log(2.0);
    const double second = std::log(3.0);

    EXPECT_NEAR(perplexity(first + second), perplexity(first) * perplexity(second),
                6.0 * K_RELATIVE_TOLERANCE);
    EXPECT_LT(perplexity(first), perplexity(second));
}

/** @brief Tests that a loss too large for a double keeps its value and an infinite perplexity. */
TEST(Small_LanguageTrainingPerplexity, AnOverflowingLossKeepsItsValue)
{
    const std::array<Metric, 2> metrics = next_token_metrics(1.0e6);

    EXPECT_NEAR(metrics[0].value, 1.0e6, 0.0);
    EXPECT_TRUE(std::isinf(metrics[1].value));
    EXPECT_GT(metrics[1].value, 0.0);
}
