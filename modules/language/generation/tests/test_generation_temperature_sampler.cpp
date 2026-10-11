/**************************************************************************/
/* test_generation_temperature_sampler.cpp                                */
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
 * @file test_generation_temperature_sampler.cpp
 * @brief Tests TemperatureSampler against hand-worked distributions and under a seed.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/status.hpp>
#include <SushiAI/language/generation/temperature_sampler.hpp>
#include <common/test_status.hpp>
#include "generation_test_support.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::Generation;

namespace
{
    /** @brief Holds the seed of every generator of these tests. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Holds ln 2, so that a logit of it has twice the weight of a logit of 0. */
    const float K_LN_2 = std::log(2.0f);
} // namespace

/** @brief Tests logits 0, ln 2, 0 at temperature 1: the shares are 1/4, 1/2 and 1/4. */
TEST(Unit_GenerationTemperatureSampler, TheSharesAreTheSoftmax)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TemperatureSampler> sampler,
                            TemperatureSampler::make(1.0));

    const std::vector<int> counts = Support::count_draws(*sampler, {0.0f, K_LN_2, 0.0f}, K_SEED);

    Support::expect_share(counts[0], 0.25);
    Support::expect_share(counts[1], 0.5);
    Support::expect_share(counts[2], 0.25);
    EXPECT_EQ(sampler->name(), K_TEMPERATURE_SAMPLER);
}

/** @brief Tests logits 0, ln 2 at temperature 1/2: the weights are 1 and 4, so 1/5 and 4/5. */
TEST(Unit_GenerationTemperatureSampler, ALowerTemperatureSharpensTheShares)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TemperatureSampler> sampler,
                            TemperatureSampler::make(0.5));

    const std::vector<int> counts = Support::count_draws(*sampler, {0.0f, K_LN_2}, K_SEED);

    Support::expect_share(counts[0], 0.2);
    Support::expect_share(counts[1], 0.8);
}

/** @brief Tests that one seed gives one sequence and another seed gives another. */
TEST(Unit_GenerationTemperatureSampler, ASeedFixesTheSequence)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TemperatureSampler> sampler,
                            TemperatureSampler::make(1.0));
    const std::vector<float> logits{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};

    const auto first = Support::draw_sequence(*sampler, logits, K_SEED, 64);
    const auto again = Support::draw_sequence(*sampler, logits, K_SEED, 64);
    const auto other = Support::draw_sequence(*sampler, logits, K_SEED + 1, 64);

    EXPECT_EQ(first, again);
    EXPECT_NE(first, other);
}

/** @brief Tests that a temperature that is not positive and finite is refused. */
TEST(Unit_GenerationTemperatureSampler, MakeRefusesATemperatureThatIsNotPositiveAndFinite)
{
    SA_EXPECT_FAILURE(TemperatureSampler::make(0.0), StatusCode::InvalidDescription,
                      "'temperature'", "positive");
    SA_EXPECT_FAILURE(TemperatureSampler::make(-1.0), StatusCode::InvalidDescription,
                      "positive");
    SA_EXPECT_FAILURE(TemperatureSampler::make(std::numeric_limits<double>::infinity()),
                      StatusCode::InvalidDescription, "finite");
    SA_EXPECT_FAILURE(TemperatureSampler::make(std::numeric_limits<double>::quiet_NaN()),
                      StatusCode::InvalidDescription, "finite");
}
