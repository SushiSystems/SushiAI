/**************************************************************************/
/* test_nucleus_sampler.cpp                                               */
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
 * @file test_nucleus_sampler.cpp
 * @brief Tests NucleusSampler against hand-worked distributions and under a seed.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "generation_test_support.hpp"
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/nucleus_sampler.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;

namespace
{
    /** @brief Holds the seed of every generator of these tests. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Holds ln 2, so that a logit of it has twice the weight of a logit of 0. */
    const float K_LN_2 = std::log(2.0f);

    /** @brief Returns logits whose probabilities are 1/2, 1/4 and 1/4. */
    std::vector<float> half_quarter_quarter()
    {
        return {K_LN_2, 0.0f, 0.0f};
    }
} // namespace

/** @brief Tests a mass of 0.6 over 1/2, 1/4, 1/4: tokens 0 and 1 share 2/3 and 1/3. */
TEST(Unit_GenerationNucleusSampler, TheFewestTokensThatReachTheMassAreDrawn)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NucleusSampler> sampler,
                            NucleusSampler::make(0.6, 1.0));

    const std::vector<int> counts =
        Support::count_draws(*sampler, half_quarter_quarter(), K_SEED);

    Support::expect_share(counts[0], 2.0 / 3.0);
    Support::expect_share(counts[1], 1.0 / 3.0);
    EXPECT_EQ(counts[2], 0);
    EXPECT_EQ(sampler->name(), K_NUCLEUS_SAMPLER);
}

/** @brief Tests a mass of 0.4, which the first token reaches alone. */
TEST(Unit_GenerationNucleusSampler, AMassTheFirstTokenReachesKeepsItAlone)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NucleusSampler> sampler,
                            NucleusSampler::make(0.4, 1.0));

    const std::vector<int> counts =
        Support::count_draws(*sampler, half_quarter_quarter(), K_SEED);

    EXPECT_EQ(counts[0], Support::K_DRAWS);
}

/** @brief Tests a mass of 1, which keeps every token: the shares are 1/2, 1/4 and 1/4. */
TEST(Unit_GenerationNucleusSampler, AMassOfOneKeepsEveryToken)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NucleusSampler> sampler,
                            NucleusSampler::make(1.0, 1.0));

    const std::vector<int> counts =
        Support::count_draws(*sampler, half_quarter_quarter(), K_SEED);

    Support::expect_share(counts[0], 0.5);
    Support::expect_share(counts[1], 0.25);
    Support::expect_share(counts[2], 0.25);
}

/** @brief Tests that one seed gives one sequence. */
TEST(Unit_GenerationNucleusSampler, ASeedFixesTheSequence)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NucleusSampler> sampler,
                            NucleusSampler::make(0.9, 1.2));
    const std::vector<float> logits{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};

    EXPECT_EQ(Support::draw_sequence(*sampler, logits, K_SEED, 64),
              Support::draw_sequence(*sampler, logits, K_SEED, 64));
}

/** @brief Tests that a mass outside (0, 1] and a bad temperature are refused. */
TEST(Unit_GenerationNucleusSampler, MakeRefusesAMassOutsideItsRangeAndABadTemperature)
{
    SA_EXPECT_FAILURE(NucleusSampler::make(0.0, 1.0), StatusCode::InvalidDescription,
                      "'nucleus'", "above 0");
    SA_EXPECT_FAILURE(NucleusSampler::make(1.5, 1.0), StatusCode::InvalidDescription,
                      "at most 1");
    SA_EXPECT_FAILURE(NucleusSampler::make(std::numeric_limits<double>::quiet_NaN(), 1.0),
                      StatusCode::InvalidDescription, "'nucleus'");
    SA_EXPECT_FAILURE(NucleusSampler::make(0.9, -2.0), StatusCode::InvalidDescription,
                      "temperature");
}
