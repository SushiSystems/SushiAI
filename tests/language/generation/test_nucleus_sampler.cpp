/**************************************************************************/
/* test_nucleus_sampler.cpp                                              */
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
 * @file test_nucleus_sampler.cpp
 * @brief Tests NucleusSampler against hand-worked distributions and under a seed.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include "generation_test_support.hpp"
#include <SushiAI/failure/status.hpp>
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

    /** @brief Names a mass and the share each token of half_quarter_quarter() is drawn with. */
    struct MassCase
    {
        double mass;                /**< The mass the kept tokens must reach. */
        std::vector<double> shares; /**< The share of each token. */
    };

    /**
     * @brief Returns the hand-worked masses: 0.6 keeps tokens 0 and 1 (1/2 and 1/4, so 2/3 and
     *        1/3), 0.4 keeps token 0 alone, and 1 keeps all three.
     */
    std::vector<MassCase> mass_cases()
    {
        return {{0.6, {2.0 / 3.0, 1.0 / 3.0, 0.0}}, {0.4, {1.0, 0.0, 0.0}}, {1.0, {0.5, 0.25, 0.25}}};
    }

    /** @brief Names the arguments of a make call and what its refusal names. */
    struct RefusalCase
    {
        double mass;             /**< The mass given to make. */
        double temperature;      /**< The temperature given to make. */
        std::string_view first;  /**< A substring the message holds. */
        std::string_view second; /**< Another substring the message holds. */
    };

    /** @brief Returns the refusals: a mass outside (0, 1], a mass that is no number, a bad temperature. */
    std::vector<RefusalCase> refusal_cases()
    {
        return {{0.0, 1.0, "'nucleus'", "above 0"},
                {1.5, 1.0, "at most 1", ""},
                {std::numeric_limits<double>::quiet_NaN(), 1.0, "'nucleus'", ""},
                {0.9, -2.0, "temperature", ""}};
    }

    /** @brief Provides the rows of the masses. */
    class Small_GenerationNucleusMass : public ::testing::TestWithParam<MassCase>
    {

    };

    /** @brief Provides the rows of the refusals. */
    class Small_GenerationNucleusRefusal : public ::testing::TestWithParam<RefusalCase>
    {

    };
} // namespace

/** @brief Tests that the fewest tokens that reach the mass are drawn, in their shares. */
TEST_P(Small_GenerationNucleusMass, TheFewestTokensThatReachTheMassAreDrawn)
{
    const MassCase& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NucleusSampler> sampler,
                            NucleusSampler::make(row.mass, 1.0));

    const std::vector<int> counts =
        Support::count_draws(*sampler, half_quarter_quarter(), K_SEED);

    Support::expect_shares(counts, row.shares);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationNucleusMass, ::testing::ValuesIn(mass_cases()));

/** @brief Tests that one seed gives one sequence. */
TEST(Small_GenerationNucleusSampler, ASeedFixesTheSequence)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NucleusSampler> sampler,
                            NucleusSampler::make(0.9, 1.2));
    const std::vector<float> logits{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};

    EXPECT_EQ(Support::draw_sequence(*sampler, logits, K_SEED, 64),
              Support::draw_sequence(*sampler, logits, K_SEED, 64));
}

/** @brief Tests that a mass outside (0, 1] and a bad temperature are refused. */
TEST_P(Small_GenerationNucleusRefusal, MakeRefusesAMassOutsideItsRangeAndABadTemperature)
{
    const RefusalCase& row = GetParam();

    SA_EXPECT_FAILURE(NucleusSampler::make(row.mass, row.temperature),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationNucleusRefusal,
                         ::testing::ValuesIn(refusal_cases()));
