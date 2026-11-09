/**************************************************************************/
/* test_temperature_sampler.cpp                                          */
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
 * @file test_temperature_sampler.cpp
 * @brief Tests TemperatureSampler against hand-worked distributions and under a seed.
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
#include <SushiAI/language/generation/temperature_sampler.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;

namespace
{
    /** @brief Holds the seed of every generator of these tests. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Holds ln 2, so that a logit of it has twice the weight of a logit of 0. */
    const float K_LN_2 = std::log(2.0f);

    /** @brief Names a temperature, a row of logits and the softmax share of each token. */
    struct ShareCase
    {
        double temperature;         /**< The temperature given to make. */
        std::vector<float> logits;  /**< The scores of one position. */
        std::vector<double> shares; /**< softmax(logits / temperature), worked by hand. */
    };

    /**
     * @brief Returns the hand-worked rows: logits 0, ln 2, 0 at temperature 1 give 1/4, 1/2 and
     *        1/4; logits 0, ln 2 at temperature 1/2 have the weights 1 and 4, so 1/5 and 4/5.
     */
    std::vector<ShareCase> share_cases()
    {
        return {{1.0, {0.0f, K_LN_2, 0.0f}, {0.25, 0.5, 0.25}}, {0.5, {0.0f, K_LN_2}, {0.2, 0.8}}};
    }

    /** @brief Names a temperature given to make and what its refusal names. */
    struct RefusalCase
    {
        double temperature;      /**< The temperature given to make. */
        std::string_view first;  /**< A substring the message holds. */
        std::string_view second; /**< Another substring the message holds. */
    };

    /** @brief Returns the refusals: zero, a negative, an infinite and a missing number. */
    std::vector<RefusalCase> refusal_cases()
    {
        return {{0.0, "'temperature'", "positive"},
                {-1.0, "positive", ""},
                {std::numeric_limits<double>::infinity(), "finite", ""},
                {std::numeric_limits<double>::quiet_NaN(), "finite", ""}};
    }

    /** @brief Provides the rows of the shares. */
    class Small_GenerationTemperatureShares : public ::testing::TestWithParam<ShareCase>
    {

    };

    /** @brief Provides the rows of the refusals. */
    class Small_GenerationTemperatureRefusal : public ::testing::TestWithParam<RefusalCase>
    {

    };
} // namespace

/** @brief Tests that a token is drawn with the probability softmax(logits / temperature) gives it. */
TEST_P(Small_GenerationTemperatureShares, TheSharesAreTheSoftmax)
{
    const ShareCase& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TemperatureSampler> sampler,
                            TemperatureSampler::make(row.temperature));

    const std::vector<int> counts = Support::count_draws(*sampler, row.logits, K_SEED);

    Support::expect_shares(counts, row.shares);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationTemperatureShares,
                         ::testing::ValuesIn(share_cases()));

/** @brief Tests that one seed gives one sequence. */
TEST(Small_GenerationTemperatureSampler, ASeedFixesTheSequence)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TemperatureSampler> sampler,
                            TemperatureSampler::make(1.0));
    const std::vector<float> logits{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};

    EXPECT_EQ(Support::draw_sequence(*sampler, logits, K_SEED, 64),
              Support::draw_sequence(*sampler, logits, K_SEED, 64));
}

/** @brief Tests that a temperature that is not positive and finite is refused. */
TEST_P(Small_GenerationTemperatureRefusal, MakeRefusesATemperatureThatIsNotPositiveAndFinite)
{
    const RefusalCase& row = GetParam();

    SA_EXPECT_FAILURE(TemperatureSampler::make(row.temperature), StatusCode::InvalidDescription,
                      row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationTemperatureRefusal,
                         ::testing::ValuesIn(refusal_cases()));
