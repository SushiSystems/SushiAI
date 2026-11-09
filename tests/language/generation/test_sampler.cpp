/**************************************************************************/
/* test_sampler.cpp                                                      */
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
 * @file test_sampler.cpp
 * @brief Tests the random generator a sampler draws from and the Sampler contract over the
 *        four samplers of the module.
 * @author Mustafa Garip
 */

#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "generation_test_support.hpp"
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/nucleus_sampler.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/generation/temperature_sampler.hpp>
#include <SushiAI/language/generation/top_k_sampler.hpp>

using namespace SushiAI::Language;
using Generation::RandomGenerator;
using Generation::Sampler;

namespace
{
    /** @brief Builds one sampler of the module. */
    using SamplerMaker = std::unique_ptr<Sampler> (*)();

    /** @brief Names a sampler and the draws one pick of it takes from the generator. */
    struct SamplerCase
    {
        SamplerMaker make; /**< Builds the sampler. */
        int draws;         /**< How many values one pick takes from the generator. */
    };

    /** @brief Builds the greedy sampler, which draws nothing. */
    std::unique_ptr<Sampler> greedy()
    {
        return std::make_unique<Generation::GreedySampler>();
    }

    /** @brief Builds a temperature sampler of temperature 1. */
    std::unique_ptr<Sampler> temperature()
    {
        return Generation::TemperatureSampler::make(1.0).value();
    }

    /** @brief Builds a top-k sampler of k = 2. */
    std::unique_ptr<Sampler> top_k()
    {
        return Generation::TopKSampler::make(2, 1.0).value();
    }

    /** @brief Builds a nucleus sampler of mass 0.9. */
    std::unique_ptr<Sampler> nucleus()
    {
        return Generation::NucleusSampler::make(0.9, 1.0).value();
    }

    /** @brief Returns the four samplers: greedy takes no draw, each other takes one. */
    std::vector<SamplerCase> sampler_cases()
    {
        return {{&greedy, 0}, {&temperature, 1}, {&top_k, 1}, {&nucleus, 1}};
    }

    /** @brief Provides the samplers of the module, each behind its interface. */
    class Small_LanguageGenerationSamplerContract : public ::testing::TestWithParam<SamplerCase>
    {

    };
} // namespace

/** @brief Tests that the generator yields the output the C++ standard fixes for it. */
TEST(Small_LanguageGenerationSampler, TheGeneratorYieldsTheValueTheStandardFixes)
{
    RandomGenerator random;
    random.discard(9999);

    // [rand.predef] states the 10000th draw of a default-seeded std::mt19937_64.
    EXPECT_EQ(random(), 9981545732273789042ull);
}

/** @brief Tests that a pick takes at most one value from the generator: none for greedy. */
TEST_P(Small_LanguageGenerationSamplerContract, APickTakesAtMostOneDrawFromTheGenerator)
{
    const SamplerCase& row = GetParam();
    const std::unique_ptr<Sampler> owned = row.make();
    const Sampler& sampler = *owned;
    RandomGenerator random(42);
    RandomGenerator expected = random;
    expected.discard(static_cast<unsigned long long>(row.draws));

    static_cast<void>(sampler.sample(Generation::Support::span_of({0.5f, 2.0f, -1.0f}), random));

    EXPECT_EQ(random, expected);
}

/** @brief Tests that two generators of one seed lead a sampler to the same token. */
TEST_P(Small_LanguageGenerationSamplerContract, OneSeedGivesOneTokenAndOneGeneratorState)
{
    const SamplerCase& row = GetParam();
    const std::unique_ptr<Sampler> owned = row.make();
    const Sampler& sampler = *owned;
    RandomGenerator first(1234);
    RandomGenerator second(1234);
    const std::vector<float> logits{0.5f, 2.0f, -1.0f, 0.0f, 1.0f, 0.25f, 1.5f};

    const auto from_first = sampler.sample(Generation::Support::span_of(logits), first);
    const auto from_second = sampler.sample(Generation::Support::span_of(logits), second);

    EXPECT_EQ(from_first, from_second);
    EXPECT_EQ(first, second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LanguageGenerationSamplerContract,
                         ::testing::ValuesIn(sampler_cases()));
