/**************************************************************************/
/* test_sampler.cpp                                                       */
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
 * @file test_sampler.cpp
 * @brief Tests the random generator a sampler draws from and the Sampler interface.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/text/token.hpp>

using namespace SushiAI::Language;
using Generation::LogitSpan;
using Generation::RandomGenerator;
using Generation::Sampler;

namespace
{
    /** @brief Picks the token whose logit is largest and draws nothing. */
    class LargestSampler final : public Sampler
    {
        public:
            [[nodiscard]] std::string_view name() const noexcept override
            {
                return "largest";
            }

            [[nodiscard]] Text::TokenID sample(LogitSpan logits,
                                               RandomGenerator&) const override
            {
                std::size_t best = 0;
                for (std::size_t i = 1; i < logits.size(); ++i)
                {
                    if (logits[i] > logits[best])
                        best = i;
                }
                return static_cast<Text::TokenID>(best);
            }
    };

    /** @brief Picks the token at the index of one draw, modulo the vocabulary. */
    class DrawingSampler final : public Sampler
    {
        public:
            [[nodiscard]] std::string_view name() const noexcept override
            {
                return "drawing";
            }

            [[nodiscard]] Text::TokenID sample(LogitSpan logits,
                                               RandomGenerator& random) const override
            {
                return static_cast<Text::TokenID>(random() % logits.size());
            }
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

/** @brief Tests that a sampler that draws nothing leaves the generator as it was. */
TEST(Small_LanguageGenerationSampler, ASamplerThatDrawsNothingLeavesTheGenerator)
{
    const std::vector<float> logits{0.5f, 2.0f, -1.0f};
    const LargestSampler largest;
    const Sampler& sampler = largest;
    RandomGenerator random(42);
    const RandomGenerator before = random;

    EXPECT_EQ(sampler.sample(LogitSpan(logits.data(), logits.size()), random), 1);
    EXPECT_EQ(sampler.name(), "largest");
    EXPECT_EQ(random, before);
}

/** @brief Tests that two generators of one seed lead a drawing sampler to the same tokens. */
TEST(Small_LanguageGenerationSampler, OneSeedRepeatsTheTokensOfADrawingSampler)
{
    const std::vector<float> logits(7, 0.0f);
    const std::unique_ptr<Sampler> sampler = std::make_unique<DrawingSampler>();
    RandomGenerator first(1234);
    RandomGenerator second(1234);

    for (int draw = 0; draw < 8; ++draw)
    {
        EXPECT_EQ(sampler->sample(LogitSpan(logits.data(), logits.size()), first),
                  sampler->sample(LogitSpan(logits.data(), logits.size()), second))
            << draw;
    }
    EXPECT_EQ(first, second);
}
