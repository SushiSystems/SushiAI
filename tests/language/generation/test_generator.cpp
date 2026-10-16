/**************************************************************************/
/* test_generator.cpp                                                     */
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
 * @file test_generator.cpp
 * @brief Tests that a prompt is continued through the Generator interface alone.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/generator.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/text/token.hpp>

using namespace SushiAI;
using Language::Generation::Generator;
using Language::Generation::LogitSpan;
using Language::Generation::RandomGenerator;
using Language::Generation::Sampler;
using Language::Text::TokenID;

namespace
{
    /** @brief Holds T, the window of the fake generator. */
    constexpr std::int64_t K_WINDOW = 4;

    /** @brief Holds V, the vocabulary of the fake generator. */
    constexpr std::int64_t K_VOCABULARY = 5;

    /** @brief Picks the token whose logit is largest and draws nothing. */
    class LargestSampler final : public Sampler
    {
        public:
            [[nodiscard]] std::string_view name() const noexcept override
            {
                return "largest";
            }

            [[nodiscard]] TokenID sample(LogitSpan logits, RandomGenerator&) const override
            {
                std::size_t best = 0;
                for (std::size_t i = 1; i < logits.size(); ++i)
                {
                    if (logits[i] > logits[best])
                        best = i;
                }
                return static_cast<TokenID>(best);
            }
    };

    /** @brief Scores the token after the last one of the context highest, modulo V. */
    class SuccessorGenerator final : public Generator
    {
        public:
            /** @brief Builds the generator; @p destroyed is set when it is destroyed. */
            explicit SuccessorGenerator(bool& destroyed) : destroyed_(destroyed)
            {

            }

            /** @brief Marks the generator as destroyed. */
            ~SuccessorGenerator() override
            {
                destroyed_ = true;
            }

            SuccessorGenerator(const SuccessorGenerator&) = delete;
            SuccessorGenerator& operator=(const SuccessorGenerator&) = delete;
            SuccessorGenerator(SuccessorGenerator&&) = delete;
            SuccessorGenerator& operator=(SuccessorGenerator&&) = delete;

            [[nodiscard]] std::int64_t window() const noexcept override
            {
                return K_WINDOW;
            }

            [[nodiscard]] std::int64_t vocabulary() const noexcept override
            {
                return K_VOCABULARY;
            }

            /** @brief Appends @p count successors; refuses an empty prompt. */
            [[nodiscard]] Status generate(SushiRuntime::span<const TokenID> prompt,
                                          std::int64_t count, const Sampler& sampler,
                                          RandomGenerator& random,
                                          std::vector<TokenID>& generated) override
            {
                SA_FAIL_IF(prompt.empty(), StatusCode::InvalidDescription,
                           "a prompt needs at least one id");
                TokenID last = prompt[prompt.size() - 1];
                for (std::int64_t step = 0; step < count; ++step)
                {
                    std::vector<float> logits(static_cast<std::size_t>(K_VOCABULARY), 0.0f);
                    logits[static_cast<std::size_t>((last + 1) % K_VOCABULARY)] = 1.0f;
                    last = sampler.sample(LogitSpan(logits.data(), logits.size()), random);
                    generated.push_back(last);
                }
                return Status();
            }

        private:
            /** @brief Holds the flag the destructor sets. */
            bool& destroyed_;
    };
} // namespace

/** @brief Tests that a caller continues a prompt and reads the extents through the base. */
TEST(Small_LanguageGenerationGenerator, ACallerContinuesAPromptThroughTheInterface)
{
    bool destroyed = false;
    SuccessorGenerator successor(destroyed);
    Generator& generator = successor;
    const LargestSampler sampler;
    RandomGenerator random(7);
    const std::vector<TokenID> prompt{1, 3};
    std::vector<TokenID> generated{9};

    SA_EXPECT_OK(generator.generate(SushiRuntime::span<const TokenID>(prompt.data(),
                                                                      prompt.size()),
                                    3, sampler, random, generated));

    EXPECT_EQ(generated, (std::vector<TokenID>{9, 4, 0, 1}));
    EXPECT_EQ(generator.window(), K_WINDOW);
    EXPECT_EQ(generator.vocabulary(), K_VOCABULARY);
}

/** @brief Tests that a refusal reaches the caller and leaves the tokens as they were. */
TEST(Small_LanguageGenerationGenerator, ARefusalLeavesTheGeneratedTokens)
{
    bool destroyed = false;
    SuccessorGenerator successor(destroyed);
    Generator& generator = successor;
    const LargestSampler sampler;
    RandomGenerator random(7);
    std::vector<TokenID> generated{9};

    SA_EXPECT_FAILURE(generator.generate(SushiRuntime::span<const TokenID>(), 2, sampler, random,
                                         generated),
                      StatusCode::InvalidDescription, "at least one id");
    EXPECT_EQ(generated, (std::vector<TokenID>{9}));
}

/** @brief Tests that a generator owned through the base runs its own destructor. */
TEST(Small_LanguageGenerationGenerator, DestroyingThroughTheBaseRunsTheDerivedDestructor)
{
    bool destroyed = false;
    std::unique_ptr<Generator> owned = std::make_unique<SuccessorGenerator>(destroyed);

    owned.reset();

    EXPECT_TRUE(destroyed);
}
