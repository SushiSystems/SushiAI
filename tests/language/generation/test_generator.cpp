/**************************************************************************/
/* test_generator.cpp                                                    */
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
 * @file test_generator.cpp
 * @brief Tests that a prompt is continued through the Generator interface alone, and that every
 *        generator of the module refuses a request that does not fit.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include "position_feed_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/language/generation/generator.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/in_memory_generator.hpp>
#include <SushiAI/language/generation/in_memory_window_logits.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/generation/window_generator.hpp>
#include <SushiAI/language/text/token.hpp>

using namespace SushiAI;
using Language::Generation::CachedGenerator;
using Language::Generation::Generator;
using Language::Generation::GreedySampler;
using Language::Generation::InMemoryGenerator;
using Language::Generation::InMemoryWindowLogits;
using Language::Generation::RandomGenerator;
using Language::Generation::WindowGenerator;
using Language::Text::TokenID;
namespace Support = Language::Generation::Support;

namespace
{
    /** @brief Holds the seed of every generator of these tests; a greedy sampler draws nothing. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Holds T, the window of the in-memory generator of the interface cases. */
    constexpr std::int64_t K_WINDOW = 4;

    /** @brief Holds V, the vocabulary of the in-memory generator of the interface cases. */
    constexpr std::int64_t K_VOCABULARY = 5;

    /** @brief Views ids as the span a generator reads. */
    SushiRuntime::span<const TokenID> span_of(const std::vector<TokenID>& ids)
    {
        return SushiRuntime::span<const TokenID>(ids.data(), ids.size());
    }

    /** @brief Builds a generator of the module on a host engine. */
    using GeneratorMaker = std::unique_ptr<Generator> (*)(SushiBLAS::Engine&);

    /** @brief Names a way to build a generator over a window and a vocabulary of the support. */
    struct GeneratorCase
    {
        GeneratorMaker make; /**< Builds the generator. */
    };

    /** @brief Names a request that does not fit and what the refusal names. */
    struct RefusalCase
    {
        std::vector<TokenID> prompt; /**< The prompt given to generate. */
        std::int64_t count;          /**< The count given to generate. */
        std::string_view first;      /**< A substring the message holds. */
        std::string_view second;     /**< Another substring the message holds. */
    };

    /** @brief Builds the generator that runs no model. */
    std::unique_ptr<Generator> in_memory_generator(SushiBLAS::Engine&)
    {
        return std::make_unique<InMemoryGenerator>(Support::K_STEP_WINDOW,
                                                   Support::K_STEP_VOCABULARY);
    }

    /** @brief Builds the window generator over the in-memory window logits. */
    std::unique_ptr<Generator> window_generator(SushiBLAS::Engine&)
    {
        return WindowGenerator::make(std::make_unique<InMemoryWindowLogits>(
                                         Support::K_STEP_WINDOW, Support::K_STEP_VOCABULARY))
            .value();
    }

    /** @brief Builds the cached generator over the in-memory table step. */
    std::unique_ptr<Generator> cached_generator(SushiBLAS::Engine& engine)
    {
        return CachedGenerator::make(engine,
                                     Support::table_step(Support::K_STEP_WINDOW,
                                                         Support::K_STEP_VOCABULARY),
                                     Support::K_STEP_OUTPUT, Support::table_step_leaves())
            .value();
    }

    /** @brief Returns the three generators of the module, each over an in-memory model. */
    std::vector<GeneratorCase> generator_cases()
    {
        return {{&in_memory_generator}, {&window_generator}, {&cached_generator}};
    }

    /** @brief Returns an empty prompt, an id outside the vocabulary and a count of zero. */
    std::vector<RefusalCase> refusal_cases()
    {
        return {{{}, 1, "no id", ""}, {{1, 32}, 1, "id 1", "0 to 31"}, {{1}, 0, "0 tokens", ""}};
    }

    /** @brief Provides a host engine for the generators that allocate feeds. */
    class Small_LanguageGenerationGenerator : public Support::HostFeedTest
    {

    };

    /** @brief Provides every generator of the module against every request that does not fit. */
    class Small_GeneratorContract
        : public Support::HostFeedTest,
          public ::testing::WithParamInterface<std::tuple<GeneratorCase, RefusalCase>>
    {

    };
} // namespace

/** @brief Tests that a caller continues a prompt through the base: 1, 3 gives 4, 0, 1 after the 9. */
TEST_F(Small_LanguageGenerationGenerator, ACallerContinuesAPromptThroughTheInterface)
{
    InMemoryGenerator successor(K_WINDOW, K_VOCABULARY);
    Generator& generator = successor;
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);
    const std::vector<TokenID> prompt{1, 3};
    std::vector<TokenID> generated{9};

    SA_EXPECT_OK(generator.generate(span_of(prompt), 3, sampler, random, generated));

    EXPECT_EQ(generated, (std::vector<TokenID>{9, 4, 0, 1}));
}

/** @brief Tests that a refusal reaches the caller and leaves the tokens as they were. */
TEST_F(Small_LanguageGenerationGenerator, ARefusalLeavesTheGeneratedTokens)
{
    InMemoryGenerator successor(K_WINDOW, K_VOCABULARY);
    Generator& generator = successor;
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);
    std::vector<TokenID> generated{9};

    SA_EXPECT_FAILURE(generator.generate(span_of({}), 2, sampler, random, generated),
                      StatusCode::InvalidDescription, "no id");
    EXPECT_EQ(generated, (std::vector<TokenID>{9}));
}

/** @brief Tests that a generator owned through the base runs its own destructor. */
TEST_F(Small_LanguageGenerationGenerator, DestroyingThroughTheBaseRunsTheDerivedDestructor)
{
    auto made = std::make_unique<InMemoryGenerator>(K_WINDOW, K_VOCABULARY);
    const std::shared_ptr<const bool> destroyed = made->destruction_witness();
    std::unique_ptr<Generator> owned = std::move(made);

    owned.reset();

    EXPECT_TRUE(*destroyed);
}

/** @brief Tests that a prompt or a count that does not fit is refused and the tokens stay. */
TEST_P(Small_GeneratorContract, APromptOrACountThatDoesNotFitIsRefused)
{
    const GeneratorCase& generators = std::get<0>(GetParam());
    const RefusalCase& refusal = std::get<1>(GetParam());
    const std::unique_ptr<Generator> generator = generators.make(engine());
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);
    std::vector<TokenID> generated;

    SA_EXPECT_FAILURE(generator->generate(span_of(refusal.prompt), refusal.count, sampler,
                                          random, generated),
                      StatusCode::InvalidDescription, refusal.first, refusal.second);
    EXPECT_TRUE(generated.empty());
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GeneratorContract,
                         ::testing::Combine(::testing::ValuesIn(generator_cases()),
                                            ::testing::ValuesIn(refusal_cases())));
