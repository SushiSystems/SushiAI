/**************************************************************************/
/* test_in_memory_generator.cpp                                          */
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
 * @file test_in_memory_generator.cpp
 * @brief Tests that InMemoryGenerator continues a prompt with successors and reports its own
 *        destruction.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/in_memory_generator.hpp>
#include <SushiAI/language/text/token.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;
using Language::Text::TokenID;

namespace
{
    /** @brief Holds the seed of the generator of these tests; a greedy sampler draws nothing. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Holds T of the generator under test. */
    constexpr std::int64_t K_WINDOW = 4;

    /** @brief Holds V of the generator under test. */
    constexpr std::int64_t K_VOCABULARY = 5;
} // namespace

/** @brief Tests that a prompt ending in 3 is continued by 4, 0 and 1 over five tokens. */
TEST(Small_InMemoryGenerator, ThePromptIsContinuedWithSuccessorsModuloTheVocabulary)
{
    InMemoryGenerator generator(K_WINDOW, K_VOCABULARY);
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);
    const std::vector<TokenID> prompt{1, 3};
    std::vector<TokenID> generated;

    SA_EXPECT_OK(generator.generate(
        SushiRuntime::span<const TokenID>(prompt.data(), prompt.size()), 3, sampler, random,
        generated));

    EXPECT_EQ(generated, (std::vector<TokenID>{4, 0, 1}));
}

/** @brief Tests that the witness reads false while the generator lives and true after it. */
TEST(Small_InMemoryGenerator, TheWitnessReadsTrueOnceTheGeneratorIsDestroyed)
{
    auto generator = std::make_unique<InMemoryGenerator>(K_WINDOW, K_VOCABULARY);
    const std::shared_ptr<const bool> witness = generator->destruction_witness();
    const bool alive = *witness;

    generator.reset();

    EXPECT_FALSE(alive);
    EXPECT_TRUE(*witness);
}
