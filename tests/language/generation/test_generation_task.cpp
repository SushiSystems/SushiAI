/**************************************************************************/
/* test_generation_task.cpp                                              */
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
 * @file test_generation_task.cpp
 * @brief Tests the reader of the "next_token_generation" task section.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/generation_task.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/top_k_sampler.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Names the section in every diagnostic of these tests. */
    constexpr std::string_view K_CONTEXT = "the task section";

    /** @brief Holds the seed of the generator of the default-sampler case. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Reads the task of a section written as JSON. */
    Result<GenerationTask> task_of(std::string_view text)
    {
        return read_generation_task(require_value(Config::parse_json(text, "<test>")), K_CONTEXT);
    }

    /** @brief Names a section and what the refusal of it names. */
    struct RefusalCase
    {
        std::string_view section; /**< The section, as JSON. */
        std::string_view first;   /**< A substring the message holds. */
        std::string_view second;  /**< Another substring the message holds. */
    };

    /** @brief Names a section and the cache flag it reads as. */
    struct CacheCase
    {
        std::string_view section; /**< The section, as JSON. */
        bool cache;               /**< The flag the reader returns. */
    };

    /** @brief Returns the eight sections that do not fit, each by the key that is wrong. */
    std::vector<RefusalCase> refusal_cases()
    {
        return {
            {R"({"type": "anchor_detection", "vocabulary": "v", "window": 8})",
             "'anchor_detection'", "'next_token_generation'"},
            {R"({"type": "next_token_generation", "window": 8})", "the task section",
             "has no 'vocabulary'"},
            {R"({"type": "next_token_generation", "vocabulary": "v"})", "has no 'window'", ""},
            {R"({"type": "next_token_generation", "vocabulary": "v", "window": 0})", "'window'",
             "positive"},
            {R"({"type": "next_token_generation", "vocabulary": "v", "window": 8, "tokens": 0})",
             "'tokens'", "positive"},
            {R"({"type": "next_token_generation", "vocabulary": "v", "window": 8, "seed": -1})",
             "'seed'", ""},
            {R"({"type": "next_token_generation", "vocabulary": "v", "window": 8,
                 "prompt": "hi"})",
             "'prompt'", "does not have"},
            {R"({"type": "next_token_generation", "vocabulary": "v", "window": 8,
                 "sampler": {"type": "beam"}})",
             "the task section 'sampler'", "'beam'"}};
    }

    /** @brief Returns a section without the key, one that turns the cache off and one that turns it on. */
    std::vector<CacheCase> cache_cases()
    {
        return {{R"({"type": "next_token_generation", "vocabulary": "v", "window": 8})", true},
                {R"({"type": "next_token_generation", "vocabulary": "v", "window": 8,
                     "cache": false})",
                 false},
                {R"({"type": "next_token_generation", "vocabulary": "v", "window": 8,
                     "cache": true})",
                 true}};
    }

    /** @brief Provides the sections that do not fit. */
    class Small_GenerationTaskRefusal : public ::testing::TestWithParam<RefusalCase>
    {

    };

    /** @brief Provides the sections that state the cache, or leave it out. */
    class Small_GenerationTaskCache : public ::testing::TestWithParam<CacheCase>
    {

    };
} // namespace

/** @brief Tests that a section with every key is read as it is stated. */
TEST(Small_GenerationTask, EveryKeyIsRead)
{
    SA_ASSERT_OK_AND_ASSIGN(const GenerationTask task, task_of(R"({
        "type": "next_token_generation", "vocabulary": "vocab.savb", "window": 256,
        "tokens": 200, "seed": 7, "sampler": {"type": "top_k", "k": 40}
    })"));

    EXPECT_EQ(task.vocabulary, "vocab.savb");
    EXPECT_EQ(task.window, 256);
    EXPECT_EQ(task.tokens, 200);
    EXPECT_EQ(task.seed, 7u);
    ASSERT_NE(task.sampler, nullptr);
    EXPECT_EQ(task.sampler->name(), K_TOP_K_SAMPLER);
}

/**
 * @brief Tests that the optional keys read as their stated defaults: 64 tokens, seed 0, and a
 *        sampler that picks the largest logit and draws nothing.
 */
TEST(Small_GenerationTask, TheOptionalKeysHaveDefaults)
{
    SA_ASSERT_OK_AND_ASSIGN(const GenerationTask task, task_of(R"({
        "type": "next_token_generation", "vocabulary": "vocab.savb", "window": 8
    })"));
    ASSERT_NE(task.sampler, nullptr);
    const std::vector<float> logits{0.1f, 0.9f, 0.5f};
    RandomGenerator random(K_SEED);
    const RandomGenerator untouched(K_SEED);

    const auto picked = task.sampler->sample(LogitSpan(logits.data(), logits.size()), random);

    EXPECT_EQ(task.tokens, 64);
    EXPECT_EQ(task.seed, 0u);
    EXPECT_EQ(picked, 1);
    EXPECT_EQ(random, untouched);
}

/** @brief Tests that an absent, an unknown and an out-of-range key are each refused. */
TEST_P(Small_GenerationTaskRefusal, AKeyThatDoesNotFitIsRefused)
{
    const RefusalCase& row = GetParam();

    SA_EXPECT_FAILURE(task_of(row.section), StatusCode::InvalidDescription, row.first,
                      row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationTaskRefusal,
                         ::testing::ValuesIn(refusal_cases()));

/** @brief Tests that "cache" is read as stated and is true when the section leaves it out. */
TEST_P(Small_GenerationTaskCache, TheCacheIsReadAndDefaultsToTrue)
{
    const CacheCase& row = GetParam();

    SA_ASSERT_OK_AND_ASSIGN(const GenerationTask task, task_of(row.section));

    EXPECT_EQ(task.cache, row.cache);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationTaskCache, ::testing::ValuesIn(cache_cases()));

/** @brief Tests that a "cache" that is neither true nor false is refused by its key. */
TEST(Small_GenerationTask, ACacheThatIsNoFlagIsRefused)
{
    SA_EXPECT_FAILURE(task_of(R"({"type": "next_token_generation", "vocabulary": "v",
                                  "window": 8, "cache": 1})"),
                      StatusCode::InvalidDescription, "the task section", "'cache'",
                      "a boolean");
}
