/**************************************************************************/
/* test_generation_task.cpp                                               */
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
 * @file test_generation_task.cpp
 * @brief Tests the reader of the "next_token_generation" task section.
 * @author Mustafa Garip
 */

#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/language/generation/generation_task.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/top_k_sampler.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;

namespace
{
    /** @brief Names the section in every diagnostic of these tests. */
    constexpr std::string_view K_CONTEXT = "the task section";

    /** @brief Reads the task of a section written as JSON. */
    Result<GenerationTask> task_of(std::string_view text)
    {
        return read_generation_task(Config::parse_json(text, "<test>"), K_CONTEXT);
    }
} // namespace

/** @brief Tests that a section with every key is read as it is stated. */
TEST(Unit_GenerationTask, EveryKeyIsRead)
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

/** @brief Tests that the optional keys have their defaults: 64 tokens, seed 0, greedy. */
TEST(Unit_GenerationTask, TheOptionalKeysHaveDefaults)
{
    SA_ASSERT_OK_AND_ASSIGN(const GenerationTask task, task_of(R"({
        "type": "next_token_generation", "vocabulary": "vocab.savb", "window": 8
    })"));

    EXPECT_EQ(task.tokens, K_DEFAULT_GENERATED_TOKENS);
    EXPECT_EQ(task.seed, 0u);
    ASSERT_NE(task.sampler, nullptr);
    EXPECT_EQ(task.sampler->name(), K_GREEDY_SAMPLER);
}

/** @brief Tests that an absent, an unknown and an out-of-range key are each refused. */
TEST(Unit_GenerationTask, AKeyThatDoesNotFitIsRefused)
{
    SA_EXPECT_FAILURE(task_of(R"({"type": "anchor_detection", "vocabulary": "v", "window": 8})"),
                      StatusCode::InvalidDescription, "'anchor_detection'",
                      "'next_token_generation'");
    SA_EXPECT_FAILURE(task_of(R"({"type": "next_token_generation", "window": 8})"),
                      StatusCode::InvalidDescription, "the task section",
                      "has no 'vocabulary'");
    SA_EXPECT_FAILURE(task_of(R"({"type": "next_token_generation", "vocabulary": "v"})"),
                      StatusCode::InvalidDescription, "has no 'window'");
    SA_EXPECT_FAILURE(
        task_of(R"({"type": "next_token_generation", "vocabulary": "v", "window": 0})"),
        StatusCode::InvalidDescription, "'window'", "positive");
    SA_EXPECT_FAILURE(task_of(R"({"type": "next_token_generation", "vocabulary": "v",
                                  "window": 8, "tokens": 0})"),
                      StatusCode::InvalidDescription, "'tokens'", "positive");
    SA_EXPECT_FAILURE(task_of(R"({"type": "next_token_generation", "vocabulary": "v",
                                  "window": 8, "seed": -1})"),
                      StatusCode::InvalidDescription, "'seed'");
    SA_EXPECT_FAILURE(task_of(R"({"type": "next_token_generation", "vocabulary": "v",
                                  "window": 8, "prompt": "hi"})"),
                      StatusCode::InvalidDescription, "the task section", "'prompt'", "does not have");
    SA_EXPECT_FAILURE(task_of(R"({"type": "next_token_generation", "vocabulary": "v",
                                  "window": 8, "sampler": {"type": "beam"}})"),
                      StatusCode::InvalidDescription, "the task section 'sampler'", "'beam'");
}

/** @brief Tests that "cache" is read as stated and is true when the section leaves it out. */
TEST(Unit_GenerationTask, TheCacheIsReadAndDefaultsToTrue)
{
    SA_ASSERT_OK_AND_ASSIGN(const GenerationTask absent, task_of(R"({
        "type": "next_token_generation", "vocabulary": "vocab.savb", "window": 8
    })"));
    SA_ASSERT_OK_AND_ASSIGN(const GenerationTask off, task_of(R"({
        "type": "next_token_generation", "vocabulary": "vocab.savb", "window": 8,
        "cache": false
    })"));
    SA_ASSERT_OK_AND_ASSIGN(const GenerationTask on, task_of(R"({
        "type": "next_token_generation", "vocabulary": "vocab.savb", "window": 8,
        "cache": true
    })"));

    EXPECT_TRUE(K_DEFAULT_GENERATION_CACHE);
    EXPECT_TRUE(absent.cache);
    EXPECT_FALSE(off.cache);
    EXPECT_TRUE(on.cache);
}

/** @brief Tests that a "cache" that is neither true nor false is refused by its key. */
TEST(Unit_GenerationTask, ACacheThatIsNoFlagIsRefused)
{
    SA_EXPECT_FAILURE(task_of(R"({"type": "next_token_generation", "vocabulary": "v",
                                  "window": 8, "cache": 1})"),
                      StatusCode::InvalidDescription, "the task section", "'cache'",
                      "a boolean");
}
