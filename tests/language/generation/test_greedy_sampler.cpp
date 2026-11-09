/**************************************************************************/
/* test_greedy_sampler.cpp                                               */
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
 * @file test_greedy_sampler.cpp
 * @brief Tests that GreedySampler picks the largest logit and draws nothing.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "generation_test_support.hpp"
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;

namespace
{
    /** @brief Holds the seed of every generator of these tests. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Names a row of logits and the token the greedy pick is. */
    struct PickCase
    {
        std::vector<float> logits; /**< The scores of one position. */
        int token;                 /**< The index the pick returns. */
    };

    /** @brief Returns the hand-worked rows: the largest, the first of equals, a lone logit. */
    std::vector<PickCase> pick_cases()
    {
        return {{{0.5f, 2.0f, -1.0f}, 1}, {{0.5f, 2.0f, 2.0f, -1.0f}, 1}, {{-3.0f}, 0}};
    }

    /** @brief Provides the rows of the pick. */
    class Small_GenerationGreedyPick : public ::testing::TestWithParam<PickCase>
    {

    };

    /** @brief Runs the process-stopping case in a child process. */
    class Medium_GenerationGreedySamplerDeathTest : public ::testing::Test
    {

    };

    /** @brief Asks the sampler for a token of no logit. */
    void sample_no_logit()
    {
        RandomGenerator random(K_SEED);
        static_cast<void>(GreedySampler().sample(LogitSpan(), random));
    }
} // namespace

/** @brief Tests that the largest logit wins, and the lowest index among equal ones. */
TEST_P(Small_GenerationGreedyPick, ThePickIsTheLargestLogitAndTheFirstOfEquals)
{
    const PickCase& row = GetParam();
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);

    EXPECT_EQ(sampler.sample(Support::span_of(row.logits), random), row.token);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationGreedyPick, ::testing::ValuesIn(pick_cases()));

/** @brief Tests that a pick leaves the generator as it was. */
TEST(Small_GenerationGreedySampler, APickDrawsNothing)
{
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);
    const RandomGenerator untouched(K_SEED);

    static_cast<void>(sampler.sample(Support::span_of({0.0f, 1.0f}), random));

    EXPECT_TRUE(random == untouched);
}

/** @brief Tests that a row of no logit stops the process. */
TEST_F(Medium_GenerationGreedySamplerDeathTest, NoLogitIsFatal)
{
    SA_EXPECT_FATAL(sample_no_logit(), "no logit");
}
