/**************************************************************************/
/* test_generation_greedy_sampler.cpp                                     */
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
 * @file test_generation_greedy_sampler.cpp
 * @brief Tests that GreedySampler picks the largest logit and draws nothing.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include "generation_test_support.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::Generation;

namespace
{
    /** @brief Holds the seed of every generator of these tests. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Asks the sampler for a token of no logit. */
    void sample_no_logit()
    {
        RandomGenerator random(K_SEED);
        static_cast<void>(GreedySampler().sample(LogitSpan(), random));
    }
} // namespace

/** @brief Tests that the largest logit wins, and the lowest index among equal ones. */
TEST(Unit_GenerationGreedySampler, ThePickIsTheLargestLogitAndTheFirstOfEquals)
{
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);

    EXPECT_EQ(sampler.sample(Support::span_of({0.5f, 2.0f, -1.0f}), random), 1);
    EXPECT_EQ(sampler.sample(Support::span_of({0.5f, 2.0f, 2.0f, -1.0f}), random), 1);
    EXPECT_EQ(sampler.sample(Support::span_of({-3.0f}), random), 0);
    EXPECT_EQ(sampler.name(), K_GREEDY_SAMPLER);
}

/** @brief Tests that a pick leaves the generator as it was. */
TEST(Unit_GenerationGreedySampler, APickDrawsNothing)
{
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);
    const RandomGenerator untouched(K_SEED);

    static_cast<void>(sampler.sample(Support::span_of({0.0f, 1.0f}), random));

    EXPECT_TRUE(random == untouched);
}

/** @brief Tests that a row of no logit stops the process. */
TEST(Unit_GenerationGreedySamplerDeathTest, NoLogitIsFatal)
{
    SA_EXPECT_FATAL(sample_no_logit(), "no logit");
}
