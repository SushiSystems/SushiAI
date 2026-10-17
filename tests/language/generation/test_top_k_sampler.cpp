/**************************************************************************/
/* test_top_k_sampler.cpp                                                 */
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
 * @file test_top_k_sampler.cpp
 * @brief Tests TopKSampler against hand-worked distributions and under a seed.
 * @author Mustafa Garip
 */

#include <cmath>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "generation_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/top_k_sampler.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;

namespace
{
    /** @brief Holds the seed of every generator of these tests. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Holds ln 2, so that a logit of it has twice the weight of a logit of 0. */
    const float K_LN_2 = std::log(2.0f);
} // namespace

/** @brief Tests logits 0, ln 2, 0, -5 with k = 2: tokens 1 and 0 share 2/3 and 1/3. */
TEST(Small_GenerationTopKSampler, OnlyTheKLargestAreDrawnAndTheLowerIndexWinsATie)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TopKSampler> sampler,
                            TopKSampler::make(2, 1.0));

    const std::vector<int> counts =
        Support::count_draws(*sampler, {0.0f, K_LN_2, 0.0f, -5.0f}, K_SEED);

    Support::expect_share(counts[0], 1.0 / 3.0);
    Support::expect_share(counts[1], 2.0 / 3.0);
    EXPECT_EQ(counts[2], 0);
    EXPECT_EQ(counts[3], 0);
    EXPECT_EQ(sampler->name(), K_TOP_K_SAMPLER);
}

/** @brief Tests that k = 1 always picks the largest logit, whatever the seed. */
TEST(Small_GenerationTopKSampler, KeepingOneTokenIsTheGreedyPick)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TopKSampler> sampler,
                            TopKSampler::make(1, 1.0));

    const std::vector<int> counts =
        Support::count_draws(*sampler, {0.3f, 0.2f, 0.9f, 0.9f}, K_SEED);

    EXPECT_EQ(counts[2], Support::K_DRAWS);
}

/** @brief Tests that a k above the row keeps every token: 0 and ln 2 share 1/3 and 2/3. */
TEST(Small_GenerationTopKSampler, AKAboveTheRowKeepsEveryToken)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TopKSampler> sampler,
                            TopKSampler::make(50, 1.0));

    const std::vector<int> counts = Support::count_draws(*sampler, {0.0f, K_LN_2}, K_SEED);

    Support::expect_share(counts[0], 1.0 / 3.0);
    Support::expect_share(counts[1], 2.0 / 3.0);
}

/** @brief Tests that one seed gives one sequence. */
TEST(Small_GenerationTopKSampler, ASeedFixesTheSequence)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TopKSampler> sampler,
                            TopKSampler::make(4, 0.8));
    const std::vector<float> logits{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};

    EXPECT_EQ(Support::draw_sequence(*sampler, logits, K_SEED, 64),
              Support::draw_sequence(*sampler, logits, K_SEED, 64));
}

/** @brief Tests that a k below one and a temperature that is not positive are refused. */
TEST(Small_GenerationTopKSampler, MakeRefusesAKBelowOneAndABadTemperature)
{
    SA_EXPECT_FAILURE(TopKSampler::make(0, 1.0), StatusCode::InvalidDescription, "'top_k'",
                      "keep 0 tokens");
    SA_EXPECT_FAILURE(TopKSampler::make(3, 0.0), StatusCode::InvalidDescription, "'top_k'",
                      "temperature");
}
