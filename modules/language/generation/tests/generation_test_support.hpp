/**************************************************************************/
/* generation_test_support.hpp                                            */
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
 * @file generation_test_support.hpp
 * @brief Defines what the tests of the samplers share: counted draws and a compared share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/text/token.hpp>

/** @brief Holds what the tests of the language generation module share. */
namespace SushiAI::Language::Generation::Support
{
    /** @brief Holds the draws a frequency is counted over. */
    inline constexpr int K_DRAWS = 20000;

    /** @brief Bounds how far a counted share may lie from its probability, absolutely. */
    inline constexpr double K_SHARE_TOLERANCE = 0.02;

    /** @brief Views the logits of a test as the span a sampler reads. */
    inline LogitSpan span_of(const std::vector<float>& logits)
    {
        return LogitSpan(logits.data(), logits.size());
    }

    /** @brief Counts how often each token is picked in K_DRAWS draws under one seed. */
    inline std::vector<int> count_draws(const Sampler& sampler, const std::vector<float>& logits,
                                        std::uint64_t seed)
    {
        RandomGenerator random(seed);
        std::vector<int> counts(logits.size(), 0);
        for (int draw = 0; draw < K_DRAWS; ++draw)
            ++counts[static_cast<std::size_t>(sampler.sample(span_of(logits), random))];

        return counts;
    }

    /** @brief Returns the tokens of @p draws picks under one seed, in order. */
    inline std::vector<Text::TokenID> draw_sequence(const Sampler& sampler,
                                                    const std::vector<float>& logits,
                                                    std::uint64_t seed, int draws)
    {
        RandomGenerator random(seed);
        std::vector<Text::TokenID> picked;
        for (int draw = 0; draw < draws; ++draw)
            picked.push_back(sampler.sample(span_of(logits), random));

        return picked;
    }

    /** @brief Expects the share of one count to lie within K_SHARE_TOLERANCE of @p wanted. */
    inline void expect_share(int count, double wanted)
    {
        EXPECT_NEAR(static_cast<double>(count) / K_DRAWS, wanted, K_SHARE_TOLERANCE);
    }

} // namespace SushiAI::Language::Generation::Support
