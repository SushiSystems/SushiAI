/**************************************************************************/
/* test_incremental.cpp                                                   */
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
 * @file test_incremental.cpp
 * @brief Tests the leaf names of the incremental blocks and the score of a masked position.
 * @author Mustafa Garip
 */

#include <cmath>
#include <set>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/language/blocks/incremental.hpp>

using namespace SushiAI::Language::Blocks;

/** @brief Tests that the five leaves an incremental run is fed carry five different names. */
TEST(Small_BlockIncremental, EveryLeafHasItsOwnName)
{
    const std::set<std::string_view> names{K_POSITION_LEAF, K_ATTENTION_MASK_LEAF,
                                           K_CACHE_SLOT_LEAF, K_ROTARY_COSINE_LEAF,
                                           K_ROTARY_SINE_LEAF};

    EXPECT_EQ(names.size(), 5u);
    EXPECT_EQ(names.count(""), 0u);
}

/** @brief Tests that a masked score is finite in float32 and its softmax weight is zero. */
TEST(Small_BlockIncremental, AMaskedScoreTakesNoWeightInTheSoftmax)
{
    const float masked = static_cast<float>(K_MASKED_ATTENTION_SCORE);

    EXPECT_TRUE(std::isfinite(masked));
    EXPECT_LT(masked, 0.0f);
    // A readable position scores 0 after the shift by the maximum, so its weight is exp(0).
    EXPECT_EQ(std::exp(masked) / (std::exp(masked) + std::exp(0.0f)), 0.0f);
}
