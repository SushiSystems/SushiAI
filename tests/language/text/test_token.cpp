/**************************************************************************/
/* test_token.cpp                                                         */
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
 * @file test_token.cpp
 * @brief Tests the token id type and the limits of a vocabulary.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <limits>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiAI/language/text/token.hpp>

using namespace SushiAI::Language::Text;

/** @brief Tests that a token id is the int32 every id tensor and token file holds. */
TEST(Small_LanguageTextToken, ATokenIDIsAnInt32)
{
    EXPECT_TRUE((std::is_same<TokenID, std::int32_t>::value));
}

/** @brief Tests that the byte tokens come first and end-of-text is the first id after them. */
TEST(Small_LanguageTextToken, EndOfTextFollowsTheByteTokens)
{
    EXPECT_EQ(K_BYTE_TOKEN_COUNT, 256);
    EXPECT_EQ(K_END_OF_TEXT, K_BYTE_TOKEN_COUNT);
    EXPECT_EQ(K_END_OF_TEXT_NAME, "<end_of_text>");
    EXPECT_LE(K_END_OF_TEXT_NAME.size(), K_MAX_SPECIAL_NAME_BYTES);
}

/** @brief Tests that the largest vocabulary holds every byte and special token and fits an id. */
TEST(Small_LanguageTextToken, TheLargestVocabularyFitsATokenID)
{
    EXPECT_EQ(K_MAX_VOCABULARY_SIZE, 1 << 20);
    EXPECT_GT(K_MAX_VOCABULARY_SIZE, K_BYTE_TOKEN_COUNT + K_MAX_SPECIAL_TOKENS);
    EXPECT_LE(K_MAX_VOCABULARY_SIZE, std::numeric_limits<TokenID>::max());
    EXPECT_GE(K_MAX_TOKEN_BYTES, 1u);
}
