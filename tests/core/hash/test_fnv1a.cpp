/**************************************************************************/
/* test_fnv1a.cpp                                                         */
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
 * @file test_fnv1a.cpp
 * @brief Tests the 64-bit FNV-1a hash of fnv1a.hpp against values computed apart from it.
 * @author Mustafa Garip
 */

#include <SushiAI/hash/fnv1a.hpp>

#include <cstdint>
#include <string>
#include <gtest/gtest.h>

using namespace SushiAI;

/** @brief Tests that the text fold gives the published FNV-1a values of three texts. */
TEST(Small_FNV1a, TheTextFoldGivesThePublishedValues)
{
    EXPECT_EQ(Hash::fnv1a_text(Hash::K_FNV1A_OFFSET_BASIS, ""), 0xcbf29ce484222325ull);
    EXPECT_EQ(Hash::fnv1a_text(Hash::K_FNV1A_OFFSET_BASIS, "a"), 0xaf63dc4c8601ec8cull);
    EXPECT_EQ(Hash::fnv1a_text(Hash::K_FNV1A_OFFSET_BASIS, "foobar"), 0x85944171f73967e8ull);
}

/** @brief Tests that a word folds as its eight bytes, low byte first. */
TEST(Small_FNV1a, AWordFoldsAsItsBytesLowFirst)
{
    const std::string bytes("\xef\xcd\xab\x89\x67\x45\x23\x01", 8);

    EXPECT_EQ(Hash::fnv1a_word(Hash::K_FNV1A_OFFSET_BASIS, 0x0123456789abcdefull),
              Hash::fnv1a_text(Hash::K_FNV1A_OFFSET_BASIS, bytes));
}

/**
 * @brief Tests that three words fold to the value the replica step's fingerprint has held,
 *        computed apart from SushiAI over the words 1, 2 and 0x0123456789abcdef.
 */
TEST(Small_FNV1a, ThreeWordsKeepTheFingerprintOfTheReplicaStep)
{
    std::uint64_t hash = Hash::K_FNV1A_OFFSET_BASIS;
    for (const std::uint64_t word : {1ull, 2ull, 0x0123456789abcdefull})
        hash = Hash::fnv1a_word(hash, word);

    EXPECT_EQ(hash, 0x78165499fd352566ull);
}

/** @brief Tests that the fold is a constant expression. */
TEST(Small_FNV1a, TheFoldIsAConstantExpression)
{
    constexpr std::uint64_t K_FOLDED = Hash::fnv1a_text(Hash::K_FNV1A_OFFSET_BASIS, "a");
    static_assert(K_FOLDED == 0xaf63dc4c8601ec8cull, "the fold runs at compile time");
    SUCCEED();
}
