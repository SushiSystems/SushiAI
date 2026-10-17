/**************************************************************************/
/* test_vocabulary.cpp                                                    */
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
 * @file test_vocabulary.cpp
 * @brief Tests the vocabulary of a byte-pair tokenizer.
 * @author Mustafa Garip
 */

#include <optional>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Text;

/** @brief Tests that a vocabulary without merges holds the bytes and end-of-text. */
TEST(Small_TextVocabulary, AVocabularyWithoutMergesHoldsTheBytesAndEndOfText)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));

    EXPECT_EQ(vocabulary.size(), 257);
    EXPECT_EQ(vocabulary.first_merged(), 257);
    EXPECT_EQ(vocabulary.kind_of(0), TokenKind::Byte);
    EXPECT_EQ(vocabulary.kind_of(255), TokenKind::Byte);
    EXPECT_EQ(vocabulary.bytes_of(65), "A");
    EXPECT_EQ(vocabulary.bytes_of(255), std::string(1, '\xFF'));
    EXPECT_EQ(vocabulary.kind_of(K_END_OF_TEXT), TokenKind::Special);
    EXPECT_TRUE(vocabulary.bytes_of(K_END_OF_TEXT).empty());
    EXPECT_EQ(vocabulary.find_special(K_END_OF_TEXT_NAME), std::optional<TokenID>(256));
    EXPECT_TRUE(vocabulary.contains(256));
    EXPECT_FALSE(vocabulary.contains(257));
    EXPECT_FALSE(vocabulary.contains(-1));
}

/** @brief Tests that merge i has the id first_merged() + i and the bytes of its two parts. */
TEST(Small_TextVocabulary, MergedTokensTakeIDsInTheOrderOfTheirMerges)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary,
                            Vocabulary::make({Merge{97, 98}, Merge{257, 99}}));

    EXPECT_EQ(vocabulary.size(), 259);
    EXPECT_EQ(vocabulary.kind_of(257), TokenKind::Merged);
    EXPECT_EQ(vocabulary.bytes_of(257), "ab");
    EXPECT_EQ(vocabulary.bytes_of(258), "abc");
    EXPECT_EQ(vocabulary.find_merge(97, 98), std::optional<TokenID>(257));
    EXPECT_EQ(vocabulary.find_merge(257, 99), std::optional<TokenID>(258));
    EXPECT_FALSE(vocabulary.find_merge(98, 97).has_value());
    EXPECT_FALSE(vocabulary.find_merge(-1, 97).has_value());
    ASSERT_EQ(vocabulary.merges().size(), 2u);
    EXPECT_EQ(vocabulary.merges()[1], (Merge{257, 99}));
}

/** @brief Tests that the special tokens follow end-of-text and precede the merged tokens. */
TEST(Small_TextVocabulary, SpecialTokensFollowEndOfText)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary,
                            Vocabulary::make({Merge{97, 98}}, {"<pad>", "<mask>"}));

    EXPECT_EQ(vocabulary.size(), 260);
    EXPECT_EQ(vocabulary.find_special("<pad>"), std::optional<TokenID>(257));
    EXPECT_EQ(vocabulary.find_special("<mask>"), std::optional<TokenID>(258));
    EXPECT_FALSE(vocabulary.find_special("<absent>").has_value());
    EXPECT_EQ(vocabulary.first_merged(), 259);
    EXPECT_EQ(vocabulary.bytes_of(259), "ab");
    ASSERT_EQ(vocabulary.special_names().size(), 3u);
    EXPECT_EQ(vocabulary.special_names()[0], K_END_OF_TEXT_NAME);
}

/** @brief Tests that a merge of itself, a later, a special or a negative token is refused. */
TEST(Small_TextVocabulary, MakeRefusesAMergeOfATokenItMayNotJoin)
{
    SA_EXPECT_FAILURE(Vocabulary::make({Merge{97, 257}}), StatusCode::InvalidDescription,
                      "Vocabulary", "merge 0", "97 and 257");
    SA_EXPECT_FAILURE(Vocabulary::make({Merge{97, 98}, Merge{259, 97}}),
                      StatusCode::InvalidDescription, "merge 1", "259 and 97");
    SA_EXPECT_FAILURE(Vocabulary::make({Merge{K_END_OF_TEXT, 97}}),
                      StatusCode::InvalidDescription, "merge 0", "256 and 97");
    SA_EXPECT_FAILURE(Vocabulary::make({Merge{-1, 97}}), StatusCode::InvalidDescription,
                      "merge 0", "-1 and 97");
}

/** @brief Tests that a pair two merges name is refused at the second. */
TEST(Small_TextVocabulary, MakeRefusesAPairNamedTwice)
{
    SA_EXPECT_FAILURE(Vocabulary::make({Merge{97, 98}, Merge{97, 98}}),
                      StatusCode::InvalidDescription, "merge 1", "97 and 98", "earlier merge");
}

/** @brief Tests that a merged token of 512 bytes is refused, and one of 256 is held. */
TEST(Small_TextVocabulary, MakeRefusesATokenAboveTheByteLimit)
{
    // Merge i joins the token before it with itself, so it stands for 2^(i + 1) bytes.
    std::vector<Merge> merges = {Merge{97, 97}};
    for (TokenID id = 257; id < 264; ++id)
        merges.push_back(Merge{id, id});

    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make(merges));
    EXPECT_EQ(vocabulary.bytes_of(264), std::string(256, 'a'));

    merges.push_back(Merge{264, 264});
    SA_EXPECT_FAILURE(Vocabulary::make(merges), StatusCode::InvalidDescription, "merge 8",
                      "512 bytes", "at most 256");
}

/** @brief Tests that an empty, an overlong and a repeated special name are refused. */
TEST(Small_TextVocabulary, MakeRefusesASpecialNameOutOfBounds)
{
    SA_EXPECT_FAILURE(Vocabulary::make({}, {""}), StatusCode::InvalidDescription,
                      "special token 1", "0 bytes");
    SA_EXPECT_FAILURE(Vocabulary::make({}, {std::string(256, 'x')}),
                      StatusCode::InvalidDescription, "256 bytes", "1 to 255");
    SA_EXPECT_FAILURE(Vocabulary::make({}, {"<pad>", "<pad>"}), StatusCode::InvalidDescription,
                      "'<pad>'", "named twice");
    SA_EXPECT_FAILURE(Vocabulary::make({}, {std::string(K_END_OF_TEXT_NAME)}),
                      StatusCode::InvalidDescription, "named twice");
    SA_EXPECT_FAILURE(Vocabulary::make({}, std::vector<std::string>(256, "x")),
                      StatusCode::InvalidDescription, "257 special tokens", "at most 256");
}

/** @brief Tests that the bytes of an id outside the vocabulary stop the process. */
TEST(Small_TextVocabulary, AnIDOutsideTheVocabularyStopsTheProcess)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));

    SA_EXPECT_FATAL((void)vocabulary.bytes_of(257), "Vocabulary", "257", "[0, 257)");
    SA_EXPECT_FATAL((void)vocabulary.kind_of(-1), "Vocabulary", "-1");
}
