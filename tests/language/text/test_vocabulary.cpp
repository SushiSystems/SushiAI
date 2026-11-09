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
 * @file test_vocabulary.cpp
 * @brief Tests the vocabulary of a byte-pair tokenizer.
 * @author Mustafa Garip
 */

#include <optional>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "text_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Text;

namespace
{
    /** @brief Names merges the vocabulary refuses and what its message says. */
    struct RefusedMerges
    {
        std::vector<Merge> merges; /**< The merges given to make. */
        std::string first;         /**< A text the message holds. */
        std::string second;        /**< A second text the message holds. */
    };

    /** @brief Returns the merges of a token the vocabulary may not join. */
    std::vector<RefusedMerges> refused_merges()
    {
        return {{{Merge{97, 257}}, "merge 0", "97 and 257"},
                {{Merge{97, 98}, Merge{259, 97}}, "merge 1", "259 and 97"},
                {{Merge{K_END_OF_TEXT, 97}}, "merge 0", "256 and 97"},
                {{Merge{-1, 97}}, "merge 0", "-1 and 97"}};
    }

    /** @brief Names special names the vocabulary refuses and what its message says. */
    struct RefusedNames
    {
        std::vector<std::string> names; /**< The special names given to make. */
        std::string first;              /**< A text the message holds. */
        std::string second;             /**< A second text the message holds. */
    };

    /** @brief Returns the special names the vocabulary refuses. */
    std::vector<RefusedNames> refused_names()
    {
        return {{{""}, "special token 1", "0 bytes"},
                {{std::string(256, 'x')}, "256 bytes", "1 to 255"},
                {{"<pad>", "<pad>"}, "'<pad>'", "named twice"},
                {{std::string(K_END_OF_TEXT_NAME)}, "named twice", "named twice"},
                {std::vector<std::string>(256, "x"), "257 special tokens", "at most 256"}};
    }

    /** @brief Provides the merges the vocabulary refuses. */
    class Small_TextVocabularyMergeRefusal : public ::testing::TestWithParam<RefusedMerges>
    {

    };

    /** @brief Provides the special names the vocabulary refuses. */
    class Small_TextVocabularyNameRefusal : public ::testing::TestWithParam<RefusedNames>
    {

    };
} // namespace

/** @brief Tests that a vocabulary without merges holds the 256 byte tokens. */
TEST(Small_TextVocabulary, AVocabularyWithoutMergesHoldsTheByteTokens)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));

    EXPECT_EQ(vocabulary.kind_of(0), TokenKind::Byte);
    EXPECT_EQ(vocabulary.kind_of(255), TokenKind::Byte);
    EXPECT_EQ(vocabulary.bytes_of(65), "A");
    EXPECT_EQ(vocabulary.bytes_of(255), std::string(1, '\xFF'));
}

/** @brief Tests that a vocabulary without merges holds end-of-text, which stands for no byte. */
TEST(Small_TextVocabulary, AVocabularyWithoutMergesHoldsEndOfText)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));

    EXPECT_EQ(vocabulary.kind_of(K_END_OF_TEXT), TokenKind::Special);
    EXPECT_TRUE(vocabulary.bytes_of(K_END_OF_TEXT).empty());
    EXPECT_EQ(vocabulary.find_special(K_END_OF_TEXT_NAME), std::optional<TokenID>(256));
}

/** @brief Tests that a vocabulary without merges holds the ids 0 to 256 and no other. */
TEST(Small_TextVocabulary, AVocabularyWithoutMergesHoldsTheIDsUpToEndOfText)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));

    EXPECT_EQ(vocabulary.size(), 257);
    EXPECT_EQ(vocabulary.first_merged(), 257);
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
    ASSERT_EQ(vocabulary.merges().size(), 2u);
    EXPECT_EQ(vocabulary.merges()[1], (Merge{257, 99}));
}

/** @brief Tests that find_merge returns the id of a merged pair and nothing for an absent pair. */
TEST(Small_TextVocabulary, FindMergeLooksUpAPairAndAnAbsentPair)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary,
                            Vocabulary::make({Merge{97, 98}, Merge{257, 99}}));

    EXPECT_EQ(vocabulary.find_merge(97, 98), std::optional<TokenID>(257));
    EXPECT_EQ(vocabulary.find_merge(257, 99), std::optional<TokenID>(258));
    EXPECT_FALSE(vocabulary.find_merge(98, 97).has_value());
    EXPECT_FALSE(vocabulary.find_merge(-1, 97).has_value());
}

/** @brief Tests that the special tokens follow end-of-text and precede the merged tokens. */
TEST(Small_TextVocabulary, SpecialTokensFollowEndOfText)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary,
                            Vocabulary::make({Merge{97, 98}}, {"<pad>", "<mask>"}));

    EXPECT_EQ(vocabulary.size(), 260);
    EXPECT_EQ(vocabulary.find_special("<pad>"), std::optional<TokenID>(257));
    EXPECT_EQ(vocabulary.find_special("<mask>"), std::optional<TokenID>(258));
    EXPECT_EQ(vocabulary.first_merged(), 259);
    EXPECT_EQ(vocabulary.bytes_of(259), "ab");
    ASSERT_EQ(vocabulary.special_names().size(), 3u);
    EXPECT_EQ(vocabulary.special_names()[0], K_END_OF_TEXT_NAME);
}

/** @brief Tests that find_special returns nothing for a name the vocabulary does not hold. */
TEST(Small_TextVocabulary, FindSpecialLooksUpAnAbsentName)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}, {"<pad>"}));

    EXPECT_FALSE(vocabulary.find_special("<absent>").has_value());
}

/** @brief Tests that a merge of itself, a later, a special or a negative token is refused. */
TEST_P(Small_TextVocabularyMergeRefusal, MakeRefusesAMergeOfATokenItMayNotJoin)
{
    const RefusedMerges& row = GetParam();

    SA_EXPECT_FAILURE(Vocabulary::make(row.merges), StatusCode::InvalidDescription, "Vocabulary",
                      row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_TextVocabularyMergeRefusal,
                         ::testing::ValuesIn(refused_merges()));

/** @brief Tests that a pair two merges name is refused at the second. */
TEST(Small_TextVocabulary, MakeRefusesAPairNamedTwice)
{
    SA_EXPECT_FAILURE(Vocabulary::make({Merge{97, 98}, Merge{97, 98}}),
                      StatusCode::InvalidDescription, "merge 1", "97 and 98", "earlier merge");
}

/** @brief Tests that a merged token of 512 bytes is refused. */
TEST(Small_TextVocabulary, MakeRefusesATokenAboveTheByteLimit)
{
    SA_EXPECT_FAILURE(Vocabulary::make(Support::doubling_merges(9)),
                      StatusCode::InvalidDescription, "merge 8", "512 bytes", "at most 256");
}

/** @brief Tests that a merged token of 256 bytes, the limit, is held. */
TEST(Small_TextVocabulary, MakeHoldsATokenOfTheByteLimit)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary,
                            Vocabulary::make(Support::doubling_merges(8)));

    EXPECT_EQ(vocabulary.bytes_of(264), std::string(256, 'a'));
}

/** @brief Tests that an empty, an overlong and a repeated special name are refused. */
TEST_P(Small_TextVocabularyNameRefusal, MakeRefusesASpecialNameOutOfBounds)
{
    const RefusedNames& row = GetParam();

    SA_EXPECT_FAILURE(Vocabulary::make({}, row.names), StatusCode::InvalidDescription, row.first,
                      row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_TextVocabularyNameRefusal,
                         ::testing::ValuesIn(refused_names()));

/** @brief Tests that the bytes of an id outside the vocabulary stop the process. */
TEST(Medium_TextVocabularyDeath, AnIDOutsideTheVocabularyStopsTheProcess)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));

    SA_EXPECT_FATAL((void)vocabulary.bytes_of(257), "Vocabulary", "257", "[0, 257)");
    SA_EXPECT_FATAL((void)vocabulary.kind_of(-1), "Vocabulary", "-1");
}
