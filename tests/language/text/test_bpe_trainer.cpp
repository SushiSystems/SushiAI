/**************************************************************************/
/* test_bpe_trainer.cpp                                                   */
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
 * @file test_bpe_trainer.cpp
 * @brief Tests the trainer that learns the merges of a byte-pair vocabulary.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "text_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/text/bpe_trainer.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Text;

namespace
{
    /** @brief Holds the seed of every generated corpus of these tests. */
    constexpr std::uint32_t K_SEED = 20261011u;

    /** @brief Returns the options of a run to a vocabulary size with end-of-text alone. */
    BPETrainerOptions options_of(std::int32_t vocabulary_size)
    {
        BPETrainerOptions options;
        options.vocabulary_size = vocabulary_size;
        return options;
    }

    /** @brief Names a vocabulary size the trainer refuses and what its message says. */
    struct SizeCase
    {
        std::int32_t size;  /**< The size asked for. */
        std::string first;  /**< A text the message holds. */
        std::string second; /**< A second text the message holds. */
    };

    /** @brief Returns the sizes the trainer refuses. */
    std::vector<SizeCase> refused_sizes()
    {
        return {{256, "256 tokens", "257 to"}, {K_MAX_VOCABULARY_SIZE + 1, "asked for", "train_bpe"}};
    }

    /** @brief Provides the sizes the trainer refuses. */
    class Small_TextBpeTrainerSizeRefusal : public ::testing::TestWithParam<SizeCase>
    {

    };
} // namespace

/**
 * @brief Tests the merges of "abababc", worked by hand.
 *
 * (a, b) occurs 3 times and becomes 257, leaving 257 257 257 c. (257, 257) occurs twice and
 * becomes 258 at the left, leaving 258 257 c, where every pair occurs once.
 */
TEST(Small_TextBpeTrainer, ItLearnsTheHandWorkedMergesOfATinyCorpus)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, train_bpe("abababc", options_of(300)));

    ASSERT_EQ(vocabulary.merges().size(), 2u);
    EXPECT_EQ(vocabulary.merges()[0], (Merge{97, 98}));
    EXPECT_EQ(vocabulary.merges()[1], (Merge{257, 257}));
    EXPECT_EQ(vocabulary.size(), 259);
    EXPECT_EQ(vocabulary.bytes_of(258), "abab");
}

/**
 * @brief Tests that pairs of one count merge by the smaller left id, then the smaller right.
 *
 * "ab ab cd cd" is the chunks "ab", " ab" and " cd" twice. (32, 99), (97, 98) and (99, 100)
 * occur twice: (32, 99) becomes 257, which ends (99, 100) and makes (257, 100) occur twice;
 * (97, 98) becomes 258; (257, 100) becomes 259.
 */
TEST(Small_TextBpeTrainer, PairsOfOneCountMergeBySmallerLeftThenRightID)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary,
                            train_bpe("ab ab cd cd", options_of(300)));

    ASSERT_EQ(vocabulary.merges().size(), 3u);
    EXPECT_EQ(vocabulary.merges()[0], (Merge{32, 99}));
    EXPECT_EQ(vocabulary.merges()[1], (Merge{97, 98}));
    EXPECT_EQ(vocabulary.merges()[2], (Merge{257, 100}));
}

/** @brief Tests that no merge joins the end of one chunk with the blank that follows it. */
TEST(Small_TextBpeTrainer, NoMergeCrossesTheStartOfABlankRun)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary,
                            train_bpe("a b a b a b", options_of(300)));

    ASSERT_EQ(vocabulary.merges().size(), 2u);
    EXPECT_EQ(vocabulary.merges()[0], (Merge{32, 98}));
    EXPECT_EQ(vocabulary.merges()[1], (Merge{32, 97}));
}

/** @brief Tests that two runs over one corpus and one seed give the same merges. */
TEST(Small_TextBpeTrainer, TheSameCorpusGivesTheSameVocabulary)
{
    std::mt19937 generator(K_SEED);
    const std::string corpus = Support::drawn_bytes(generator, 20000, "abcdef  \n");

    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary first, train_bpe(corpus, options_of(400)));
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary second, train_bpe(corpus, options_of(400)));

    ASSERT_FALSE(first.merges().empty());
    Support::expect_same_merges(first, second);
}

/** @brief Tests that a run to a smaller size learns the first merges of a run to a larger. */
TEST(Small_TextBpeTrainer, ARunStoppedEarlyLearnsTheFirstMergesOfALongerRun)
{
    std::mt19937 generator(K_SEED);
    const std::string corpus = Support::drawn_bytes(generator, 4000, "abc ");

    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary whole, train_bpe(corpus, options_of(290)));

    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary stopped, train_bpe(corpus, options_of(270)));

    ASSERT_FALSE(stopped.merges().empty());
    Support::expect_merges_begin_with(whole, stopped);
}

/** @brief Tests that training stops when the vocabulary has the size asked for. */
TEST(Small_TextBpeTrainer, ItStopsAtTheSizeAskedFor)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, train_bpe("abababc", options_of(258)));

    ASSERT_EQ(vocabulary.merges().size(), 1u);
    EXPECT_EQ(vocabulary.merges()[0], (Merge{97, 98}));
}

/** @brief Tests that a pair that occurs once is not merged. */
TEST(Small_TextBpeTrainer, APairThatOccursOnceIsNotMerged)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, train_bpe("abcdef", options_of(300)));

    EXPECT_EQ(vocabulary.size(), 257);
}

/** @brief Tests that an empty corpus trains to the byte tokens and end-of-text alone. */
TEST(Small_TextBpeTrainer, AnEmptyCorpusTrainsToTheBaseVocabulary)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, train_bpe("", options_of(300)));

    EXPECT_EQ(vocabulary.size(), 257);
}

/** @brief Tests that the special tokens take their ids before the first merged token. */
TEST(Small_TextBpeTrainer, SpecialTokensPrecedeTheMergedTokens)
{
    BPETrainerOptions options = options_of(300);
    options.special_names = {"<pad>"};

    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, train_bpe("abababc", options));

    ASSERT_EQ(vocabulary.merges().size(), 2u);
    EXPECT_EQ(vocabulary.merges()[1], (Merge{258, 258}));
    EXPECT_EQ(vocabulary.bytes_of(259), "abab");
}

/** @brief Tests that a size below the base tokens or above the limit is refused. */
TEST_P(Small_TextBpeTrainerSizeRefusal, ItRefusesASizeOutOfBounds)
{
    const SizeCase& row = GetParam();

    SA_EXPECT_FAILURE(train_bpe("abab", options_of(row.size)), StatusCode::InvalidDescription,
                      "train_bpe", row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_TextBpeTrainerSizeRefusal,
                         ::testing::ValuesIn(refused_sizes()));
