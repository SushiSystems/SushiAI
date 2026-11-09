/**************************************************************************/
/* test_codec.cpp                                                         */
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
 * @file test_codec.cpp
 * @brief Tests the encoder from text bytes to token ids and the decoder back.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "text_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/text/bpe_trainer.hpp>
#include <SushiAI/language/text/codec.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Text;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the seed of every generated text of these tests. */
    constexpr std::uint32_t K_SEED = 20261011u;

    /** @brief Returns the ids as the span decode_tokens reads. */
    SushiRuntime::span<const TokenID> span_of(const std::vector<TokenID>& ids)
    {
        return SushiRuntime::span<const TokenID>(ids.data(), ids.size());
    }

    /** @brief Expects a text to encode to ids of the vocabulary that decode to the text. */
    void expect_round_trip(const Vocabulary& vocabulary, const std::string& text)
    {
        std::vector<TokenID> ids;
        ASSERT_TRUE(SaTest::status_is_ok(encode_text(vocabulary, text, ids)));

        for (TokenID id : ids)
        {
            ASSERT_TRUE(vocabulary.contains(id)) << id;
            ASSERT_NE(vocabulary.kind_of(id), TokenKind::Special) << id;
        }

        std::string decoded;
        ASSERT_TRUE(SaTest::status_is_ok(decode_tokens(vocabulary, span_of(ids), decoded)));
        EXPECT_EQ(decoded, text);
    }

    /** @brief Returns the vocabulary of 400 tokens trained on 20000 drawn bytes. */
    Vocabulary trained_vocabulary(const std::string& corpus)
    {
        BPETrainerOptions options;
        options.vocabulary_size = 400;
        return SaTest::require_value(train_bpe(corpus, options));
    }

    /** @brief Returns the 20000 bytes the vocabulary of these tests is trained on. */
    std::string training_corpus()
    {
        std::mt19937 generator(K_SEED);
        return Support::drawn_bytes(generator, 20000, "abcdef  \n");
    }

    /** @brief Holds the rounds of the property test. */
    constexpr std::size_t K_ROUNDS = 64;

    /** @brief Expects K_ROUNDS texts of random bytes of up to 1999 bytes each to round trip. */
    void expect_random_texts_round_trip(const Vocabulary& vocabulary, std::mt19937& generator)
    {
        for (std::size_t round = 0; round < K_ROUNDS; ++round)
        {
            const std::size_t length = generator() % 2000;
            expect_round_trip(vocabulary, Support::drawn_bytes(generator, length));
        }
    }

    /** @brief Returns the texts a round trip is checked for beside the random ones. */
    std::vector<std::string> edge_texts()
    {
        return {training_corpus(), "", std::string("\xFF\xFE\xC0\x80\x00\xED\xA0\x80", 8)};
    }

    /** @brief Names the ids a decode refuses and what its message says. */
    struct RefusedIDs
    {
        std::vector<TokenID> ids; /**< The ids to decode. */
        std::string first;        /**< A text the message holds. */
        std::string second;       /**< A second text the message holds. */
        std::string third;        /**< A third text the message holds. */
    };

    /** @brief Returns the ids the decoder refuses. */
    std::vector<RefusedIDs> refused_ids()
    {
        return {{{72, 9999, 105}, "9999", "index 1", "[0, 257)"},
                {{-3}, "-3", "index 0", "decode_tokens"}};
    }

    /** @brief Provides the edge texts. */
    class Small_TextCodecEdgeText : public ::testing::TestWithParam<std::string>
    {

    };

    /** @brief Provides the ids the decoder refuses. */
    class Small_TextCodecRefusal : public ::testing::TestWithParam<RefusedIDs>
    {

    };
} // namespace

/** @brief Tests that random bytes of every value, invalid UTF-8 among them, come back. */
TEST(Small_TextCodec, RandomBytesSurviveARoundTrip)
{
    std::mt19937 generator(K_SEED + 1);
    const Vocabulary vocabulary = trained_vocabulary(training_corpus());

    expect_random_texts_round_trip(vocabulary, generator);
}

/** @brief Tests that a fixed text, the training corpus and invalid UTF-8 among them, comes back. */
TEST_P(Small_TextCodecEdgeText, AnEdgeTextSurvivesARoundTrip)
{
    const Vocabulary vocabulary = trained_vocabulary(training_corpus());

    expect_round_trip(vocabulary, GetParam());
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_TextCodecEdgeText, ::testing::ValuesIn(edge_texts()));

/** @brief Tests that a run of 5000 bytes without a blank, cut into chunks, comes back. */
TEST(Small_TextCodec, ATextLongerThanOneChunkSurvivesARoundTrip)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary,
                            Vocabulary::make(Support::doubling_merges(8)));

    expect_round_trip(vocabulary, std::string(5000, 'a'));
}

/** @brief Tests that 5000 bytes without a blank are cut into chunks of 256 bytes. */
TEST(Small_TextCodec, ATextLongerThanOneChunkEncodesChunkByChunk)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary,
                            Vocabulary::make(Support::doubling_merges(8)));

    // 5000 bytes are 19 chunks of 256 bytes, one token each, and one of 136 = 128 + 8.
    std::vector<TokenID> ids;
    ASSERT_TRUE(SaTest::status_is_ok(encode_text(vocabulary, std::string(5000, 'a'), ids)));
    ASSERT_EQ(ids.size(), 21u);
    EXPECT_EQ(ids[0], 264);
    EXPECT_EQ(ids[19], 263);
    EXPECT_EQ(ids[20], 259);
}

/** @brief Tests that the merge learned first is applied first, from the left. */
TEST(Small_TextCodec, TheFirstMergeAppliesFirstFromTheLeft)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary doubled,
                            Vocabulary::make({Merge{97, 98}, Merge{257, 257}}));

    std::vector<TokenID> ids;
    ASSERT_TRUE(SaTest::status_is_ok(encode_text(doubled, "abababc", ids)));
    EXPECT_EQ(ids, (std::vector<TokenID>{258, 257, 99}));
}

/** @brief Tests that a merge learned earlier wins an overlap over one learned later. */
TEST(Small_TextCodec, AnEarlierMergeWinsAnOverlapOverALaterOne)
{
    // (b, c) was learned before (a, b), so "abc" is a and bc, not ab and c.
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary ordered,
                            Vocabulary::make({Merge{98, 99}, Merge{97, 98}}));

    std::vector<TokenID> ids;
    ASSERT_TRUE(SaTest::status_is_ok(encode_text(ordered, "abc ab", ids)));
    EXPECT_EQ(ids, (std::vector<TokenID>{97, 257, 32, 258}));
}

/** @brief Tests that a text encodes to the tokens its training left it in. */
TEST(Small_TextCodec, ATrainedCorpusEncodesToItsTrainedTokens)
{
    BPETrainerOptions options;
    options.vocabulary_size = 300;
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, train_bpe("ab ab cd cd", options));

    std::vector<TokenID> ids;
    ASSERT_TRUE(SaTest::status_is_ok(encode_text(vocabulary, "ab ab cd cd", ids)));
    EXPECT_EQ(ids, (std::vector<TokenID>{258, 32, 258, 259, 259}));
}

/** @brief Tests that the encoder appends to the ids it is given. */
TEST(Small_TextCodec, EncodeAppendsToTheIDsItIsGiven)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));

    std::vector<TokenID> ids = {7};
    ASSERT_TRUE(SaTest::status_is_ok(encode_text(vocabulary, "A", ids)));
    EXPECT_EQ(ids, (std::vector<TokenID>{7, 65}));
}

/** @brief Tests that a special token decodes to no byte. */
TEST(Small_TextCodec, ASpecialTokenDecodesToNoByte)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));

    std::string text = ">";
    const std::vector<TokenID> ids = {72, K_END_OF_TEXT, 105};
    ASSERT_TRUE(SaTest::status_is_ok(decode_tokens(vocabulary, span_of(ids), text)));
    EXPECT_EQ(text, ">Hi");
}

/** @brief Tests that an id the vocabulary does not hold is a failure naming its index. */
TEST_P(Small_TextCodecRefusal, DecodeRefusesAnIDOutsideTheVocabulary)
{
    const RefusedIDs& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));

    std::string text;
    SA_EXPECT_FAILURE(decode_tokens(vocabulary, span_of(row.ids), text),
                      StatusCode::InvalidDescription, "decode_tokens", row.first, row.second,
                      row.third);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_TextCodecRefusal, ::testing::ValuesIn(refused_ids()));
