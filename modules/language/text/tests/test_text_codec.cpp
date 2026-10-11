/**************************************************************************/
/* test_text_codec.cpp                                                    */
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
 * @file test_text_codec.cpp
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
#include <SushiAI/core/status.hpp>
#include <SushiAI/language/text/bpe_trainer.hpp>
#include <SushiAI/language/text/codec.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>
#include <common/test_status.hpp>

#include "text_test_support.hpp"

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
} // namespace

/** @brief Tests that random bytes of every value, invalid UTF-8 among them, come back. */
TEST(Unit_TextCodec, RandomBytesSurviveARoundTrip)
{
    std::mt19937 generator(K_SEED);
    const std::string corpus = Support::drawn_bytes(generator, 20000, "abcdef  \n");

    BPETrainerOptions options;
    options.vocabulary_size = 400;
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, train_bpe(corpus, options));

    for (std::size_t round = 0; round < 64; ++round)
    {
        const std::size_t length = generator() % 2000;
        expect_round_trip(vocabulary, Support::drawn_bytes(generator, length));
    }

    expect_round_trip(vocabulary, corpus);
    expect_round_trip(vocabulary, "");
    expect_round_trip(vocabulary, std::string("\xFF\xFE\xC0\x80\x00\xED\xA0\x80", 8));
}

/** @brief Tests that a run of 5000 bytes without a blank, cut into chunks, comes back. */
TEST(Unit_TextCodec, ATextLongerThanOneChunkSurvivesARoundTrip)
{
    std::vector<Merge> merges = {Merge{97, 97}};
    for (TokenID id = 257; id < 264; ++id)
        merges.push_back(Merge{id, id});
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make(merges));

    const std::string text(5000, 'a');
    expect_round_trip(vocabulary, text);

    // 5000 bytes are 19 chunks of 256 bytes, one token each, and one of 136 = 128 + 8.
    std::vector<TokenID> ids;
    ASSERT_TRUE(SaTest::status_is_ok(encode_text(vocabulary, text, ids)));
    ASSERT_EQ(ids.size(), 21u);
    EXPECT_EQ(ids[0], 264);
    EXPECT_EQ(ids[19], 263);
    EXPECT_EQ(ids[20], 259);
}

/** @brief Tests that the merge learned first is applied first, from the left. */
TEST(Unit_TextCodec, MergesAreAppliedInTheOrderTheyWereLearned)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary doubled,
                            Vocabulary::make({Merge{97, 98}, Merge{257, 257}}));
    std::vector<TokenID> ids;
    ASSERT_TRUE(SaTest::status_is_ok(encode_text(doubled, "abababc", ids)));
    EXPECT_EQ(ids, (std::vector<TokenID>{258, 257, 99}));

    // (b, c) was learned before (a, b), so "abc" is a and bc, not ab and c.
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary ordered,
                            Vocabulary::make({Merge{98, 99}, Merge{97, 98}}));
    ids.clear();
    ASSERT_TRUE(SaTest::status_is_ok(encode_text(ordered, "abc ab", ids)));
    EXPECT_EQ(ids, (std::vector<TokenID>{97, 257, 32, 258}));
}

/** @brief Tests that a text encodes to the tokens its training left it in. */
TEST(Unit_TextCodec, ATrainedCorpusEncodesToItsTrainedTokens)
{
    BPETrainerOptions options;
    options.vocabulary_size = 300;
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, train_bpe("ab ab cd cd", options));

    std::vector<TokenID> ids;
    ASSERT_TRUE(SaTest::status_is_ok(encode_text(vocabulary, "ab ab cd cd", ids)));
    EXPECT_EQ(ids, (std::vector<TokenID>{258, 32, 258, 259, 259}));
}

/** @brief Tests that the encoder appends to the ids it is given. */
TEST(Unit_TextCodec, EncodeAppendsToTheIDsItIsGiven)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));

    std::vector<TokenID> ids = {7};
    ASSERT_TRUE(SaTest::status_is_ok(encode_text(vocabulary, "A", ids)));
    EXPECT_EQ(ids, (std::vector<TokenID>{7, 65}));
}

/** @brief Tests that a special token decodes to no byte. */
TEST(Unit_TextCodec, ASpecialTokenDecodesToNoByte)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));

    std::string text = ">";
    const std::vector<TokenID> ids = {72, K_END_OF_TEXT, 105};
    ASSERT_TRUE(SaTest::status_is_ok(decode_tokens(vocabulary, span_of(ids), text)));
    EXPECT_EQ(text, ">Hi");
}

/** @brief Tests that an id the vocabulary does not hold is a failure naming its index. */
TEST(Unit_TextCodec, DecodeRefusesAnIDOutsideTheVocabulary)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));

    std::string text;
    const std::vector<TokenID> ids = {72, 9999, 105};
    SA_EXPECT_FAILURE(decode_tokens(vocabulary, span_of(ids), text),
                      StatusCode::InvalidDescription, "decode_tokens", "9999", "index 1",
                      "[0, 257)");

    const std::vector<TokenID> negative = {-3};
    SA_EXPECT_FAILURE(decode_tokens(vocabulary, span_of(negative), text),
                      StatusCode::InvalidDescription, "-3", "index 0");
}
