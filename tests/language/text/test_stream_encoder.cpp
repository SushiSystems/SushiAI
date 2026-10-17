/**************************************************************************/
/* test_stream_encoder.cpp                                                */
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
 * @file test_stream_encoder.cpp
 * @brief Tests that encode_stream gives the ids of encode_text for every piece size.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/text/bpe_trainer.hpp>
#include <SushiAI/language/text/codec.hpp>
#include <SushiAI/language/text/stream_encoder.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Text;

namespace
{
    /** @brief Holds the vocabulary size the trainer is asked for. */
    constexpr std::int32_t K_VOCABULARY_SIZE = 300;

    /** @brief Returns a text with repeated words, runs of blanks and a word of 600 bytes. */
    std::string make_text()
    {
        std::string text;
        for (int round = 0; round < 40; ++round)
            text += "the cat sat on the mat\n  and the  cat\tsat again \r\n";

        text += std::string(600, 'x');
        text += " the end";
        return text;
    }

    /** @brief Expects the stream form to equal the whole-text form at one piece size. */
    void expect_same_ids(const Vocabulary& vocabulary, const std::string& text,
                         std::size_t piece_bytes)
    {
        std::vector<TokenID> whole;
        SA_EXPECT_OK(encode_text(vocabulary, text, whole));

        std::istringstream in(text);
        std::vector<TokenID> streamed;
        SA_EXPECT_OK(encode_stream(vocabulary, in, piece_bytes, streamed));

        EXPECT_EQ(streamed, whole) << "piece_bytes " << piece_bytes;
    }
} // namespace

/** @brief Tests pieces of 1, 7, 64 and 1000 bytes and the default against encode_text. */
TEST(Small_TextStreamEncoder, EveryPieceSizeGivesTheIDsOfTheWholeText)
{
    const std::string text = make_text();
    BPETrainerOptions options;
    options.vocabulary_size = K_VOCABULARY_SIZE;
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, train_bpe(text, options));

    for (const std::size_t piece_bytes :
         {std::size_t{1}, std::size_t{7}, std::size_t{64}, std::size_t{1000},
          K_STREAM_PIECE_BYTES})
        expect_same_ids(vocabulary, text, piece_bytes);
}

/** @brief Tests that an empty stream appends nothing and that the ids are appended. */
TEST(Small_TextStreamEncoder, AnEmptyStreamAppendsNothing)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({}));
    std::istringstream empty;
    std::istringstream word("ab");
    std::vector<TokenID> ids{7};

    SA_EXPECT_OK(encode_stream(vocabulary, empty, 16, ids));
    EXPECT_EQ(ids, (std::vector<TokenID>{7}));

    SA_EXPECT_OK(encode_stream(vocabulary, word, 16, ids));
    EXPECT_EQ(ids, (std::vector<TokenID>{7, 'a', 'b'}));
}
