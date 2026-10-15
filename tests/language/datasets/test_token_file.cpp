/**************************************************************************/
/* test_token_file.cpp                                                    */
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
 * @file test_token_file.cpp
 * @brief Tests the writer and the readers of the token file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/token_file.hpp>
#include <SushiAI/language/text/token.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Datasets;
using Language::Text::TokenID;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Returns the ids as the span write_token_file reads. */
    SushiRuntime::span<const TokenID> span_of(const std::vector<TokenID>& ids)
    {
        return SushiRuntime::span<const TokenID>(ids.data(), ids.size());
    }

    /** @brief Returns the ids as the span read_token_range writes. */
    SushiRuntime::span<TokenID> span_of(std::vector<TokenID>& ids)
    {
        return SushiRuntime::span<TokenID>(ids.data(), ids.size());
    }

    /**
     * @brief Returns the 36 bytes of the version 1 file of the ids 1, 258 and 0 over 300 tokens.
     *
     * A header of 24 bytes: the magic, version 1, 300, an id width of 4 and a count of 3.
     */
    std::string three_ids()
    {
        std::string bytes = "SATK";
        bytes.append(std::string("\x01\x00\x00\x00", 4));
        bytes.append(std::string("\x2C\x01\x00\x00", 4));
        bytes.append(std::string("\x04\x00\x00\x00", 4));
        bytes.append(std::string("\x03\x00\x00\x00\x00\x00\x00\x00", 8));
        bytes.append(std::string("\x01\x00\x00\x00", 4));
        bytes.append(std::string("\x02\x01\x00\x00", 4));
        bytes.append(std::string("\x00\x00\x00\x00", 4));
        return bytes;
    }
} // namespace

/** @brief Tests that the writer produces the bytes the module README documents. */
TEST(Small_LanguageDatasetsTokenFile, TheWriterProducesTheDocumentedBytes)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_file_bytes");
    const std::filesystem::path file = scratch.path() / "tiny.satk";

    const std::vector<TokenID> ids = {1, 258, 0};
    ASSERT_TRUE(SaTest::status_is_ok(write_token_file(file, span_of(ids), 300)));

    EXPECT_EQ(Support::read_bytes(file), three_ids());
}

/** @brief Tests that the header and any range of the ids come back from the file. */
TEST(Small_LanguageDatasetsTokenFile, TheHeaderAndARangeOfIDsSurviveTheFile)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_file_round_trip");
    const std::filesystem::path file = scratch.path() / "corpus.satk";

    // 40000 ids are more than the 16384 the writer converts at once.
    const std::vector<TokenID> ids = Support::rising_ids(40000);
    ASSERT_TRUE(SaTest::status_is_ok(write_token_file(file, span_of(ids), 40000)));

    SA_ASSERT_OK_AND_ASSIGN(const TokenFileHeader header, read_token_file_header(file));
    EXPECT_EQ(header.vocabulary_size, 40000);
    EXPECT_EQ(header.token_count, 40000u);

    std::vector<TokenID> range(5);
    ASSERT_TRUE(SaTest::status_is_ok(read_token_range(file, header, 16382, span_of(range))));
    EXPECT_EQ(range, (std::vector<TokenID>{16382, 16383, 16384, 16385, 16386}));

    ASSERT_TRUE(SaTest::status_is_ok(read_token_range(file, header, 39995, span_of(range))));
    EXPECT_EQ(range.back(), 39999);

    std::vector<TokenID> none;
    EXPECT_TRUE(SaTest::status_is_ok(read_token_range(file, header, 40000, span_of(none))));
}

/** @brief Tests that an id outside the vocabulary is refused before the file is opened. */
TEST(Small_LanguageDatasetsTokenFile, TheWriterRefusesAnIDOutsideTheVocabulary)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_file_bad_id");
    const std::filesystem::path file = scratch.path() / "refused.satk";

    const std::vector<TokenID> above = {1, 300};
    SA_EXPECT_FAILURE(write_token_file(file, span_of(above), 300),
                      StatusCode::InvalidDescription, "write_token_file", "id 300", "index 1",
                      "[0, 300)");

    const std::vector<TokenID> negative = {-2};
    SA_EXPECT_FAILURE(write_token_file(file, span_of(negative), 300),
                      StatusCode::InvalidDescription, "id -2", "index 0");
    SA_EXPECT_FAILURE(write_token_file(file, span_of(above), 0),
                      StatusCode::InvalidDescription, "vocabulary of 0 tokens");

    EXPECT_FALSE(std::filesystem::exists(file));
}

/** @brief Tests that a file cut at any length is refused, naming the byte it ends at. */
TEST(Small_LanguageDatasetsTokenFile, ATruncatedFileIsRefusedAtEveryLength)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_file_truncated");
    const std::string whole = three_ids();

    for (std::size_t length = 0; length < whole.size(); ++length)
    {
        const std::filesystem::path cut = scratch.write("cut.satk", whole.substr(0, length));
        SA_EXPECT_FAILURE(read_token_file_header(cut), StatusCode::InvalidFile, "cut.satk",
                          "read_token_file_header", "ends at byte")
            << length;
    }

    const std::filesystem::path in_header = scratch.write("cut.satk", whole.substr(0, 10));
    SA_EXPECT_FAILURE(read_token_file_header(in_header), StatusCode::InvalidFile,
                      "ends at byte 10", "header of 24 bytes");

    const std::filesystem::path in_ids = scratch.write("cut.satk", whole.substr(0, 30));
    SA_EXPECT_FAILURE(read_token_file_header(in_ids), StatusCode::InvalidFile,
                      "ends at byte 30", "declares 36 bytes");
}

/** @brief Tests that bytes after the declared end are refused, naming where they begin. */
TEST(Small_LanguageDatasetsTokenFile, BytesAfterTheDeclaredEndAreRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_file_oversized");
    const std::filesystem::path file = scratch.write("long.satk", three_ids() + "xy");

    SA_EXPECT_FAILURE(read_token_file_header(file), StatusCode::InvalidFile, "long.satk",
                      "2 bytes follow", "byte 36");
}

/** @brief Tests that a header field out of its bounds names the byte it is at. */
TEST(Small_LanguageDatasetsTokenFile, AHeaderFieldOutOfBoundsNamesItsOffset)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_file_header");

    std::string bytes = three_ids();
    bytes[0] = 'X';
    SA_EXPECT_FAILURE(read_token_file_header(scratch.write("magic.satk", bytes)),
                      StatusCode::InvalidFile, "'SATK'", "no token file");

    bytes = three_ids();
    bytes[4] = '\x02';
    SA_EXPECT_FAILURE(read_token_file_header(scratch.write("version.satk", bytes)),
                      StatusCode::InvalidFile, "byte 4", "format version 2", "versions 1 to 1");

    bytes = three_ids();
    bytes[8] = '\x00';
    bytes[9] = '\x00';
    SA_EXPECT_FAILURE(read_token_file_header(scratch.write("vocabulary.satk", bytes)),
                      StatusCode::InvalidFile, "byte 8", "vocabulary of 0 tokens");

    bytes = three_ids();
    bytes[12] = '\x08';
    SA_EXPECT_FAILURE(read_token_file_header(scratch.write("width.satk", bytes)),
                      StatusCode::InvalidFile, "byte 12", "ids of 8 bytes");

    bytes = three_ids();
    bytes[23] = '\x7F';
    SA_EXPECT_FAILURE(read_token_file_header(scratch.write("count.satk", bytes)),
                      StatusCode::InvalidFile, "byte 16", "ids");
}

/** @brief Tests that an id of the file outside its vocabulary names its byte offset. */
TEST(Small_LanguageDatasetsTokenFile, AnIDOutsideTheVocabularyNamesItsOffset)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_file_range");

    // The second id, 258 at byte 28, is outside a vocabulary of 258 tokens.
    std::string bytes = three_ids();
    bytes[8] = '\x02';
    const std::filesystem::path above = scratch.write("above.satk", bytes);
    SA_ASSERT_OK_AND_ASSIGN(const TokenFileHeader header, read_token_file_header(above));

    std::vector<TokenID> ids(3);
    SA_EXPECT_FAILURE(read_token_range(above, header, 0, span_of(ids)), StatusCode::InvalidFile,
                      "above.satk", "read_token_range", "id 258", "byte 28", "[0, 258)");

    std::vector<TokenID> first(1);
    ASSERT_TRUE(SaTest::status_is_ok(read_token_range(above, header, 0, span_of(first))));
    EXPECT_EQ(first[0], 1);

    bytes = three_ids();
    bytes.replace(32, 4, std::string("\xFF\xFF\xFF\xFF", 4));
    const std::filesystem::path negative = scratch.write("negative.satk", bytes);
    SA_EXPECT_FAILURE(read_token_range(negative, header, 2, span_of(first)),
                      StatusCode::InvalidFile, "id -1", "byte 32");
}

/** @brief Tests that a range past the last id, and an absent file, are refused. */
TEST(Small_LanguageDatasetsTokenFile, ARangeOutsideTheFileAndAnAbsentFileAreRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_file_outside");
    const std::filesystem::path file = scratch.write("three.satk", three_ids());
    SA_ASSERT_OK_AND_ASSIGN(const TokenFileHeader header, read_token_file_header(file));

    std::vector<TokenID> ids(2);
    SA_EXPECT_FAILURE(read_token_range(file, header, 2, span_of(ids)),
                      StatusCode::InvalidDescription, "2 ids from id 2", "file of 3");

    SA_EXPECT_FAILURE(read_token_file_header(scratch.path() / "absent.satk"),
                      StatusCode::FileNotFound, "absent.satk");
    SA_EXPECT_FAILURE(read_token_range(scratch.path() / "absent.satk", header, 0, span_of(ids)),
                      StatusCode::InvalidFile, "absent.satk", "cannot be opened");
}
