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

/**
 * @file test_token_file.cpp
 * @brief Tests the writer and the readers of the token file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/status.hpp>
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

    /** @brief Names one byte of a file and the value it is set to. */
    struct ByteEdit
    {
        std::size_t offset; /**< The byte, counted from the start of the file. */
        char value;         /**< The value the byte is set to. */
    };

    /** @brief Returns the bytes of three_ids() with the edits applied, in order. */
    std::string edited(const std::vector<ByteEdit>& edits)
    {
        std::string bytes = three_ids();
        for (const ByteEdit& edit : edits)
            bytes[edit.offset] = edit.value;

        return bytes;
    }

    /** @brief Reads the header of the bytes of a file through an in-memory stream. */
    Result<TokenFileHeader> header_of(const std::string& bytes)
    {
        std::istringstream in(bytes, std::ios::binary);
        return read_token_file_header(in, bytes.size());
    }

    /** @brief Names a write that is refused and what the refusal says. */
    struct WriteCase
    {
        const char* label;          /**< The name the row prints. */
        std::vector<TokenID> ids;   /**< The ids to write. */
        std::int32_t vocabulary;    /**< The vocabulary size to write. */
        const char* first;          /**< A fragment the refusal names. */
        const char* second;         /**< A second fragment, or an empty text. */
        const char* third;          /**< A third fragment, or an empty text. */
        const char* fourth;         /**< A fourth fragment, or an empty text. */
    };

    /** @brief Provides the writes the writer refuses. */
    class Small_LanguageDatasetsTokenFileWriterRefusal : public ::testing::TestWithParam<WriteCase>
    {

    };

    /** @brief Provides the byte counts at which the file is cut. */
    class Small_LanguageDatasetsTokenFileTruncation : public ::testing::TestWithParam<std::size_t>
    {

    };

    /** @brief Names a header field out of its bounds and what the refusal says. */
    struct HeaderCase
    {
        const char* label;              /**< The name the row prints. */
        std::vector<ByteEdit> edits;    /**< The edits that spoil the field. */
        const char* first;              /**< A fragment the refusal names. */
        const char* second;             /**< A second fragment, or an empty text. */
        const char* third;              /**< A third fragment, or an empty text. */
    };

    /** @brief Provides the header fields the reader refuses. */
    class Small_LanguageDatasetsTokenFileHeaderRefusal : public ::testing::TestWithParam<HeaderCase>
    {

    };

    /** @brief Names an id outside the vocabulary and where the reader finds it. */
    struct IdCase
    {
        const char* label;            /**< The name the row prints. */
        std::vector<ByteEdit> edits;  /**< The edits that put the id outside the vocabulary. */
        std::uint64_t first;          /**< The first id the range reads. */
        std::size_t count;            /**< How many ids the range reads. */
        const char* first_fragment;   /**< A fragment the refusal names. */
        const char* second_fragment;  /**< A second fragment, or an empty text. */
        const char* third_fragment;   /**< A third fragment, or an empty text. */
    };

    /** @brief Provides the ids outside their vocabulary. */
    class Small_LanguageDatasetsTokenFileIdRefusal : public ::testing::TestWithParam<IdCase>
    {

    };

    /** @brief Returns the name a row prints. */
    template <typename Case>
    std::string label_of(const ::testing::TestParamInfo<Case>& info)
    {
        return std::string(info.param.label);
    }
} // namespace

/** @brief Tests that the writer produces the bytes the module README documents. */
TEST(Small_LanguageDatasetsTokenFile, TheWriterProducesTheDocumentedBytes)
{
    std::ostringstream out(std::ios::binary);

    const std::vector<TokenID> ids = {1, 258, 0};
    ASSERT_TRUE(SaTest::status_is_ok(write_token_file(out, span_of(ids), 300)));

    EXPECT_EQ(out.str(), three_ids());
}

/** @brief Tests that the header fields come back from the bytes of a stream. */
TEST(Small_LanguageDatasetsTokenFile, TheHeaderSurvivesTheStream)
{
    std::ostringstream out(std::ios::binary);
    const std::vector<TokenID> ids = Support::rising_ids(40000);
    ASSERT_TRUE(SaTest::status_is_ok(write_token_file(out, span_of(ids), 40000)));

    SA_ASSERT_OK_AND_ASSIGN(const TokenFileHeader header, header_of(out.str()));

    EXPECT_EQ(header.vocabulary_size, 40000);
    EXPECT_EQ(header.token_count, 40000u);
}

/** @brief Tests that a range of ids comes back at its offset, as does the last and an empty one. */
TEST(Small_LanguageDatasetsTokenFile, ARangeOfIDsComesBackAtItsOffset)
{
    // 40000 ids are more than the 16384 the writer converts at once.
    std::ostringstream out(std::ios::binary);
    const std::vector<TokenID> ids = Support::rising_ids(40000);
    ASSERT_TRUE(SaTest::status_is_ok(write_token_file(out, span_of(ids), 40000)));
    const std::string bytes = out.str();
    SA_ASSERT_OK_AND_ASSIGN(const TokenFileHeader header, header_of(bytes));
    std::istringstream in(bytes, std::ios::binary);

    std::vector<TokenID> range(5);
    ASSERT_TRUE(SaTest::status_is_ok(read_token_range(in, header, 16382, span_of(range))));
    EXPECT_EQ(range, (std::vector<TokenID>{16382, 16383, 16384, 16385, 16386}));

    ASSERT_TRUE(SaTest::status_is_ok(read_token_range(in, header, 39995, span_of(range))));
    EXPECT_EQ(range.back(), 39999);

    std::vector<TokenID> none;
    EXPECT_TRUE(SaTest::status_is_ok(read_token_range(in, header, 40000, span_of(none))));
}

/** @brief Tests that an id outside the vocabulary, or a vocabulary of 0, is refused. */
TEST_P(Small_LanguageDatasetsTokenFileWriterRefusal, TheWriterRefusesAnIDOutsideTheVocabulary)
{
    const WriteCase& row = GetParam();
    std::ostringstream out(std::ios::binary);

    SA_EXPECT_FAILURE(write_token_file(out, span_of(row.ids), row.vocabulary),
                      StatusCode::InvalidDescription, row.first, row.second, row.third,
                      row.fourth);
    EXPECT_TRUE(out.str().empty());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageDatasetsTokenFileWriterRefusal,
    ::testing::Values(
        WriteCase{"AnIDAboveTheVocabulary", {1, 300}, 300, "write_token_file", "id 300",
                  "index 1", "[0, 300)"},
        WriteCase{"ANegativeID", {-2}, 300, "id -2", "index 0", "", ""},
        WriteCase{"AVocabularyOfZero", {1, 300}, 0, "vocabulary of 0 tokens", "", "", ""}),
    label_of<WriteCase>);

/** @brief Tests that a file cut at any length of the 36 bytes is refused, naming where it ends. */
TEST_P(Small_LanguageDatasetsTokenFileTruncation, ATruncatedFileIsRefusedAtEveryLength)
{
    const std::string cut = three_ids().substr(0, GetParam());

    SA_EXPECT_FAILURE(header_of(cut), StatusCode::InvalidFile, "read_token_file_header",
                      "ends at byte " + std::to_string(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LanguageDatasetsTokenFileTruncation,
                         ::testing::Range<std::size_t>(0, 36));

/** @brief Tests that a cut inside the header and a cut inside the ids name their cause. */
TEST(Small_LanguageDatasetsTokenFile, ACutNamesTheHeaderOrTheDeclaredBytes)
{
    SA_EXPECT_FAILURE(header_of(three_ids().substr(0, 10)), StatusCode::InvalidFile,
                      "ends at byte 10", "header of 24 bytes");
    SA_EXPECT_FAILURE(header_of(three_ids().substr(0, 30)), StatusCode::InvalidFile,
                      "ends at byte 30", "declares 36 bytes");
}

/** @brief Tests that bytes after the declared end are refused, naming where they begin. */
TEST(Small_LanguageDatasetsTokenFile, BytesAfterTheDeclaredEndAreRefused)
{
    SA_EXPECT_FAILURE(header_of(three_ids() + "xy"), StatusCode::InvalidFile, "2 bytes follow",
                      "byte 36");
}

/** @brief Tests that a header field out of its bounds names the byte it is at. */
TEST_P(Small_LanguageDatasetsTokenFileHeaderRefusal, AHeaderFieldOutOfBoundsNamesItsOffset)
{
    const HeaderCase& row = GetParam();

    SA_EXPECT_FAILURE(header_of(edited(row.edits)), StatusCode::InvalidFile, row.first,
                      row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageDatasetsTokenFileHeaderRefusal,
    ::testing::Values(
        HeaderCase{"TheMagic", {{0, 'X'}}, "'SATK'", "no token file", ""},
        HeaderCase{"TheVersion", {{4, '\x02'}}, "byte 4", "format version 2", "versions 1 to 1"},
        HeaderCase{"TheVocabulary", {{8, '\x00'}, {9, '\x00'}}, "byte 8",
                   "vocabulary of 0 tokens", ""},
        HeaderCase{"TheIdWidth", {{12, '\x08'}}, "byte 12", "ids of 8 bytes", ""},
        HeaderCase{"TheCount", {{23, '\x7F'}}, "byte 16", "ids", ""}),
    label_of<HeaderCase>);

/** @brief Tests that an id of the file outside its vocabulary names its byte offset. */
TEST_P(Small_LanguageDatasetsTokenFileIdRefusal, AnIDOutsideTheVocabularyNamesItsOffset)
{
    const IdCase& row = GetParam();
    const std::string bytes = edited(row.edits);
    SA_ASSERT_OK_AND_ASSIGN(const TokenFileHeader header, header_of(bytes));
    std::istringstream in(bytes, std::ios::binary);
    std::vector<TokenID> ids(row.count);

    SA_EXPECT_FAILURE(read_token_range(in, header, row.first, span_of(ids)),
                      StatusCode::InvalidFile, row.first_fragment, row.second_fragment,
                      row.third_fragment);
}

// The second id, 258 at byte 28, is outside a vocabulary of 258 tokens; -1 is at byte 32.
INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageDatasetsTokenFileIdRefusal,
    ::testing::Values(
        IdCase{"AnIDAboveTheVocabulary", {{8, '\x02'}}, 0, 3, "read_token_range", "id 258",
               "byte 28"},
        IdCase{"ANegativeID",
               {{32, '\xFF'}, {33, '\xFF'}, {34, '\xFF'}, {35, '\xFF'}}, 2, 1, "id -1",
               "byte 32", ""}),
    label_of<IdCase>);

/** @brief Tests that an id inside the vocabulary reads back as written. */
TEST(Small_LanguageDatasetsTokenFile, AValidIDReadsBackAsWritten)
{
    const std::string bytes = edited({{8, '\x02'}});
    SA_ASSERT_OK_AND_ASSIGN(const TokenFileHeader header, header_of(bytes));
    std::istringstream in(bytes, std::ios::binary);

    std::vector<TokenID> first(1);
    ASSERT_TRUE(SaTest::status_is_ok(read_token_range(in, header, 0, span_of(first))));

    EXPECT_EQ(first[0], 1);
}

/** @brief Tests that a range past the last id is refused. */
TEST(Small_LanguageDatasetsTokenFile, ARangeOutsideTheFileIsRefused)
{
    const std::string bytes = three_ids();
    SA_ASSERT_OK_AND_ASSIGN(const TokenFileHeader header, header_of(bytes));
    std::istringstream in(bytes, std::ios::binary);

    std::vector<TokenID> ids(2);
    SA_EXPECT_FAILURE(read_token_range(in, header, 2, span_of(ids)),
                      StatusCode::InvalidDescription, "2 ids from id 2", "file of 3");
}

/** @brief Tests that a file written to the disk comes back as its header and its ids. */
TEST(Medium_LanguageDatasetsTokenFile, AFileSurvivesTheDisk)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_file_round_trip");
    const std::filesystem::path file = scratch.path() / "tiny.satk";
    const std::vector<TokenID> ids = {1, 258, 0};
    ASSERT_TRUE(SaTest::status_is_ok(write_token_file(file, span_of(ids), 300)));

    EXPECT_EQ(Support::read_bytes(file), three_ids());
    SA_ASSERT_OK_AND_ASSIGN(const TokenFileHeader header, read_token_file_header(file));
    EXPECT_EQ(header.vocabulary_size, 300);
    EXPECT_EQ(header.token_count, 3u);

    std::vector<TokenID> range(2);
    ASSERT_TRUE(SaTest::status_is_ok(read_token_range(file, header, 1, span_of(range))));
    EXPECT_EQ(range, (std::vector<TokenID>{258, 0}));
}

/** @brief Tests that an id outside the vocabulary is refused before the file is opened. */
TEST(Medium_LanguageDatasetsTokenFile, TheWriterLeavesNoFileForARefusedID)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_file_bad_id");
    const std::filesystem::path file = scratch.path() / "refused.satk";

    const std::vector<TokenID> above = {1, 300};
    SA_EXPECT_FAILURE(write_token_file(file, span_of(above), 300),
                      StatusCode::InvalidDescription, "write_token_file", "id 300");

    EXPECT_FALSE(std::filesystem::exists(file));
}

/** @brief Tests that a refusal of the header reader opened on a file names the file. */
TEST(Medium_LanguageDatasetsTokenFile, ARefusalOfTheHeaderNamesTheFile)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_file_oversized");
    const std::filesystem::path file = scratch.write("long.satk", three_ids() + "xy");

    SA_EXPECT_FAILURE(read_token_file_header(file), StatusCode::InvalidFile, "long.satk",
                      "2 bytes follow", "byte 36");
}

/** @brief Tests that an absent file is refused, naming it. */
TEST(Medium_LanguageDatasetsTokenFile, AnAbsentFileIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_file_absent");
    const std::filesystem::path file = scratch.write("three.satk", three_ids());
    SA_ASSERT_OK_AND_ASSIGN(const TokenFileHeader header, read_token_file_header(file));
    std::vector<TokenID> ids(2);

    SA_EXPECT_FAILURE(read_token_file_header(scratch.path() / "absent.satk"),
                      StatusCode::FileNotFound, "absent.satk");
    SA_EXPECT_FAILURE(read_token_range(scratch.path() / "absent.satk", header, 0, span_of(ids)),
                      StatusCode::InvalidFile, "absent.satk", "cannot be opened");
}
