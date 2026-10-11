/**************************************************************************/
/* test_text_vocabulary_file.cpp                                          */
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
 * @file test_text_vocabulary_file.cpp
 * @brief Tests the reader and the writer of the vocabulary file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/status.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>
#include <SushiAI/language/text/vocabulary_file.hpp>
#include <common/test_status.hpp>

#include "text_test_support.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::Text;

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the byte count of the file of one_merge(). */
    constexpr std::size_t K_ONE_MERGE_BYTES = 49;

    /**
     * @brief Returns the bytes of the version 1 file of end-of-text and the merge (97, 98).
     *
     * A header of 24 bytes, a special section of 17 (a length of 13 and the name) and one
     * merge of 8.
     */
    std::string one_merge()
    {
        std::string bytes = "SAVB";
        bytes.append(std::string("\x01\x00\x00\x00", 4));
        bytes.append(std::string("\x00\x01\x00\x00", 4));
        bytes.append(std::string("\x01\x00\x00\x00", 4));
        bytes.append(std::string("\x01\x00\x00\x00", 4));
        bytes.append(std::string("\x11\x00\x00\x00", 4));
        bytes.append(std::string("\x0D\x00\x00\x00", 4));
        bytes.append("<end_of_text>");
        bytes.append(std::string("\x61\x00\x00\x00", 4));
        bytes.append(std::string("\x62\x00\x00\x00", 4));
        return bytes;
    }
} // namespace

/** @brief Tests that the writer produces the bytes the module README documents. */
TEST(Unit_TextVocabularyFile, TheWriterProducesTheDocumentedBytes)
{
    const Support::ScratchDirectory scratch("sushiai_test_text_vocabulary_file_bytes");
    const std::filesystem::path file = scratch.path() / "tiny.savb";

    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({Merge{97, 98}}));
    ASSERT_TRUE(SaTest::status_is_ok(write_vocabulary_file(file, vocabulary)));

    const std::string written = Support::read_bytes(file);
    ASSERT_EQ(written.size(), K_ONE_MERGE_BYTES);
    EXPECT_EQ(written, one_merge());
}

/** @brief Tests that a vocabulary with special tokens and merges comes back from its file. */
TEST(Unit_TextVocabularyFile, AVocabularySurvivesItsFile)
{
    const Support::ScratchDirectory scratch("sushiai_test_text_vocabulary_file_round_trip");
    const std::filesystem::path file = scratch.path() / "vocabulary.savb";

    SA_ASSERT_OK_AND_ASSIGN(
        const Vocabulary vocabulary,
        Vocabulary::make({Merge{97, 98}, Merge{259, 99}, Merge{32, 260}}, {"<pad>", "<mask>"}));
    ASSERT_TRUE(SaTest::status_is_ok(write_vocabulary_file(file, vocabulary)));

    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary read, read_vocabulary_file(file));

    EXPECT_EQ(read.size(), vocabulary.size());
    EXPECT_EQ(read.special_names(), vocabulary.special_names());
    ASSERT_EQ(read.merges().size(), 3u);
    for (std::size_t i = 0; i < read.merges().size(); ++i)
        EXPECT_EQ(read.merges()[i], vocabulary.merges()[i]) << i;
    EXPECT_EQ(read.bytes_of(261), " abc");
}

/** @brief Tests that every proper prefix of a file is refused as a file that ends early. */
TEST(Unit_TextVocabularyFile, ATruncatedFileIsRefusedAtEveryLength)
{
    const std::string whole = one_merge();

    for (std::size_t length = 0; length < whole.size(); ++length)
    {
        SA_EXPECT_FAILURE(parse_vocabulary_file(std::string_view(whole.data(), length)),
                          StatusCode::InvalidFile, "read_vocabulary_file", "ends at byte")
            << length;
    }

    SA_EXPECT_FAILURE(parse_vocabulary_file(std::string_view(whole.data(), 10)),
                      StatusCode::InvalidFile, "ends at byte 10", "header of 24 bytes");
    SA_EXPECT_FAILURE(parse_vocabulary_file(std::string_view(whole.data(), 30)),
                      StatusCode::InvalidFile, "ends at byte 30", "declares 49 bytes");
}

/** @brief Tests that bytes after the declared end are refused, naming where they begin. */
TEST(Unit_TextVocabularyFile, BytesAfterTheDeclaredEndAreRefused)
{
    SA_EXPECT_FAILURE(parse_vocabulary_file(one_merge() + "xyz"), StatusCode::InvalidFile,
                      "3 bytes follow", "byte 49");
}

/** @brief Tests that a file is refused for another magic and for a version it cannot read. */
TEST(Unit_TextVocabularyFile, AnotherMagicOrVersionIsRefused)
{
    std::string bytes = one_merge();
    bytes[0] = 'X';
    SA_EXPECT_FAILURE(parse_vocabulary_file(bytes), StatusCode::InvalidFile, "'SAVB'",
                      "no vocabulary file");

    bytes = one_merge();
    bytes[4] = '\x02';
    SA_EXPECT_FAILURE(parse_vocabulary_file(bytes), StatusCode::InvalidFile, "byte 4",
                      "format version 2", "versions 1 to 1");

    bytes[4] = '\x00';
    SA_EXPECT_FAILURE(parse_vocabulary_file(bytes), StatusCode::InvalidFile,
                      "format version 0");
}

/** @brief Tests that a count of the header out of its bounds names the byte it is at. */
TEST(Unit_TextVocabularyFile, AHeaderCountOutOfBoundsNamesItsOffset)
{
    std::string bytes = one_merge();
    bytes[8] = '\x01';
    SA_EXPECT_FAILURE(parse_vocabulary_file(bytes), StatusCode::InvalidFile, "byte 8",
                      "257 byte tokens");

    bytes = one_merge();
    bytes[12] = '\x00';
    SA_EXPECT_FAILURE(parse_vocabulary_file(bytes), StatusCode::InvalidFile, "byte 12",
                      "0 special tokens");

    bytes = one_merge();
    bytes[19] = '\x7F';
    SA_EXPECT_FAILURE(parse_vocabulary_file(bytes), StatusCode::InvalidFile, "byte 16",
                      "merges");
}

/** @brief Tests that a special token record that leaves its section names its offset. */
TEST(Unit_TextVocabularyFile, ASpecialTokenPastItsSectionNamesItsOffset)
{
    std::string bytes = one_merge();
    bytes[24] = '\x0E';
    SA_EXPECT_FAILURE(parse_vocabulary_file(bytes), StatusCode::InvalidFile,
                      "special token 0", "byte 28", "14 bytes", "ends at byte 41");

    bytes = one_merge();
    bytes[24] = '\x0C';
    SA_EXPECT_FAILURE(parse_vocabulary_file(bytes), StatusCode::InvalidFile, "byte 28",
                      "'<end_of_text'");

    bytes = one_merge();
    bytes[24] = '\x00';
    SA_EXPECT_FAILURE(parse_vocabulary_file(bytes), StatusCode::InvalidFile, "byte 24",
                      "0 bytes");
}

/** @brief Tests that a merge the vocabulary refuses is a failure of the read. */
TEST(Unit_TextVocabularyFile, AMergeOfALaterTokenIsRefused)
{
    std::string bytes = one_merge();
    bytes[42] = '\x01';
    SA_EXPECT_FAILURE(parse_vocabulary_file(bytes), StatusCode::InvalidDescription, "merge 0",
                      "353 and 98");

    bytes = one_merge();
    bytes[44] = '\xFF';
    SA_EXPECT_FAILURE(parse_vocabulary_file(bytes), StatusCode::InvalidFile, "merge 0",
                      "byte 41");
}

/** @brief Tests that the reader names the path of a file that is absent or ends early. */
TEST(Unit_TextVocabularyFile, TheReaderNamesThePathOfAFileItRefuses)
{
    const Support::ScratchDirectory scratch("sushiai_test_text_vocabulary_file_refusals");

    SA_EXPECT_FAILURE(read_vocabulary_file(scratch.path() / "absent.savb"),
                      StatusCode::FileNotFound, "absent.savb", "read_vocabulary_file");

    const std::filesystem::path cut = scratch.write("cut.savb", one_merge().substr(0, 45));
    SA_EXPECT_FAILURE(read_vocabulary_file(cut), StatusCode::InvalidFile, "cut.savb",
                      "ends at byte 45", "declares 49 bytes");
}
