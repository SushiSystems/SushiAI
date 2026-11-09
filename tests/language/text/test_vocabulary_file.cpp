/**************************************************************************/
/* test_vocabulary_file.cpp                                               */
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
 * @file test_vocabulary_file.cpp
 * @brief Tests the serialiser, the reader and the writer of the vocabulary file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include "text_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>
#include <SushiAI/language/text/vocabulary_file.hpp>

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

    /** @brief Returns the file of one_merge() with the byte at @p offset set to @p value. */
    std::string with_byte(std::size_t offset, char value)
    {
        std::string bytes = one_merge();
        bytes[offset] = value;
        return bytes;
    }

    /** @brief Names a byte of the file to change and what the refusal says of it. */
    struct Corruption
    {
        std::size_t offset;                  /**< The byte to change. */
        char value;                          /**< The value it gets. */
        std::vector<std::string> needles;    /**< The texts the message holds. */
    };

    /** @brief Names a prefix of the file and what the refusal says of it. */
    struct Truncation
    {
        std::size_t length;                  /**< The bytes of the file kept. */
        std::vector<std::string> needles;    /**< The texts the message holds. */
    };

    /** @brief Returns the rows of a refused magic or version. */
    std::vector<Corruption> magic_and_version_rows()
    {
        return {{0, 'X', {"'SAVB'", "no vocabulary file"}},
                {4, '\x02', {"byte 4", "format version 2", "versions 1 to 1"}},
                {4, '\x00', {"format version 0"}}};
    }

    /** @brief Returns the rows of a count of the header out of its bounds. */
    std::vector<Corruption> header_count_rows()
    {
        return {{8, '\x01', {"byte 8", "257 byte tokens"}},
                {12, '\x00', {"byte 12", "0 special tokens"}},
                {19, '\x7F', {"byte 16", "merges"}}};
    }

    /** @brief Returns the rows of a special token record that leaves its section. */
    std::vector<Corruption> special_record_rows()
    {
        return {{24, '\x0E', {"special token 0", "byte 28", "14 bytes", "ends at byte 41"}},
                {24, '\x0C', {"byte 28", "'<end_of_text'"}},
                {24, '\x00', {"byte 24", "0 bytes"}}};
    }

    /** @brief Returns one row per proper prefix of the file of one_merge(). */
    std::vector<Truncation> truncation_rows()
    {
        std::vector<Truncation> rows;
        for (std::size_t length = 0; length < K_ONE_MERGE_BYTES; ++length)
        {
            const std::string where = "ends at byte " + std::to_string(length);
            const std::string what = length < K_VOCABULARY_FILE_HEADER_BYTES
                ? "header of 24 bytes"
                : "declares 49 bytes";
            rows.push_back(Truncation{length, {"read_vocabulary_file", where, what}});
        }

        return rows;
    }

    /** @brief Names a file that the reader refuses and what the refusal says of it. */
    struct RefusedFile
    {
        const char* name;                    /**< The file name under the scratch directory. */
        std::optional<std::size_t> kept;     /**< The bytes of one_merge() it holds; none: absent. */
        StatusCode code;                     /**< The code of the refusal. */
        std::vector<std::string> needles;    /**< The texts the message holds. */
    };

    /** @brief Returns the files the reader refuses. */
    std::vector<RefusedFile> refused_files()
    {
        return {{"absent.savb", std::nullopt, StatusCode::FileNotFound,
                 {"absent.savb", "read_vocabulary_file"}},
                {"cut.savb", 45, StatusCode::InvalidFile,
                 {"cut.savb", "ends at byte 45", "declares 49 bytes"}}};
    }

    /** @brief Provides the rows of a refused magic or version. */
    class Small_TextVocabularyFileMagic : public ::testing::TestWithParam<Corruption>
    {

    };

    /** @brief Provides the rows of the header counts. */
    class Small_TextVocabularyFileHeaderCount : public ::testing::TestWithParam<Corruption>
    {

    };

    /** @brief Provides the rows of the special records. */
    class Small_TextVocabularyFileSpecialRecord : public ::testing::TestWithParam<Corruption>
    {

    };

    /** @brief Provides the prefixes of the file. */
    class Small_TextVocabularyFileTruncation : public ::testing::TestWithParam<Truncation>
    {

    };

    /** @brief Provides the files the reader refuses. */
    class Medium_TextVocabularyFileRefusal : public ::testing::TestWithParam<RefusedFile>
    {

    };
} // namespace

/** @brief Tests that the serialiser produces the bytes the module README documents. */
TEST(Small_TextVocabularyFile, TheSerialiserProducesTheDocumentedBytes)
{
    SA_ASSERT_OK_AND_ASSIGN(const Vocabulary vocabulary, Vocabulary::make({Merge{97, 98}}));

    SA_ASSERT_OK_AND_ASSIGN(const std::string written, serialise_vocabulary_file(vocabulary));

    ASSERT_EQ(written.size(), K_ONE_MERGE_BYTES);
    EXPECT_EQ(written, one_merge());
}

/** @brief Tests that a vocabulary with special tokens and merges comes back from its file. */
TEST(Medium_TextVocabularyFile, AVocabularySurvivesItsFile)
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
    Support::expect_same_merges(read, vocabulary);
    EXPECT_EQ(read.bytes_of(261), " abc");
}

/** @brief Tests that every proper prefix of a file is refused as a file that ends early. */
TEST_P(Small_TextVocabularyFileTruncation, ATruncatedFileIsRefusedAtEveryLength)
{
    const Truncation& row = GetParam();
    const std::string whole = one_merge();

    Support::expect_failure_naming(
        parse_vocabulary_file(std::string_view(whole.data(), row.length)), StatusCode::InvalidFile,
        row.needles);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_TextVocabularyFileTruncation,
                         ::testing::ValuesIn(truncation_rows()));

/** @brief Tests that bytes after the declared end are refused, naming where they begin. */
TEST(Small_TextVocabularyFile, BytesAfterTheDeclaredEndAreRefused)
{
    SA_EXPECT_FAILURE(parse_vocabulary_file(one_merge() + "xyz"), StatusCode::InvalidFile,
                      "3 bytes follow", "byte 49");
}

/** @brief Tests that a file is refused for another magic and for a version it cannot read. */
TEST_P(Small_TextVocabularyFileMagic, AnotherMagicOrVersionIsRefused)
{
    const Corruption& row = GetParam();

    Support::expect_failure_naming(parse_vocabulary_file(with_byte(row.offset, row.value)),
                                   StatusCode::InvalidFile, row.needles);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_TextVocabularyFileMagic,
                         ::testing::ValuesIn(magic_and_version_rows()));

/** @brief Tests that a count of the header out of its bounds names the byte it is at. */
TEST_P(Small_TextVocabularyFileHeaderCount, AHeaderCountOutOfBoundsNamesItsOffset)
{
    const Corruption& row = GetParam();

    Support::expect_failure_naming(parse_vocabulary_file(with_byte(row.offset, row.value)),
                                   StatusCode::InvalidFile, row.needles);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_TextVocabularyFileHeaderCount,
                         ::testing::ValuesIn(header_count_rows()));

/** @brief Tests that a special token record that leaves its section names its offset. */
TEST_P(Small_TextVocabularyFileSpecialRecord, ASpecialTokenPastItsSectionNamesItsOffset)
{
    const Corruption& row = GetParam();

    Support::expect_failure_naming(parse_vocabulary_file(with_byte(row.offset, row.value)),
                                   StatusCode::InvalidFile, row.needles);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_TextVocabularyFileSpecialRecord,
                         ::testing::ValuesIn(special_record_rows()));

/** @brief Tests that a merge of a token the vocabulary does not hold yet is refused by it. */
TEST(Small_TextVocabularyFile, AMergeOfALaterTokenIsRefusedByTheVocabulary)
{
    SA_EXPECT_FAILURE(parse_vocabulary_file(with_byte(42, '\x01')),
                      StatusCode::InvalidDescription, "merge 0", "353 and 98");
}

/** @brief Tests that a merge id above what a token id holds is refused by the file structure. */
TEST(Small_TextVocabularyFile, AMergeIDAboveATokenIDIsRefusedByTheFile)
{
    SA_EXPECT_FAILURE(parse_vocabulary_file(with_byte(44, '\xFF')), StatusCode::InvalidFile,
                      "merge 0", "byte 41");
}

/** @brief Tests that the reader names the path of a file that is absent or ends early. */
TEST_P(Medium_TextVocabularyFileRefusal, TheReaderNamesThePathOfAFileItRefuses)
{
    const RefusedFile& row = GetParam();
    const Support::ScratchDirectory scratch("sushiai_test_text_vocabulary_file_refusals");
    const std::filesystem::path file = row.kept.has_value()
        ? scratch.write(row.name, one_merge().substr(0, *row.kept))
        : scratch.path() / row.name;

    Support::expect_failure_naming(read_vocabulary_file(file), row.code, row.needles);
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_TextVocabularyFileRefusal,
                         ::testing::ValuesIn(refused_files()));
