/**************************************************************************/
/* test_decision_file.cpp                                                 */
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
 * @file test_decision_file.cpp
 * @brief Tests the writer and the reader of the decision file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/decision_file.hpp>
#include <SushiAI/language/text/token.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Datasets;
using Language::Text::TokenID;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the vocabulary of the sample file. */
    constexpr std::int32_t K_SAMPLE_VOCABULARY = 20;

    /** @brief Returns a choice record. */
    DecisionRecord choice_of(std::vector<TokenID> text,
                             std::vector<std::vector<TokenID>> options, std::int32_t index)
    {
        DecisionRecord record;
        record.type = DecisionType::Choice;
        record.text = std::move(text);
        record.options = std::move(options);
        record.choice = index;
        return record;
    }

    /** @brief Returns a yes-or-no record. */
    DecisionRecord yes_no_of(std::vector<TokenID> text, bool yes)
    {
        DecisionRecord record;
        record.type = DecisionType::YesNo;
        record.text = std::move(text);
        record.yes = yes;
        return record;
    }

    /** @brief Returns a number record. */
    DecisionRecord number_of(std::vector<TokenID> text, float value)
    {
        DecisionRecord record;
        record.type = DecisionType::Number;
        record.text = std::move(text);
        record.number = value;
        return record;
    }

    /**
     * @brief Returns four records, one of each answer and a second yes-or-no.
     *
     * Their file is 148 bytes: a header of 20, then records of 64, 20, 24 and 20 bytes. The
     * first record's words start at byte 20, its option count is the word at byte 32; the
     * second starts at byte 84 and its answer is at byte 88; the third starts at byte 104 and
     * its answer is at byte 108.
     */
    std::vector<DecisionRecord> sample_records()
    {
        return {choice_of({1, 2, 3}, {{4, 5}, {6}, {7, 8, 9}}, 2), yes_no_of({10}, true),
                number_of({11, 12}, -2.5f), yes_no_of({3}, false)};
    }

    /** @brief Returns a yes-or-no record that also holds an option. */
    DecisionRecord yes_no_with_an_option()
    {
        DecisionRecord record = yes_no_of({1}, true);
        record.options = {{2}};
        return record;
    }

    /** @brief Returns a yes-or-no record that also states a number. */
    DecisionRecord yes_no_with_a_number()
    {
        DecisionRecord record = yes_no_of({1}, true);
        record.number = 1.0f;
        return record;
    }

    /** @brief Returns a number record that also states a yes. */
    DecisionRecord number_with_a_yes()
    {
        DecisionRecord record = number_of({1}, 1.0f);
        record.yes = true;
        return record;
    }

    /** @brief Returns the records as the span the writer reads. */
    SushiRuntime::span<const DecisionRecord> span_of(const std::vector<DecisionRecord>& records)
    {
        return SushiRuntime::span<const DecisionRecord>(records.data(), records.size());
    }

    /** @brief Writes records to memory, expects it to succeed, and returns the bytes. */
    std::string expect_bytes_of(const std::vector<DecisionRecord>& records, std::int32_t vocabulary)
    {
        std::ostringstream out(std::ios::binary);
        EXPECT_TRUE(SaTest::status_is_ok(write_decision_file(out, span_of(records), vocabulary)));
        return out.str();
    }

    /** @brief Returns the bytes of the sample file. */
    std::string sample_file()
    {
        return expect_bytes_of(sample_records(), K_SAMPLE_VOCABULARY);
    }

    /** @brief Reads a whole file held in memory. */
    Result<DecisionFile> read_bytes_of(const std::string& bytes)
    {
        std::istringstream in(bytes, std::ios::binary);
        return read_decision_file(in, bytes.size());
    }

    /** @brief Names one byte of a file and the value it is set to. */
    struct ByteEdit
    {
        std::size_t offset; /**< The byte, counted from the start of the file. */
        char value;         /**< The value the byte is set to. */
    };

    /** @brief Returns the sample file with the edits applied, in order, and a suffix appended. */
    std::string edited(const std::vector<ByteEdit>& edits, const std::string& suffix = "")
    {
        std::string bytes = sample_file();
        for (const ByteEdit& edit : edits)
            bytes[edit.offset] = edit.value;

        return bytes + suffix;
    }

    /** @brief Names a write that is refused and what the refusal says. */
    struct WriteCase
    {
        const char* label;                    /**< The name the row prints. */
        std::vector<DecisionRecord> records;  /**< The records to write. */
        std::int32_t vocabulary;              /**< The vocabulary size to write. */
        const char* first;                    /**< A fragment the refusal names. */
        const char* second;                   /**< A second fragment, or an empty text. */
    };

    /** @brief Provides the writes the writer refuses. */
    class Small_LanguageDatasetsDecisionFileWriterRefusal
        : public ::testing::TestWithParam<WriteCase>
    {

    };

    /** @brief Provides the byte counts at which the sample file is cut. */
    class Small_LanguageDatasetsDecisionFileTruncation
        : public ::testing::TestWithParam<std::size_t>
    {

    };

    /** @brief Names a spoiled file and what the refusal says. */
    struct FileCase
    {
        const char* label;              /**< The name the row prints. */
        std::vector<ByteEdit> edits;    /**< The edits that spoil the file. */
        const char* suffix;             /**< The bytes appended after the edits. */
        const char* first;              /**< A fragment the refusal names. */
        const char* second;             /**< A second fragment, or an empty text. */
    };

    /** @brief Provides the files the reader refuses. */
    class Small_LanguageDatasetsDecisionFileRefusal : public ::testing::TestWithParam<FileCase>
    {

    };

    /** @brief Returns the name a row prints. */
    template <typename Case>
    std::string label_of(const ::testing::TestParamInfo<Case>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.label);
    }
} // namespace

/** @brief Tests that the writer produces the bytes the module README documents. */
TEST(Small_LanguageDatasetsDecisionFile, TheWriterProducesTheDocumentedBytes)
{
    // Header: SADC, version 1, vocabulary 10, one record; record: type 1, answer 1, 1 id, 0 options, id 5.
    std::string expected = "SADC";
    expected.append(std::string("\x01\x00\x00\x00", 4));
    expected.append(std::string("\x0A\x00\x00\x00", 4));
    expected.append(std::string("\x01\x00\x00\x00\x00\x00\x00\x00", 8));
    expected.append(std::string("\x01\x00\x00\x00", 4));
    expected.append(std::string("\x01\x00\x00\x00", 4));
    expected.append(std::string("\x01\x00\x00\x00", 4));
    expected.append(std::string("\x00\x00\x00\x00", 4));
    expected.append(std::string("\x05\x00\x00\x00", 4));

    EXPECT_EQ(expect_bytes_of({yes_no_of({5}, true)}, 10), expected);
}

/** @brief Tests that a number is written as the little-endian bits of its float. */
TEST(Small_LanguageDatasetsDecisionFile, ANumberIsWrittenAsTheBitsOfItsFloat)
{
    // 1.0f is 0x3F800000; the answer word of the first record is at byte 24.
    const std::string bytes = expect_bytes_of({number_of({0}, 1.0f)}, 10);

    EXPECT_EQ(bytes.substr(24, 4), std::string("\x00\x00\x80\x3F", 4));
}

/** @brief Tests that a file reads back equal to the records and the vocabulary written. */
TEST(Small_LanguageDatasetsDecisionFile, AWrittenFileReadsBackEqual)
{
    SA_ASSERT_OK_AND_ASSIGN(const DecisionFile file, read_bytes_of(sample_file()));

    EXPECT_EQ(file.vocabulary_size, K_SAMPLE_VOCABULARY);
    EXPECT_EQ(file.records, sample_records());
}

/** @brief Tests that a file of no record reads back as no record. */
TEST(Small_LanguageDatasetsDecisionFile, AFileOfNoRecordReadsBackEmpty)
{
    SA_ASSERT_OK_AND_ASSIGN(const DecisionFile file, read_bytes_of(expect_bytes_of({}, 7)));

    EXPECT_EQ(file.vocabulary_size, 7);
    EXPECT_TRUE(file.records.empty());
}

/** @brief Tests that the extreme floats and ids a record may hold come back as written. */
TEST(Small_LanguageDatasetsDecisionFile, TheExtremesOfAFloatAndAnIDReadBackAsWritten)
{
    const std::vector<DecisionRecord> records = {
        number_of({0}, std::numeric_limits<float>::max()),
        number_of({0}, std::numeric_limits<float>::denorm_min()),
        number_of({19}, -1.0f)};

    SA_ASSERT_OK_AND_ASSIGN(const DecisionFile file,
                            read_bytes_of(expect_bytes_of(records, K_SAMPLE_VOCABULARY)));

    EXPECT_EQ(file.records, records);
}

/** @brief Tests that a record that breaks a rule of its type is refused, and nothing is written. */
TEST_P(Small_LanguageDatasetsDecisionFileWriterRefusal, TheWriterRefusesARecordThatBreaksARule)
{
    const WriteCase& row = GetParam();
    std::ostringstream out(std::ios::binary);

    SA_EXPECT_FAILURE(write_decision_file(out, span_of(row.records), row.vocabulary),
                      StatusCode::InvalidDescription, row.first, row.second);
    EXPECT_TRUE(out.str().empty());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageDatasetsDecisionFileWriterRefusal,
    ::testing::Values(
        WriteCase{"AnEmptyText", {yes_no_of({}, true)}, 20, "text of record 0", "holds 0 ids"},
        WriteCase{"AnIDAboveTheVocabulary", {yes_no_of({1}, true), yes_no_of({20}, true)}, 20,
                  "id 20", "record 1"},
        WriteCase{"ANegativeID", {yes_no_of({-1}, true)}, 20, "id -1", "record 0"},
        WriteCase{"AVocabularyOfZero", {yes_no_of({0}, true)}, 0, "vocabulary of 0 tokens", ""},
        WriteCase{"AChoiceWithNoOption", {choice_of({1}, {}, 0)}, 20, "record 0", "0 options"},
        WriteCase{"AnOptionOfNoID", {choice_of({1}, {{2}, {}}, 0)}, 20, "option of record 0",
                  "holds 0 ids"},
        WriteCase{"AnIndexPastTheOptions", {choice_of({1}, {{2}, {3}}, 2)}, 20,
                  "answers option 2 of 2", ""},
        WriteCase{"ANegativeIndex", {choice_of({1}, {{2}}, -1)}, 20, "answers option -1", ""},
        WriteCase{"OptionsOfAYesNo", {yes_no_with_an_option()}, 20, "record 0", "1 options"},
        WriteCase{"ANumberThatIsNotFinite",
                  {number_of({1}, std::numeric_limits<float>::quiet_NaN())}, 20,
                  "number record 0", "not finite"},
        WriteCase{"AnInfiniteNumber",
                  {number_of({1}, std::numeric_limits<float>::infinity())}, 20,
                  "number record 0", "not finite"},
        WriteCase{"ANumberOnAYesNo", {yes_no_with_a_number()}, 20, "yes-or-no record 0",
                  "another type"},
        WriteCase{"AYesOnANumber", {number_with_a_yes()}, 20, "number record 0", "another type"}),
    label_of<WriteCase>);

/** @brief Tests that a file cut at any length of the 148 bytes is refused. */
TEST_P(Small_LanguageDatasetsDecisionFileTruncation, ATruncatedFileIsRefusedAtEveryLength)
{
    SA_EXPECT_FAILURE(read_bytes_of(sample_file().substr(0, GetParam())),
                      StatusCode::InvalidFile, "read_decision_file");
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LanguageDatasetsDecisionFileTruncation,
                         ::testing::Range<std::size_t>(0, 148));

/** @brief Tests that a cut inside the header names the header. */
TEST(Small_LanguageDatasetsDecisionFile, ACutInsideTheHeaderNamesTheHeader)
{
    SA_EXPECT_FAILURE(read_bytes_of(sample_file().substr(0, 10)), StatusCode::InvalidFile,
                      "ends at byte 10", "header of 20 bytes");
}

/** @brief Tests that a field out of its bounds, or bytes after the end, name their offset. */
TEST_P(Small_LanguageDatasetsDecisionFileRefusal, AFieldOutOfItsBoundsNamesItsOffset)
{
    const FileCase& row = GetParam();

    SA_EXPECT_FAILURE(read_bytes_of(edited(row.edits, row.suffix)), StatusCode::InvalidFile,
                      row.first, row.second);
}

// Bytes 108 to 111 hold the bits 0x7FC00000, a NaN, as the number of the third record.
INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageDatasetsDecisionFileRefusal,
    ::testing::Values(
        FileCase{"TheMagic", {{0, 'X'}}, "", "'SADC'", "no decision file"},
        FileCase{"TheVersion", {{4, '\x02'}}, "", "byte 4", "format version 2"},
        FileCase{"TheVocabulary", {{8, '\x00'}}, "", "byte 8", "vocabulary of 0 tokens"},
        FileCase{"TheCount", {{19, '\x7F'}}, "", "byte 12", "records"},
        FileCase{"TheTypeOfARecord", {{20, '\x03'}}, "", "byte 20", "record type 3"},
        FileCase{"TheAnswerOfAYesNo", {{88, '\x02'}}, "", "byte 88", "yes-or-no answer 2"},
        FileCase{"TheOptionCount", {{35, '\x7F'}}, "", "byte 32", "options"},
        FileCase{"ANumberThatIsNotFinite",
                 {{108, '\x00'}, {109, '\x00'}, {110, '\xC0'}, {111, '\x7F'}}, "",
                 "number record 2", "not finite"},
        FileCase{"AnIDOutsideTheVocabulary", {{8, '\x03'}}, "", "id 3", "record 0"},
        FileCase{"BytesAfterTheLastRecord", {}, "xy", "2 bytes follow", "byte 148"}),
    label_of<FileCase>);

/** @brief Tests that a file survives the disk and comes back equal. */
TEST(Medium_LanguageDatasetsDecisionFile, AFileSurvivesTheDisk)
{
    const Support::ScratchDirectory scratch("decision_file_disk");
    const std::filesystem::path file = scratch.path() / "set.sadc";
    const std::vector<DecisionRecord> records = sample_records();

    ASSERT_TRUE(SaTest::status_is_ok(
        write_decision_file(file, span_of(records), K_SAMPLE_VOCABULARY)));
    SA_ASSERT_OK_AND_ASSIGN(const DecisionFile read, read_decision_file(file));

    EXPECT_EQ(read.vocabulary_size, K_SAMPLE_VOCABULARY);
    EXPECT_EQ(read.records, records);
    EXPECT_EQ(Support::read_bytes(file), sample_file());
}

/** @brief Tests that a refused record leaves no file behind. */
TEST(Medium_LanguageDatasetsDecisionFile, TheWriterLeavesNoFileForARefusedRecord)
{
    const Support::ScratchDirectory scratch("decision_file_refused");
    const std::filesystem::path file = scratch.path() / "refused.sadc";
    const std::vector<DecisionRecord> records = {yes_no_of({}, true)};

    SA_EXPECT_FAILURE(write_decision_file(file, span_of(records), K_SAMPLE_VOCABULARY),
                      StatusCode::InvalidDescription, "text of record 0");
    EXPECT_FALSE(std::filesystem::exists(file));
}

/** @brief Tests that a refusal of a file on the disk names the file. */
TEST(Medium_LanguageDatasetsDecisionFile, ARefusalNamesTheFile)
{
    const Support::ScratchDirectory scratch("decision_file_named");
    const std::filesystem::path file = scratch.write("long.sadc", sample_file() + "xy");

    SA_EXPECT_FAILURE(read_decision_file(file), StatusCode::InvalidFile, "long.sadc",
                      "2 bytes follow");
}

/** @brief Tests that a path that names nothing is a missing file. */
TEST(Medium_LanguageDatasetsDecisionFile, AnAbsentFileIsRefused)
{
    const Support::ScratchDirectory scratch("decision_file_absent");

    SA_EXPECT_FAILURE(read_decision_file(scratch.path() / "absent.sadc"),
                      StatusCode::FileNotFound, "absent.sadc");
}
