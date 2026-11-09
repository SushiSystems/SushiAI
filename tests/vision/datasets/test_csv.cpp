/**************************************************************************/
/* test_csv.cpp                                                           */
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
 * @file test_csv.cpp
 * @brief Tests the reader of a dataset kept as a header row and numeric columns.
 * @author Mustafa Garip
 */
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/csv.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;

namespace
{
    /** @brief Returns options that read "train.csv" of the folder data with the default columns. */
    CSVOptions default_options()
    {
        CSVOptions options;
        options.root = "data";
        return options;
    }

    /** @brief Returns a source holding the text as the file path. */
    MemorySource source_of(const char* path, const std::string& text)
    {
        MemorySource source;
        source.add_file(path, text);
        return source;
    }

    /** @brief Returns a source holding the text as data/train.csv. */
    MemorySource train_file(const std::string& text)
    {
        return source_of("data/train.csv", text);
    }

    /** @brief Holds a file with a cell that is no finite number and where the message points. */
    struct CellRow
    {
        std::string name;                   /**< The row's name. */
        std::string text;                   /**< The whole file. */
        std::vector<std::string> features;  /**< The feature columns; empty for the default. */
        std::string line;                   /**< The line the message names. */
        std::string column;                 /**< The column the message names. */
        std::string cell;                   /**< The cell the message quotes. */
    };

    /** @brief Holds a file with a malformed quote and where the message points. */
    struct QuoteRow
    {
        std::string name;    /**< The row's name. */
        std::string text;    /**< The whole file. */
        std::string line;    /**< The line the message names. */
        std::string column;  /**< The column the message names. */
        std::string cause;   /**< The cause the message states. */
    };

    /** @brief Runs read_csv over each file with a non-numeric cell. */
    class Small_DatasetsCsvCell : public ::testing::TestWithParam<CellRow>
    {
    };

    /** @brief Runs read_csv over each file with a malformed quote. */
    class Small_DatasetsCsvQuote : public ::testing::TestWithParam<QuoteRow>
    {
    };

    /** @brief Names a test after its cell row. */
    std::string cell_row_name(const ::testing::TestParamInfo<CellRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a test after its quote row. */
    std::string quote_row_name(const ::testing::TestParamInfo<QuoteRow>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that every column but the last is a feature and the last is the target. */
TEST(Small_DatasetsCsv, TheLastColumnIsTheTargetByDefault)
{
    const MemorySource source = train_file("a,b,y\n1,2,3\n-0.5,+4,1e1\n");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_csv(default_options(), source));

    ASSERT_EQ(list.files.size(), 1u);
    EXPECT_EQ(list.files[0].filename().generic_string(), "train.csv");
    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(list.samples[1].file, 0u);
    EXPECT_EQ(list.class_count, 0);

    // The cells are short decimals a float holds exactly, so they compare exactly.
    EXPECT_EQ(list.table.feature_count, 2);
    EXPECT_EQ(list.table.target_count, 1);
    EXPECT_EQ(list.table.columns, (std::vector<std::string>{"a", "b", "y"}));
    EXPECT_EQ(list.table.values, (std::vector<float>{1.0f, 2.0f, 3.0f, -0.5f, 4.0f, 10.0f}));
    EXPECT_EQ(list.table.row_count(), 2u);
    EXPECT_EQ(list.table.features(1)[1], 4.0f);
    EXPECT_EQ(list.table.targets(1)[0], 10.0f);
}

/** @brief Tests that named columns are taken in the order named and the others are not read. */
TEST(Small_DatasetsCsv, NamedColumnsAreTakenInTheOrderNamed)
{
    const MemorySource source =
        source_of("data/data/set.csv", "id,x,y,z\nfirst,1,2,3\nsecond,4,5,6\n");
    CSVOptions options = default_options();
    options.file = "data/set.csv";
    options.features = {"z", "x"};
    options.targets = {"y"};

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_csv(options, source));

    EXPECT_EQ(list.table.columns, (std::vector<std::string>{"z", "x", "y"}));
    EXPECT_EQ(list.table.values, (std::vector<float>{3.0f, 1.0f, 2.0f, 6.0f, 4.0f, 5.0f}));
}

/** @brief Tests that a column named as a feature and as a target is refused. */
TEST(Small_DatasetsCsv, AColumnNamedAsFeatureAndTargetIsRefused)
{
    const MemorySource source =
        source_of("data/data/set.csv", "id,x,y,z\nfirst,1,2,3\nsecond,4,5,6\n");
    CSVOptions options = default_options();
    options.file = "data/set.csv";
    options.features = {"z", "x"};
    options.targets = {"y", "z"};

    SA_EXPECT_FAILURE(read_csv(options, source), StatusCode::InvalidDescription, "set.csv",
                      "read_csv", "the column 'z' is named as a feature and as a target");
}

/** @brief Tests that quoted fields, a doubled quote and a line break inside quotes are read. */
TEST(Small_DatasetsCsv, QuotedFieldsKeepTheirQuotesAndLineBreaks)
{
    const MemorySource source = train_file("\"the \"\"x\"\", one\",\"note\nlong\", y \n"
                                           "\" 1.5 \",\"a,\nb\",2\n"
                                           "3,,4");
    CSVOptions options = default_options();
    options.features = {"the \"x\", one"};

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_csv(options, source));

    EXPECT_EQ(list.table.columns, (std::vector<std::string>{"the \"x\", one", "y"}));
    EXPECT_EQ(list.table.values, (std::vector<float>{1.5f, 2.0f, 3.0f, 4.0f}));
}

/** @brief Tests that carriage returns end a record and blank lines are skipped. */
TEST(Small_DatasetsCsv, CarriageReturnsEndRecordsAndBlankLinesAreSkipped)
{
    const MemorySource source = train_file("a,y\r\n1,2\r\n\r\n3,4");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_csv(default_options(), source));

    EXPECT_EQ(list.table.columns, (std::vector<std::string>{"a", "y"}));
    EXPECT_EQ(list.table.values, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

/** @brief Tests that a leading byte-order mark is skipped. */
TEST(Small_DatasetsCsv, ALeadingByteOrderMarkIsSkipped)
{
    const MemorySource source = train_file("\xEF\xBB\xBF" "a,y\n1,2\n");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_csv(default_options(), source));

    EXPECT_EQ(list.table.columns, (std::vector<std::string>{"a", "y"}));
}

/** @brief Tests that a cell that is no finite number is refused with its line and column. */
TEST_P(Small_DatasetsCsvCell, ANonNumericCellNamesItsLineAndColumn)
{
    const MemorySource source = train_file(GetParam().text);
    CSVOptions options = default_options();
    options.features = GetParam().features;

    SA_EXPECT_FAILURE(read_csv(options, source), StatusCode::InvalidFile, "train.csv",
                      GetParam().line, GetParam().column, "read_csv", GetParam().cell);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsCsvCell,
    ::testing::Values(
        CellRow{"Word", "a,b,y\n1,2,3\n4,five,6\n", {}, "line 3", "column 2 ('b')",
                "the cell 'five'"},
        CellRow{"Empty", "a,b,y\n1,2,\n", {}, "line 2", "column 3 ('y')", "the cell ''"},
        // The first record spans lines 2 and 3, so the second starts on line 4.
        CellRow{"Infinity", "note,a,y\n\"x\ny\",1,2\n\"z\",3,inf\n", {"a"}, "line 4",
                "column 3 ('y')", "the cell 'inf'"}),
    cell_row_name);

/** @brief Tests that a record of another field count is refused with its line. */
TEST(Small_DatasetsCsv, ARecordOfAnotherFieldCountIsRefused)
{
    const MemorySource source = train_file("a,b,y\n1,2,3\n\n4,5\n");

    SA_EXPECT_FAILURE(read_csv(default_options(), source), StatusCode::InvalidFile, "train.csv",
                      "line 4", "read_csv", "the record has 2 fields; the header has 3");
}

/** @brief Tests that a quote left open, text after a closing quote and a stray quote are refused. */
TEST_P(Small_DatasetsCsvQuote, MalformedQuotesAreRefused)
{
    const MemorySource source = train_file(GetParam().text);

    SA_EXPECT_FAILURE(read_csv(default_options(), source), StatusCode::InvalidFile,
                      GetParam().line, GetParam().column, GetParam().cause);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsCsvQuote,
    ::testing::Values(
        QuoteRow{"NotClosed", "a,y\n1,\"2\n3,4\n", "line 2", "column 2",
                 "the quoted field is not closed"},
        QuoteRow{"TextAfterTheClosingQuote", "a,y\n\"1\"x,2\n", "line 2", "column 1",
                 "text follows the closing quote"},
        QuoteRow{"QuoteInsideAField", "a,y\n1,2\"\n", "line 2", "column 2",
                 "a quote stands inside a field"}),
    quote_row_name);

/** @brief Tests that a target column the header lacks is refused with the names it has. */
TEST(Small_DatasetsCsv, ATargetColumnTheHeaderLacksIsRefused)
{
    const MemorySource source = train_file("a,b,y\n1,2,3\n");
    CSVOptions options = default_options();
    options.targets = {"z"};

    SA_EXPECT_FAILURE(read_csv(options, source), StatusCode::InvalidDescription, "train.csv",
                      "no target column 'z'", "'a', 'b', 'y'");
}

/** @brief Tests that a feature column the header lacks is refused. */
TEST(Small_DatasetsCsv, AFeatureColumnTheHeaderLacksIsRefused)
{
    const MemorySource source = train_file("a,b,y\n1,2,3\n");
    CSVOptions options = default_options();
    options.features = {"c"};

    SA_EXPECT_FAILURE(read_csv(options, source), StatusCode::InvalidDescription,
                      "no feature column 'c'");
}

/** @brief Tests that a header name used twice is refused. */
TEST(Small_DatasetsCsv, AHeaderMustNameItsColumnsOnce)
{
    const MemorySource source = train_file("\na, b ,a\n1,2,3\n");

    SA_EXPECT_FAILURE(read_csv(default_options(), source), StatusCode::InvalidFile, "line 2",
                      "column 3", "the name 'a' is also the name of column 1");
}

/** @brief Tests that a header of one column, which leaves no feature, is refused. */
TEST(Small_DatasetsCsv, AHeaderMustLeaveAFeature)
{
    const MemorySource source = train_file("y\n1\n");

    SA_EXPECT_FAILURE(read_csv(default_options(), source), StatusCode::InvalidDescription,
                      "no column is left as a feature");
}

/** @brief Tests that an absent file is reported with its name. */
TEST(Small_DatasetsCsv, AnAbsentFileIsNotFound)
{
    const MemorySource source;
    CSVOptions options = default_options();
    options.split = "val";

    SA_EXPECT_FAILURE(read_csv(options, source), StatusCode::FileNotFound, "val.csv", "read_csv");
}

/** @brief Tests that a file with no header row is refused. */
TEST(Small_DatasetsCsv, AFileWithoutAHeaderIsRefused)
{
    const MemorySource source = train_file("\r\n\r\n");

    SA_EXPECT_FAILURE(read_csv(default_options(), source), StatusCode::InvalidFile, "train.csv",
                      "holds no header row");
}

/** @brief Tests that a file with a header and no record is refused. */
TEST(Small_DatasetsCsv, AFileWithoutARecordIsRefused)
{
    const MemorySource source = train_file("a,y\n\n");

    SA_EXPECT_FAILURE(read_csv(default_options(), source), StatusCode::InvalidFile, "train.csv",
                      "holds no record below its header");
}

/** @brief Tests that the file-system form reads a file written to disk into the same table. */
TEST(Medium_DatasetsCsv, AFileOnDiskReadsIntoATable)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_csv_disk");
    scratch.write_text("train.csv", "a,b,y\n1,2,3\n4,5,6\n");
    CSVOptions options;
    options.root = scratch.path();

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_csv(options));

    EXPECT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(list.table.values, (std::vector<float>{1, 2, 3, 4, 5, 6}));
}
