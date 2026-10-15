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
 * @file test_csv.cpp
 * @brief Tests the reader of a dataset kept as a header row and numeric columns.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/csv.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;

namespace
{
    /** @brief Returns options that read "train.csv" of a directory with the default columns. */
    CSVOptions options_of(const Support::ScratchDirectory& scratch)
    {
        CSVOptions options;
        options.root = scratch.path();
        return options;
    }
} // namespace

/** @brief Tests that every column but the last is a feature and the last is the target. */
TEST(Small_DatasetsCsv, TheLastColumnIsTheTargetByDefault)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_csv_default");
    scratch.write_text("train.csv", "a,b,y\n1,2,3\n-0.5,+4,1e1\n");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_csv(options_of(scratch)));

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
    const Support::ScratchDirectory scratch("sushiai_test_datasets_csv_named");
    scratch.write_text("data/set.csv", "id,x,y,z\nfirst,1,2,3\nsecond,4,5,6\n");

    CSVOptions options = options_of(scratch);
    options.file = "data/set.csv";
    options.features = {"z", "x"};
    options.targets = {"y", "z"};
    SA_EXPECT_FAILURE(read_csv(options), StatusCode::InvalidDescription, "set.csv", "read_csv",
                      "the column 'z' is named as a feature and as a target");

    options.targets = {"y"};
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_csv(options));

    EXPECT_EQ(list.table.columns, (std::vector<std::string>{"z", "x", "y"}));
    EXPECT_EQ(list.table.values, (std::vector<float>{3.0f, 1.0f, 2.0f, 6.0f, 4.0f, 5.0f}));
}

/** @brief Tests quoted fields, a doubled quote, a line break inside quotes, CRLF and a mark. */
TEST(Small_DatasetsCsv, QuotedFieldsAndCarriageReturnsAreRead)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_csv_quoted");
    scratch.write_text("train.csv", "\xEF\xBB\xBF\"the \"\"x\"\", one\",\"note\nlong\", y \r\n"
                                    "\" 1.5 \",\"a,\nb\",2\r\n"
                                    "\r\n"
                                    "3,,4");

    CSVOptions options = options_of(scratch);
    options.features = {"the \"x\", one"};
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_csv(options));

    EXPECT_EQ(list.table.columns, (std::vector<std::string>{"the \"x\", one", "y"}));
    EXPECT_EQ(list.table.values, (std::vector<float>{1.5f, 2.0f, 3.0f, 4.0f}));
}

/** @brief Tests that a cell that is no finite number is refused with its line and column. */
TEST(Small_DatasetsCsv, ANonNumericCellNamesItsLineAndColumn)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_csv_cell");

    scratch.write_text("train.csv", "a,b,y\n1,2,3\n4,five,6\n");
    SA_EXPECT_FAILURE(read_csv(options_of(scratch)), StatusCode::InvalidFile, "train.csv",
                      "line 3", "column 2 ('b')", "read_csv", "the cell 'five'");

    scratch.write_text("train.csv", "a,b,y\n1,2,\n");
    SA_EXPECT_FAILURE(read_csv(options_of(scratch)), StatusCode::InvalidFile, "line 2",
                      "column 3 ('y')", "the cell ''");

    // The first record spans lines 2 and 3, so the second starts on line 4.
    CSVOptions options = options_of(scratch);
    options.features = {"a"};
    scratch.write_text("train.csv", "note,a,y\n\"x\ny\",1,2\n\"z\",3,inf\n");
    SA_EXPECT_FAILURE(read_csv(options), StatusCode::InvalidFile, "line 4", "column 3 ('y')",
                      "the cell 'inf'");
}

/** @brief Tests that a record of another field count is refused with its line. */
TEST(Small_DatasetsCsv, ARecordOfAnotherFieldCountIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_csv_fields");
    scratch.write_text("train.csv", "a,b,y\n1,2,3\n\n4,5\n");

    SA_EXPECT_FAILURE(read_csv(options_of(scratch)), StatusCode::InvalidFile, "train.csv",
                      "line 4", "read_csv", "the record has 2 fields; the header has 3");
}

/** @brief Tests that a quote left open, and text after a closing quote, are refused. */
TEST(Small_DatasetsCsv, MalformedQuotesAreRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_csv_quotes");

    scratch.write_text("train.csv", "a,y\n1,\"2\n3,4\n");
    SA_EXPECT_FAILURE(read_csv(options_of(scratch)), StatusCode::InvalidFile, "line 2",
                      "column 2", "the quoted field is not closed");

    scratch.write_text("train.csv", "a,y\n\"1\"x,2\n");
    SA_EXPECT_FAILURE(read_csv(options_of(scratch)), StatusCode::InvalidFile, "line 2",
                      "column 1", "text follows the closing quote");

    scratch.write_text("train.csv", "a,y\n1,2\"\n");
    SA_EXPECT_FAILURE(read_csv(options_of(scratch)), StatusCode::InvalidFile, "line 2",
                      "column 2", "a quote stands inside a field");
}

/** @brief Tests that a column the header lacks is refused with the names it has. */
TEST(Small_DatasetsCsv, AColumnTheHeaderLacksIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_csv_columns");
    scratch.write_text("train.csv", "a,b,y\n1,2,3\n");

    CSVOptions options = options_of(scratch);
    options.targets = {"z"};
    SA_EXPECT_FAILURE(read_csv(options), StatusCode::InvalidDescription, "train.csv",
                      "no target column 'z'", "'a', 'b', 'y'");

    options.targets = {};
    options.features = {"c"};
    SA_EXPECT_FAILURE(read_csv(options), StatusCode::InvalidDescription,
                      "no feature column 'c'");
}

/** @brief Tests that a header name used twice, and a single column, are refused. */
TEST(Small_DatasetsCsv, AHeaderMustNameItsColumnsOnceAndLeaveAFeature)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_csv_header");

    scratch.write_text("train.csv", "\na, b ,a\n1,2,3\n");
    SA_EXPECT_FAILURE(read_csv(options_of(scratch)), StatusCode::InvalidFile, "line 2",
                      "column 3", "the name 'a' is also the name of column 1");

    scratch.write_text("train.csv", "y\n1\n");
    SA_EXPECT_FAILURE(read_csv(options_of(scratch)), StatusCode::InvalidDescription,
                      "no column is left as a feature");
}

/** @brief Tests that an absent file, an empty one and one with no record are refused. */
TEST(Small_DatasetsCsv, AFileWithoutRecordsIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_csv_empty");

    CSVOptions options = options_of(scratch);
    options.split = "val";
    SA_EXPECT_FAILURE(read_csv(options), StatusCode::FileNotFound, "val.csv", "read_csv");

    scratch.write_text("val.csv", "\r\n\r\n");
    SA_EXPECT_FAILURE(read_csv(options), StatusCode::InvalidFile, "val.csv",
                      "holds no header row");

    scratch.write_text("val.csv", "a,y\n\n");
    SA_EXPECT_FAILURE(read_csv(options), StatusCode::InvalidFile, "val.csv",
                      "holds no record below its header");
}
