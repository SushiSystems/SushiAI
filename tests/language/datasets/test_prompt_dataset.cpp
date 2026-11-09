/**************************************************************************/
/* test_prompt_dataset.cpp                                                */
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
 * @file test_prompt_dataset.cpp
 * @brief Tests PromptDataset's one batch and the "prompt" data block that encodes a text.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/prompt_dataset.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>
#include <SushiAI/language/text/vocabulary_file.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Datasets;
using Language::Text::TokenID;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine the batches are allocated on. */
    class Medium_LanguageDatasetsPromptDataset : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns the row and the stated length of the one batch. */
            void read_row(const Data::Dataset& dataset, std::vector<TokenID>& row,
                          std::int32_t& length)
            {
                Data::Batch batch;
                SA_EXPECT_OK(dataset.read(0, batch));

                const TokenID* const ids = batch.inputs.data_as<TokenID>();
                row.assign(ids, ids + batch.inputs.num_elements);

                const Data::BatchTarget* const stated =
                    Data::find_target(batch.targets, K_ROLE_PROMPT_LENGTH);
                ASSERT_NE(stated, nullptr);
                length = stated->values.data_as<std::int32_t>()[0];
            }
    };

    /** @brief Names a prompt and a window that do not fit, and what the refusal says. */
    struct PromptCase
    {
        const char* label;        /**< The name the row prints. */
        std::vector<TokenID> ids; /**< The ids of the prompt. */
        std::int64_t window;      /**< The window. */
        const char* message;      /**< A fragment the refusal names. */
    };

    /** @brief Provides the prompts and windows lay_out_prompt refuses. */
    class Small_LanguageDatasetsPromptDatasetLayout : public ::testing::TestWithParam<PromptCase>
    {

    };

    /** @brief Names a data block that does not fit, and what the refusal says. */
    struct BlockCase
    {
        const char* label;  /**< The name the row prints. */
        const char* json;   /**< The text of the data block. */
        const char* first;  /**< A fragment the refusal names. */
        const char* second; /**< A second fragment, or an empty text. */
    };

    /** @brief Provides the data blocks read_prompt_section refuses. */
    class Small_LanguageDatasetsPromptDatasetBlock : public ::testing::TestWithParam<BlockCase>
    {

    };

    /** @brief Names a row range and a batch index outside the one batch of one row. */
    struct RangeCase
    {
        const char* label;   /**< The name the row prints. */
        std::size_t index;   /**< The batch index. */
        Data::RowRange rows; /**< The rows asked for. */
        const char* first;   /**< A fragment the refusal names. */
        const char* second;  /**< A second fragment, or an empty text. */
        const char* third;   /**< A third fragment, or an empty text. */
    };

    /** @brief Provides the ranges the dataset refuses, on an engine for the dataset to hold. */
    class Medium_LanguageDatasetsPromptDatasetRange
        : public SaTest::EngineTest,
          public ::testing::WithParamInterface<RangeCase>
    {

    };

    /** @brief Returns the name a row prints. */
    template <typename Case>
    std::string label_of(const ::testing::TestParamInfo<Case>& info)
    {
        return std::string(info.param.label);
    }
} // namespace

/** @brief Tests that the dataset holds one batch of one row with the sizes, type and role asked. */
TEST_F(Medium_LanguageDatasetsPromptDataset, TheDatasetDescribesItsOneBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PromptDataset> dataset,
                            PromptDataset::make(*engine, {3, 1, 4, 1, 5}, 8));

    EXPECT_EQ(dataset->batch_size(), 1);
    EXPECT_EQ(dataset->batch_count(), 1u);
    EXPECT_EQ(dataset->sample_shape(), (Shape{8}));
    EXPECT_EQ(dataset->sample_dtype(), DType::Int32);
    EXPECT_EQ(dataset->name(), K_PROMPT_DATA_TYPE);
    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].role, K_ROLE_PROMPT_LENGTH);
}

/** @brief Tests that a read writes the laid-out row and its length into the batch tensors. */
TEST_F(Medium_LanguageDatasetsPromptDataset, AReadWritesTheRowAndItsLengthIntoTheBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PromptDataset> dataset,
                            PromptDataset::make(*engine, {3, 1, 4, 1, 5}, 8));

    std::vector<TokenID> row;
    std::int32_t length = 0;
    read_row(*dataset, row, length);

    EXPECT_EQ(row, (std::vector<TokenID>{3, 1, 4, 1, 5, 0, 0, 0}));
    EXPECT_EQ(length, 5);
}

/** @brief Tests that a short prompt fills the front of the row and zeros follow it. */
TEST(Small_LanguageDatasetsPromptDataset, AShortPromptIsFollowedByZeros)
{
    SA_ASSERT_OK_AND_ASSIGN(const PromptRow row, lay_out_prompt({3, 1, 4, 1, 5}, 8));

    EXPECT_EQ(row.ids, (std::vector<TokenID>{3, 1, 4, 1, 5, 0, 0, 0}));
    EXPECT_EQ(row.length, 5);
}

/** @brief Tests that a prompt longer than the window keeps its last ids. */
TEST(Small_LanguageDatasetsPromptDataset, ALongPromptKeepsItsLastIDs)
{
    SA_ASSERT_OK_AND_ASSIGN(const PromptRow row,
                            lay_out_prompt({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 4));

    EXPECT_EQ(row.ids, (std::vector<TokenID>{6, 7, 8, 9}));
    EXPECT_EQ(row.length, 4);
}

/** @brief Tests that no id, a negative id and a window of no position are each refused. */
TEST_P(Small_LanguageDatasetsPromptDatasetLayout, LayOutRefusesAPromptOrAWindowThatDoesNotFit)
{
    const PromptCase& row = GetParam();

    SA_EXPECT_FAILURE(lay_out_prompt(row.ids, row.window), StatusCode::InvalidDescription,
                      row.message);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LanguageDatasetsPromptDatasetLayout,
                         ::testing::Values(PromptCase{"NoId", {}, 4, "no id"},
                                           PromptCase{"ANegativeId", {1, -2}, 4, "-2"},
                                           PromptCase{"AWindowOfZero", {1}, 0, "window of 0"}),
                         label_of<PromptCase>);

/** @brief Tests that the data block encodes "text" with the vocabulary file it names. */
TEST_F(Medium_LanguageDatasetsPromptDataset, TheDataBlockEncodesItsText)
{
    const Support::ScratchDirectory scratch("sushiai_prompt_section");
    const std::filesystem::path file = scratch.path() / "bytes.savb";
    SA_ASSERT_OK_AND_ASSIGN(const Language::Text::Vocabulary vocabulary,
                            Language::Text::Vocabulary::make({}));
    SA_EXPECT_OK(Language::Text::write_vocabulary_file(file, vocabulary));

    const std::string stated = "\"vocabulary\": \"" + file.generic_string() + "\"";
    const Config::Value section = SaTest::require_value(
        Config::parse_json("{\"text\": \"hi\", \"window\": 4, " + stated + "}", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> dataset,
                            make_prompt_dataset(*engine, section, "the data block"));
    std::vector<TokenID> row;
    std::int32_t length = 0;
    read_row(*dataset, row, length);

    EXPECT_EQ(row, (std::vector<TokenID>{'h', 'i', 0, 0}));
    EXPECT_EQ(length, 2);
}

/** @brief Tests that the keys of a data block are read as the text, the file and the window. */
TEST(Small_LanguageDatasetsPromptDataset, TheKeysOfADataBlockAreRead)
{
    const Config::Value section = SaTest::require_value(Config::parse_json(
        "{\"text\": \"hi\", \"vocabulary\": \"bytes.savb\", \"window\": 4}", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(const PromptSection keys,
                            read_prompt_section(section, "the data block"));

    EXPECT_EQ(keys.text, "hi");
    EXPECT_EQ(keys.vocabulary, "bytes.savb");
    EXPECT_EQ(keys.window, 4);
}

/** @brief Tests that a data block without "text", or with an unknown key, is refused naming it. */
TEST_P(Small_LanguageDatasetsPromptDatasetBlock, AKeyThatDoesNotFitIsRefused)
{
    const BlockCase& row = GetParam();
    const Config::Value section = SaTest::require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(read_prompt_section(section, "the data block"),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageDatasetsPromptDatasetBlock,
    ::testing::Values(
        BlockCase{"NoText", "{\"window\": 4, \"vocabulary\": \"bytes.savb\"}", "the data block",
                  "has no 'text'"},
        BlockCase{"AnUnknownKey",
                  "{\"text\": \"hi\", \"window\": 4, \"tokens\": 9, \"vocabulary\": "
                  "\"bytes.savb\"}",
                  "'tokens'", ""}),
    label_of<BlockCase>);

/** @brief Tests that a range read of the one row holds what the whole read holds. */
TEST_F(Medium_LanguageDatasetsPromptDataset, ARangeReadHoldsTheRowOfTheWholeBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PromptDataset> dataset,
                            PromptDataset::make(*engine, {3, 1, 4, 1, 5}, 8));

    const Data::RowRange rows{0, 1};
    Data::Batch whole;
    Data::Batch part;
    SA_EXPECT_OK(dataset->read(0, whole));
    SA_EXPECT_OK(dataset->read_rows(0, rows, part));

    EXPECT_TRUE(Support::holds_rows_of(whole, dataset->batch_size(), rows, part));
}

/** @brief Tests that a range outside the prompt dataset is refused with a Status, not a stop. */
TEST_P(Medium_LanguageDatasetsPromptDatasetRange, RegressionABadRangeIsRefusedWithAStatus)
{
    const RangeCase& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PromptDataset> dataset,
                            PromptDataset::make(*engine, {3, 1, 4, 1, 5}, 8));
    Data::Batch batch;

    SA_EXPECT_FAILURE((dataset->read_rows(row.index, row.rows, batch)),
                      SushiAI::StatusCode::InvalidDescription, row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_LanguageDatasetsPromptDatasetRange,
    ::testing::Values(
        RangeCase{"RowsPastTheBatch", 0, Data::RowRange{0, 2}, "Rows 0 to 2", "of batch 0",
                  "a batch of 1"},
        RangeCase{"ABatchOutOfRange", 99, Data::RowRange{0, 1}, "Batch 99 is out of range", "",
                  ""}),
    label_of<RangeCase>);
