/**************************************************************************/
/* test_decision_dataset.cpp                                              */
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
 * @file test_decision_dataset.cpp
 * @brief Tests the row layout, the dataset and the "decisions" data block of the decision set.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/decision_dataset.hpp>
#include <SushiAI/language/datasets/decision_file.hpp>
#include <SushiAI/language/text/token.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Datasets;
using Language::Text::TokenID;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the vocabulary of the records these tests use. */
    constexpr std::int32_t K_VOCABULARY = 20;

    /** @brief Holds T, the ids of a padded text and option. */
    constexpr std::int64_t K_WINDOW = 4;

    /** @brief Holds K, the options of a padded choice. */
    constexpr std::int64_t K_OPTIONS = 3;

    /** @brief Returns the extents the rows are padded to. */
    DecisionShape row_shape()
    {
        DecisionShape shape;
        shape.window = K_WINDOW;
        shape.options = K_OPTIONS;
        return shape;
    }

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

    /** @brief Returns the four records of the batches: a choice, a yes, a number and a no. */
    std::vector<DecisionRecord> four_records()
    {
        return {choice_of({1, 2, 3}, {{4, 5}, {6}}, 1), yes_no_of({7}, true),
                number_of({8, 9, 10, 11}, 2.5f), yes_no_of({3, 3}, false)};
    }

    /** @brief Returns the records as the span the layout reads. */
    SushiRuntime::span<const DecisionRecord> span_of(const std::vector<DecisionRecord>& records)
    {
        return SushiRuntime::span<const DecisionRecord>(records.data(), records.size());
    }

    /** @brief Returns the first @p count elements of a tensor as a vector. */
    template <typename Element>
    std::vector<Element> elements_of(const SushiBLAS::Tensor& tensor, std::size_t count)
    {
        const Element* const data = tensor.data_as<Element>();
        return std::vector<Element>(data, data + count);
    }

    /** @brief Provides an engine and the decision file of four records. */
    class Medium_LanguageDatasetsDecisionDataset : public SaTest::EngineTest
    {
        protected:
            /** @brief Writes the four records and returns the path of the file. */
            std::filesystem::path four_records_file(const Support::ScratchDirectory& scratch)
            {
                const std::filesystem::path file = scratch.path() / "four.sadc";
                const std::vector<DecisionRecord> records = four_records();
                SA_EXPECT_OK(write_decision_file(file, span_of(records), K_VOCABULARY));
                return file;
            }

            /** @brief Returns the options of two records a batch, T = 4 and K = 3. */
            static DecisionDatasetOptions pair_options()
            {
                DecisionDatasetOptions options;
                options.batch_size = 2;
                options.shape = row_shape();
                return options;
            }
    };

    /** @brief Names a record or an extent that does not fit and what the refusal says. */
    struct FitCase
    {
        const char* label;                    /**< The name the row prints. */
        std::vector<DecisionRecord> records;  /**< The records to lay out. */
        std::int64_t window;                  /**< T. */
        std::int64_t options;                 /**< K. */
        const char* first;                    /**< A fragment the refusal names. */
        const char* second;                   /**< A second fragment, or an empty text. */
    };

    /** @brief Provides the layouts lay_out_decisions refuses. */
    class Small_LanguageDatasetsDecisionDatasetFit : public ::testing::TestWithParam<FitCase>
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

    /** @brief Provides the data blocks read_decisions_section refuses. */
    class Small_LanguageDatasetsDecisionDatasetBlock : public ::testing::TestWithParam<BlockCase>
    {

    };

    /** @brief Names the options of a dataset that cannot be made, and what the refusal says. */
    struct RefusalCase
    {
        const char* label;       /**< The name the row prints. */
        std::int64_t batch_size; /**< The records of a batch. */
        std::int64_t window;     /**< T. */
        std::int64_t options;    /**< K. */
        bool absent_file;        /**< Whether the file named does not exist. */
        StatusCode code;         /**< The failure's code. */
        const char* first;       /**< A fragment the refusal names. */
        const char* second;      /**< A second fragment, or an empty text. */
    };

    /** @brief Provides the options make refuses, over a file of four records. */
    class Medium_LanguageDatasetsDecisionDatasetRefusal
        : public Medium_LanguageDatasetsDecisionDataset,
          public ::testing::WithParamInterface<RefusalCase>
    {

    };

    /** @brief Returns the name a row prints. */
    template <typename Case>
    std::string label_of(const ::testing::TestParamInfo<Case>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.label);
    }
} // namespace

/** @brief Tests that the rows of a choice, a yes and a number are laid out role by role. */
TEST(Small_LanguageDatasetsDecisionDataset, TheRowsOfThreeTypesAreLaidOutAsDocumented)
{
    // A choice of two options answering the second, a yes, and a number of four ids.
    const std::vector<DecisionRecord> records = {choice_of({1, 2, 3}, {{4, 5}, {6}}, 1),
                                                 yes_no_of({7}, true),
                                                 number_of({8, 9, 10, 11}, 2.5f)};

    SA_ASSERT_OK_AND_ASSIGN(const DecisionRows rows,
                            lay_out_decisions(span_of(records), row_shape()));

    EXPECT_EQ(rows.tokens, (std::vector<std::int32_t>{1, 2, 3, 0, 7, 0, 0, 0, 8, 9, 10, 11}));
    EXPECT_EQ(rows.lengths, (std::vector<float>{3.0f, 1.0f, 4.0f}));
    EXPECT_EQ(rows.options, (std::vector<std::int32_t>{4, 5, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
                                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(rows.option_lengths,
              (std::vector<float>{2.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    EXPECT_EQ(rows.option_valid,
              (std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    EXPECT_EQ(rows.choice, (std::vector<std::int32_t>{1, 0, 0}));
    EXPECT_EQ(rows.yes_no, (std::vector<float>{0.0f, 1.0f, 0.0f}));
    EXPECT_EQ(rows.number, (std::vector<float>{0.0f, 0.0f, 2.5f}));
    EXPECT_EQ(rows.type_mask,
              (std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}));
}

/** @brief Tests that a text and an option of exactly T ids, and K options, fit. */
TEST(Small_LanguageDatasetsDecisionDataset, ARecordThatFillsTheExtentsFits)
{
    const std::vector<DecisionRecord> records = {
        choice_of({1, 2, 3, 4}, {{1, 2, 3, 4}, {5}, {6}}, 2)};

    SA_ASSERT_OK_AND_ASSIGN(const DecisionRows rows,
                            lay_out_decisions(span_of(records), row_shape()));

    EXPECT_EQ(rows.lengths, (std::vector<float>{4.0f}));
    EXPECT_EQ(rows.option_lengths, (std::vector<float>{4.0f, 1.0f, 1.0f}));
    EXPECT_EQ(rows.option_valid, (std::vector<float>{1.0f, 1.0f, 1.0f}));
    EXPECT_EQ(rows.choice, (std::vector<std::int32_t>{2}));
}

/** @brief Tests that no record lays out as arrays of no element. */
TEST(Small_LanguageDatasetsDecisionDataset, NoRecordLaysOutAsNoRow)
{
    SA_ASSERT_OK_AND_ASSIGN(const DecisionRows rows, lay_out_decisions({}, row_shape()));

    EXPECT_TRUE(rows.tokens.empty());
    EXPECT_TRUE(rows.options.empty());
    EXPECT_TRUE(rows.type_mask.empty());
}

/** @brief Tests that a record or an extent that does not fit is refused naming what it is. */
TEST_P(Small_LanguageDatasetsDecisionDatasetFit, LayOutRefusesWhatDoesNotFit)
{
    const FitCase& row = GetParam();
    DecisionShape shape;
    shape.window = row.window;
    shape.options = row.options;

    SA_EXPECT_FAILURE(lay_out_decisions(span_of(row.records), shape),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageDatasetsDecisionDatasetFit,
    ::testing::Values(
        FitCase{"ATextLongerThanTheWindow", {yes_no_of({1}, true), yes_no_of({1, 2, 3}, true)},
                2, 3, "text of record 1", "holds 3 ids"},
        FitCase{"AnOptionLongerThanTheWindow", {choice_of({1}, {{1}, {1, 2, 3}}, 0)}, 2, 3,
                "option of record 0", "holds 3 ids"},
        FitCase{"MoreOptionsThanTheExtent", {choice_of({1}, {{1}, {2}, {3}}, 0)}, 2, 2,
                "record 0 holds 3 options", "a row holds 2"},
        FitCase{"AWindowOfZero", {yes_no_of({1}, true)}, 0, 3, "window of 0", ""},
        FitCase{"NoOptions", {yes_no_of({1}, true)}, 2, 0, "0 options", ""}),
    label_of<FitCase>);

/** @brief Tests that the keys of a data block are read as the file, T, K and the batch size. */
TEST(Small_LanguageDatasetsDecisionDataset, TheKeysOfADataBlockAreRead)
{
    const Config::Value section = SaTest::require_value(Config::parse_json(
        "{\"file\": \"set.sadc\", \"window\": 128, \"options\": 8, \"batch_size\": 32}",
        "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(const DecisionsSection keys,
                            read_decisions_section(section, "the data block"));

    EXPECT_EQ(keys.file, "set.sadc");
    EXPECT_EQ(keys.options.shape.window, 128);
    EXPECT_EQ(keys.options.shape.options, 8);
    EXPECT_EQ(keys.options.batch_size, 32);
}

/** @brief Tests that an absent batch size keeps the default of the options. */
TEST(Small_LanguageDatasetsDecisionDataset, AnAbsentBatchSizeKeepsTheDefault)
{
    const Config::Value section = SaTest::require_value(
        Config::parse_json("{\"file\": \"set.sadc\", \"window\": 128, \"options\": 8}", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(const DecisionsSection keys,
                            read_decisions_section(section, "the data block"));

    EXPECT_EQ(keys.options.batch_size, DecisionDatasetOptions().batch_size);
}

/** @brief Tests that a data block with a key absent, unknown or not positive is refused. */
TEST_P(Small_LanguageDatasetsDecisionDatasetBlock, AKeyThatDoesNotFitIsRefused)
{
    const BlockCase& row = GetParam();
    const Config::Value section = SaTest::require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(read_decisions_section(section, "the data block"),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageDatasetsDecisionDatasetBlock,
    ::testing::Values(
        BlockCase{"NoFile", "{\"window\": 4, \"options\": 3}", "the data block",
                  "has no 'file'"},
        BlockCase{"NoWindow", "{\"file\": \"a.sadc\", \"options\": 3}", "the data block",
                  "has no 'window'"},
        BlockCase{"NoOptions", "{\"file\": \"a.sadc\", \"window\": 4}", "the data block",
                  "has no 'options'"},
        BlockCase{"AnUnknownKey",
                  "{\"file\": \"a.sadc\", \"window\": 4, \"options\": 3, \"stride\": 1}",
                  "'stride'", ""},
        BlockCase{"OptionsOfZero", "{\"file\": \"a.sadc\", \"window\": 4, \"options\": 0}",
                  "options", "must be positive"}),
    label_of<BlockCase>);

/** @brief Tests that the dataset states its input, its vocabulary, its batches and its name. */
TEST_F(Medium_LanguageDatasetsDecisionDataset, TheDatasetDescribesItsBatches)
{
    const Support::ScratchDirectory scratch("decision_dataset_describe");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<DecisionDataset> dataset,
        DecisionDataset::make(*engine, four_records_file(scratch), pair_options()));

    EXPECT_EQ(dataset->batch_size(), 2);
    EXPECT_EQ(dataset->batch_count(), 2u);
    EXPECT_EQ(dataset->sample_shape(), (Shape{K_WINDOW}));
    EXPECT_EQ(dataset->sample_dtype(), DType::Int32);
    EXPECT_EQ(dataset->vocabulary_size(), K_VOCABULARY);
    EXPECT_EQ(dataset->name(), "decisions");
}

/** @brief Tests that an option of no id is flagged as padding, as an absent option is. */
TEST(Small_LanguageDatasetsDecisionDataset, AnOptionOfNoIdIsFlaggedAsPadding)
{
    const std::vector<DecisionRecord> records = {choice_of({1}, {{4}, {}, {5, 6}}, 2)};

    SA_ASSERT_OK_AND_ASSIGN(const DecisionRows rows,
                            lay_out_decisions(span_of(records), row_shape()));

    EXPECT_EQ(rows.option_lengths, (std::vector<float>{1.0f, 0.0f, 2.0f}));
    EXPECT_EQ(rows.option_valid, (std::vector<float>{1.0f, 0.0f, 1.0f}));
}

/** @brief Tests that the eight roles are stated with their per-sample shapes and types. */
TEST_F(Medium_LanguageDatasetsDecisionDataset, TheEightRolesAreDescribedInOrder)
{
    const Support::ScratchDirectory scratch("decision_dataset_roles");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<DecisionDataset> dataset,
        DecisionDataset::make(*engine, four_records_file(scratch), pair_options()));
    const std::vector<Data::TargetDescription>& roles = dataset->targets();

    ASSERT_EQ(roles.size(), 8u);
    EXPECT_EQ(roles[0].role, K_ROLE_LENGTHS);
    EXPECT_EQ(roles[0].sample_shape, (Shape{1}));
    EXPECT_EQ(roles[0].dtype, DType::Float32);
    EXPECT_EQ(roles[1].role, K_ROLE_OPTIONS);
    EXPECT_EQ(roles[1].sample_shape, (Shape{K_OPTIONS, K_WINDOW}));
    EXPECT_EQ(roles[1].dtype, DType::Int32);
    EXPECT_EQ(roles[2].role, K_ROLE_OPTION_LENGTHS);
    EXPECT_EQ(roles[2].sample_shape, (Shape{K_OPTIONS}));
    EXPECT_EQ(roles[2].dtype, DType::Float32);
    EXPECT_EQ(roles[3].role, K_ROLE_OPTION_VALID);
    EXPECT_EQ(roles[3].sample_shape, (Shape{K_OPTIONS}));
    EXPECT_EQ(roles[3].dtype, DType::Float32);
    EXPECT_EQ(roles[4].role, K_ROLE_CHOICE);
    EXPECT_EQ(roles[4].sample_shape, (Shape{}));
    EXPECT_EQ(roles[4].dtype, DType::Int32);
    EXPECT_EQ(roles[4].classes, K_OPTIONS);
    EXPECT_EQ(roles[5].role, K_ROLE_YES_NO);
    EXPECT_EQ(roles[5].sample_shape, (Shape{1}));
    EXPECT_EQ(roles[5].dtype, DType::Float32);
    EXPECT_EQ(roles[6].role, K_ROLE_NUMBER);
    EXPECT_EQ(roles[6].sample_shape, (Shape{1}));
    EXPECT_EQ(roles[6].dtype, DType::Float32);
    EXPECT_EQ(roles[7].role, K_ROLE_TYPE_MASK);
    EXPECT_EQ(roles[7].sample_shape, (Shape{K_DECISION_TYPES}));
    EXPECT_EQ(roles[7].dtype, DType::Float32);
}

/** @brief Tests that the roles the dataset replaced are not published any more. */
TEST_F(Medium_LanguageDatasetsDecisionDataset, TheRolesLabelsAndValuesAreNotPublished)
{
    const Support::ScratchDirectory scratch("decision_dataset_no_labels");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<DecisionDataset> dataset,
        DecisionDataset::make(*engine, four_records_file(scratch), pair_options()));

    EXPECT_EQ(Data::find_target(dataset->targets(), Data::K_ROLE_LABELS), nullptr);
    EXPECT_EQ(Data::find_target(dataset->targets(), Data::K_ROLE_VALUES), nullptr);
}

/** @brief Tests that a batch holds the input and the eight roles with the stated shapes. */
TEST_F(Medium_LanguageDatasetsDecisionDataset, ABatchHasTheEightRolesWithTheirShapes)
{
    const Support::ScratchDirectory scratch("decision_dataset_shapes");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<DecisionDataset> dataset,
        DecisionDataset::make(*engine, four_records_file(scratch), pair_options()));
    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    // N = 2 records, T = 4, K = 3.
    ASSERT_EQ(batch.inputs.dtype, DType::Int32);
    EXPECT_EQ(batch.inputs.num_elements, 2 * 4);
    ASSERT_EQ(batch.targets.size(), 8u);
    EXPECT_EQ(batch.targets[0].values.num_elements, 2);
    EXPECT_EQ(batch.targets[1].values.num_elements, 2 * 3 * 4);
    EXPECT_EQ(batch.targets[2].values.num_elements, 2 * 3);
    EXPECT_EQ(batch.targets[3].values.num_elements, 2 * 3);
    EXPECT_EQ(batch.targets[4].values.num_elements, 2);
    EXPECT_EQ(batch.targets[5].values.num_elements, 2);
    EXPECT_EQ(batch.targets[6].values.num_elements, 2);
    EXPECT_EQ(batch.targets[7].values.num_elements, 2 * 3);
    ASSERT_EQ(batch.targets[0].values.rank, 2);
    EXPECT_EQ(batch.targets[0].values.shape[0], 2);
    EXPECT_EQ(batch.targets[0].values.shape[1], 1);
    ASSERT_EQ(batch.targets[1].values.rank, 3);
    EXPECT_EQ(batch.targets[1].values.shape[0], 2);
    EXPECT_EQ(batch.targets[1].values.shape[1], 3);
    EXPECT_EQ(batch.targets[1].values.shape[2], 4);
    ASSERT_EQ(batch.targets[5].values.rank, 2);
    EXPECT_EQ(batch.targets[5].values.shape[0], 2);
    EXPECT_EQ(batch.targets[5].values.shape[1], 1);
    ASSERT_EQ(batch.targets[7].values.rank, 2);
    EXPECT_EQ(batch.targets[7].values.shape[0], 2);
    EXPECT_EQ(batch.targets[7].values.shape[1], 3);
}

/** @brief Tests that batch 0 holds the choice and the yes, role by role. */
TEST_F(Medium_LanguageDatasetsDecisionDataset, ABatchHoldsTheValuesOfItsRecords)
{
    const Support::ScratchDirectory scratch("decision_dataset_values");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<DecisionDataset> dataset,
        DecisionDataset::make(*engine, four_records_file(scratch), pair_options()));
    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    EXPECT_EQ(elements_of<std::int32_t>(batch.inputs, 8),
              (std::vector<std::int32_t>{1, 2, 3, 0, 7, 0, 0, 0}));
    EXPECT_EQ(elements_of<float>(batch.targets[0].values, 2), (std::vector<float>{3.0f, 1.0f}));
    EXPECT_EQ(elements_of<std::int32_t>(batch.targets[1].values, 24),
              (std::vector<std::int32_t>{4, 5, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(elements_of<float>(batch.targets[2].values, 6),
              (std::vector<float>{2.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    EXPECT_EQ(elements_of<float>(batch.targets[3].values, 6),
              (std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
    EXPECT_EQ(elements_of<std::int32_t>(batch.targets[4].values, 2),
              (std::vector<std::int32_t>{1, 0}));
    EXPECT_EQ(elements_of<float>(batch.targets[5].values, 2), (std::vector<float>{0.0f, 1.0f}));
    EXPECT_EQ(elements_of<float>(batch.targets[6].values, 2), (std::vector<float>{0.0f, 0.0f}));
    EXPECT_EQ(elements_of<float>(batch.targets[7].values, 6),
              (std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}));
}

/** @brief Tests that batch 1 holds the number and the no, the records after the first batch. */
TEST_F(Medium_LanguageDatasetsDecisionDataset, TheSecondBatchHoldsTheLaterRecords)
{
    const Support::ScratchDirectory scratch("decision_dataset_second");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<DecisionDataset> dataset,
        DecisionDataset::make(*engine, four_records_file(scratch), pair_options()));
    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, batch)));

    EXPECT_EQ(elements_of<std::int32_t>(batch.inputs, 8),
              (std::vector<std::int32_t>{8, 9, 10, 11, 3, 3, 0, 0}));
    EXPECT_EQ(elements_of<float>(batch.targets[0].values, 2), (std::vector<float>{4.0f, 2.0f}));
    EXPECT_EQ(elements_of<float>(batch.targets[5].values, 2), (std::vector<float>{0.0f, 0.0f}));
    EXPECT_EQ(elements_of<float>(batch.targets[6].values, 2), (std::vector<float>{2.5f, 0.0f}));
    EXPECT_EQ(elements_of<float>(batch.targets[7].values, 6),
              (std::vector<float>{0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f}));
}

/** @brief Tests that a range read holds the rows of the whole read it names. */
TEST_F(Medium_LanguageDatasetsDecisionDataset, ARangeReadHoldsTheRowOfTheWholeBatch)
{
    const Support::ScratchDirectory scratch("decision_dataset_range");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<DecisionDataset> dataset,
        DecisionDataset::make(*engine, four_records_file(scratch), pair_options()));

    const Data::RowRange rows{1, 1};
    Data::Batch whole;
    Data::Batch part;
    SA_EXPECT_OK(dataset->read(0, whole));
    SA_EXPECT_OK(dataset->read_rows(0, rows, part));

    EXPECT_TRUE(Support::holds_rows_of(whole, dataset->batch_size(), rows, part));
}

/** @brief Tests that make refuses options and files no batch can be built from. */
TEST_P(Medium_LanguageDatasetsDecisionDatasetRefusal, MakeRefusesOptionsOutsideTheirBounds)
{
    const RefusalCase& row = GetParam();
    const Support::ScratchDirectory scratch("decision_dataset_refusal");
    const std::filesystem::path file =
        row.absent_file ? scratch.path() / "absent.sadc" : four_records_file(scratch);
    DecisionDatasetOptions options = pair_options();
    options.batch_size = row.batch_size;
    options.shape.window = row.window;
    options.shape.options = row.options;

    SA_EXPECT_FAILURE(DecisionDataset::make(*engine, file, options), row.code, row.first,
                      row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_LanguageDatasetsDecisionDatasetRefusal,
    ::testing::Values(
        RefusalCase{"BatchSizeOfZero", 0, 4, 3, false, StatusCode::InvalidDescription,
                    "batch size of 0", ""},
        RefusalCase{"AWindowOfZero", 2, 0, 3, false, StatusCode::InvalidDescription,
                    "window of 0", ""},
        // The third record has four ids, so a window of 3 cuts it.
        RefusalCase{"ARecordLongerThanTheWindow", 2, 3, 3, false,
                    StatusCode::InvalidDescription, "four.sadc", "text of record 2"},
        // The first record has two options.
        RefusalCase{"ARecordWithMoreOptionsThanK", 2, 4, 1, false,
                    StatusCode::InvalidDescription, "four.sadc", "record 0 holds 2 options"},
        RefusalCase{"FewerRecordsThanABatch", 5, 4, 3, false, StatusCode::InvalidDescription,
                    "four.sadc", "4 records, fewer than one batch of 5"},
        RefusalCase{"AnAbsentFile", 2, 4, 3, true, StatusCode::FileNotFound, "absent.sadc",
                    ""}),
    label_of<RefusalCase>);

/** @brief Tests that a data block builds the dataset over the file it names. */
TEST_F(Medium_LanguageDatasetsDecisionDataset, ADataBlockBuildsTheDataset)
{
    const Support::ScratchDirectory scratch("decision_dataset_block");
    const std::string file = four_records_file(scratch).generic_string();
    const Config::Value section = SaTest::require_value(Config::parse_json(
        "{\"file\": \"" + file + "\", \"window\": 4, \"options\": 3, \"batch_size\": 2}",
        "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> dataset,
                            make_decisions_dataset(*engine, section, "the data block"));

    EXPECT_EQ(dataset->batch_size(), 2);
    EXPECT_EQ(dataset->batch_count(), 2u);
    EXPECT_EQ(dataset->sample_shape(), (Shape{K_WINDOW}));
    EXPECT_EQ(dataset->targets().size(), 8u);
}
