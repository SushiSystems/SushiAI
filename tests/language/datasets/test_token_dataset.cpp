/**************************************************************************/
/* test_token_dataset.cpp                                                 */
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
 * @file test_token_dataset.cpp
 * @brief Tests the dataset that reads each batch of windows from a token file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/language/datasets/collator.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/next_token_collator.hpp>
#include <SushiAI/language/datasets/token_dataset.hpp>
#include <SushiAI/language/datasets/token_file.hpp>
#include <SushiAI/language/text/token.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Datasets;
using Language::Text::TokenID;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the ids of the file these tests read, and its vocabulary size. */
    constexpr std::int32_t K_IDS = 20;

    /** @brief Holds the positions of one input row of these tests. */
    constexpr std::int64_t K_LENGTH = 4;

    /** @brief Provides an engine and the token file these tests read. */
    class Medium_LanguageDatasetsTokenDataset : public SaTest::EngineTest
    {
        protected:
            /** @brief Writes the ids 0 to 19 over 20 tokens, so an id is its own position. */
            std::filesystem::path rising_file(const Support::ScratchDirectory& scratch)
            {
                const std::filesystem::path file = scratch.path() / "rising.satk";
                const std::vector<TokenID> ids = Support::rising_ids(K_IDS);

                EXPECT_TRUE(SaTest::status_is_ok(write_token_file(
                    file, SushiRuntime::span<const TokenID>(ids.data(), ids.size()), K_IDS)));
                return file;
            }

            /** @brief Returns a next-token collator of four positions over a vocabulary. */
            std::unique_ptr<const Collator> next_token(std::int32_t vocabulary_size = K_IDS)
            {
                Result<std::unique_ptr<NextTokenCollator>> made =
                    NextTokenCollator::make(K_LENGTH, vocabulary_size);
                EXPECT_TRUE(made.status().is_ok());
                return std::move(made).value();
            }

            /** @brief Returns the options of batches of two windows named "tiny". */
            TokenDatasetOptions tiny_options() const
            {
                TokenDatasetOptions options;
                options.batch_size = 2;
                options.name = "tiny";
                return options;
            }
    };

    /** @brief Returns the ids @p first to @p first + @p count - 1, a window of the rising file. */
    std::vector<TokenID> ids_from(TokenID first, std::size_t count)
    {
        std::vector<TokenID> ids(count);
        std::iota(ids.begin(), ids.end(), first);
        return ids;
    }

    /** @brief Returns @p count ids of a tensor as a vector, so a check compares them whole. */
    std::vector<TokenID> ids_of(const SushiBLAS::Tensor& tensor, std::size_t count)
    {
        const TokenID* const data = tensor.data_as<TokenID>();
        return std::vector<TokenID>(data, data + count);
    }

    /** @brief Returns the ids of the windows at @p first and @p second, one after the other. */
    std::vector<TokenID> two_windows(TokenID first, TokenID second, TokenID shift)
    {
        std::vector<TokenID> ids = ids_from(first + shift, static_cast<std::size_t>(K_LENGTH));
        const std::vector<TokenID> next =
            ids_from(second + shift, static_cast<std::size_t>(K_LENGTH));
        ids.insert(ids.end(), next.begin(), next.end());
        return ids;
    }

    /** @brief Names the batch of a stride of 3 and where its two windows begin. */
    struct StrideCase
    {
        const char* label;  /**< The name the row prints. */
        std::size_t batch;  /**< The batch index. */
        TokenID first;      /**< The id the first window of the batch begins at. */
        TokenID second;     /**< The id the second window of the batch begins at. */
    };

    /** @brief Provides the batches of a stride of 3 over the rising file. */
    class Medium_LanguageDatasetsTokenDatasetStride
        : public Medium_LanguageDatasetsTokenDataset,
          public ::testing::WithParamInterface<StrideCase>
    {

    };

    /** @brief Names the options make is given and what its refusal says. */
    struct RefusalCase
    {
        const char* label;                 /**< The name the row prints. */
        bool no_collator;                  /**< Whether make is given no collator. */
        bool absent_file;                  /**< Whether make is given a file that is not there. */
        std::int64_t batch_size;           /**< The windows of a batch. */
        std::optional<std::int64_t> stride; /**< The stride, or none for the window length. */
        std::int32_t vocabulary;           /**< The vocabulary size of the collator. */
        StatusCode code;                   /**< The code of the refusal. */
        const char* first;                 /**< A fragment the refusal names. */
        const char* second;                /**< A second fragment, or an empty text. */
        const char* third;                 /**< A third fragment, or an empty text. */
    };

    /** @brief Provides the options make refuses. */
    class Medium_LanguageDatasetsTokenDatasetRefusal
        : public Medium_LanguageDatasetsTokenDataset,
          public ::testing::WithParamInterface<RefusalCase>
    {

    };

    /** @brief Names a row range and a batch index outside the dataset. */
    struct RangeCase
    {
        const char* label;   /**< The name the row prints. */
        std::size_t index;   /**< The batch index. */
        Data::RowRange rows; /**< The rows asked for. */
        const char* first;   /**< A fragment the refusal names. */
        const char* second;  /**< A second fragment, or an empty text. */
        const char* third;   /**< A third fragment, or an empty text. */
    };

    /** @brief Provides the ranges the dataset refuses. */
    class Medium_LanguageDatasetsTokenDatasetRange
        : public Medium_LanguageDatasetsTokenDataset,
          public ::testing::WithParamInterface<RangeCase>
    {

    };

    /** @brief Returns the name a row prints. */
    template <typename Case>
    std::string label_of(const ::testing::TestParamInfo<Case>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.label);
    }
} // namespace

/** @brief Tests that the dataset reports its batch size, sample shape, input type and batches. */
TEST_F(Medium_LanguageDatasetsTokenDataset, ItDescribesItsBatches)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_describe");

    // 20 ids hold the windows of 5 ids at 0, 4, 8 and 12: two batches of two.
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, rising_file(scratch), next_token(), tiny_options()));

    EXPECT_EQ(dataset->batch_size(), 2);
    EXPECT_EQ(dataset->sample_shape(), (Shape{K_LENGTH}));
    EXPECT_EQ(dataset->sample_dtype(), DType::Int32);
    EXPECT_EQ(dataset->batch_count(), 2u);
    EXPECT_EQ(dataset->name(), "tiny");
}

/** @brief Tests that the dataset reports its vocabulary and the labels role of its collator. */
TEST_F(Medium_LanguageDatasetsTokenDataset, ItDescribesItsVocabularyAndLabels)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_labels");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, rising_file(scratch), next_token(), tiny_options()));

    EXPECT_EQ(dataset->vocabulary_size(), K_IDS);
    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(dataset->targets()[0].sample_shape, (Shape{K_LENGTH}));
    EXPECT_EQ(dataset->targets()[0].classes, K_IDS);
}

/** @brief Tests that batch 1 holds windows 2 and 3 as int32 inputs and labels of [2, 4]. */
TEST_F(Medium_LanguageDatasetsTokenDataset, ABatchHoldsItsWindowsAsInt32)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_batch");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, rising_file(scratch), next_token(), tiny_options()));
    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, batch)));

    ASSERT_EQ(batch.inputs.dtype, DType::Int32);
    ASSERT_EQ(batch.inputs.rank, 2);
    EXPECT_EQ(batch.inputs.shape[0], 2);
    EXPECT_EQ(batch.inputs.shape[1], K_LENGTH);
    ASSERT_EQ(batch.targets.size(), 1u);
    ASSERT_EQ(batch.targets[0].values.dtype, DType::Int32);
    ASSERT_EQ(batch.targets[0].values.num_elements, 8);

    EXPECT_EQ(ids_of(batch.inputs, 8), (std::vector<TokenID>{8, 9, 10, 11, 12, 13, 14, 15}));
    EXPECT_EQ(ids_of(batch.targets[0].values, 8),
              (std::vector<TokenID>{9, 10, 11, 12, 13, 14, 15, 16}));
}

/** @brief Tests, over each batch of overlapping windows, that a label is the next token. */
TEST_P(Medium_LanguageDatasetsTokenDatasetStride, EveryLabelIsTheTokenAfterItsInput)
{
    const StrideCase& row = GetParam();
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_next");

    // A stride of 3 gives the windows at 0, 3, 6, 9, 12 and 15: three batches of two.
    TokenDatasetOptions options = tiny_options();
    options.stride = 3;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, rising_file(scratch), next_token(), options));
    ASSERT_EQ(dataset->batch_count(), 3u);
    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(row.batch, batch)));

    // The file holds the id p at position p, so a label is the input id plus one.
    EXPECT_EQ(ids_of(batch.inputs, 8), two_windows(row.first, row.second, 0));
    EXPECT_EQ(ids_of(batch.targets[0].values, 8), two_windows(row.first, row.second, 1));
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_LanguageDatasetsTokenDatasetStride,
                         ::testing::Values(StrideCase{"BatchZero", 0, 0, 3},
                                           StrideCase{"BatchOne", 1, 6, 9},
                                           StrideCase{"BatchTwo", 2, 12, 15}),
                         label_of<StrideCase>);

/** @brief Tests that make refuses options no batch can be built from. */
TEST_P(Medium_LanguageDatasetsTokenDatasetRefusal, MakeRefusesOptionsOutsideTheirBounds)
{
    const RefusalCase& row = GetParam();
    const Support::ScratchDirectory scratch((std::string("sushiai_test_language_token_dataset_refusal_") + row.label).c_str());
    const std::filesystem::path file =
        row.absent_file ? scratch.path() / "absent.satk" : rising_file(scratch);
    TokenDatasetOptions options = tiny_options();
    options.batch_size = row.batch_size;
    options.stride = row.stride;

    SA_EXPECT_FAILURE(
        TokenDataset::make(*engine, file,
                           row.no_collator ? std::unique_ptr<const Collator>()
                                           : next_token(row.vocabulary),
                           options),
        row.code, row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_LanguageDatasetsTokenDatasetRefusal,
    ::testing::Values(
        RefusalCase{"NoCollator", true, false, 2, std::nullopt, K_IDS,
                    StatusCode::InvalidDescription, "TokenDataset", "tiny", "no collator"},
        RefusalCase{"BatchSizeOfZero", false, false, 0, std::nullopt, K_IDS,
                    StatusCode::InvalidDescription, "batch size of 0", "", ""},
        RefusalCase{"StrideOfZero", false, false, 2, 0, K_IDS, StatusCode::InvalidDescription,
                    "TokenDataset", "tiny", "stride of 0"},
        // 20 ids hold four windows of 5 at a stride of 4, fewer than one batch of five.
        RefusalCase{"FewerWindowsThanABatch", false, false, 5, std::nullopt, K_IDS,
                    StatusCode::InvalidDescription, "rising.satk", "4 windows of 5",
                    "one batch of 5"},
        RefusalCase{"AVocabularyBelowTheFile", false, false, 2, std::nullopt, 19,
                    StatusCode::Incompatible, "rising.satk", "vocabulary of 20", "one of 19"},
        RefusalCase{"AnAbsentFile", false, true, 2, std::nullopt, K_IDS,
                    StatusCode::FileNotFound, "absent.satk", "", ""}),
    label_of<RefusalCase>);

/** @brief Tests that a file cut after the dataset was made is a failure naming where it ends. */
TEST_F(Medium_LanguageDatasetsTokenDataset, AFileCutAfterMakeNamesWhereItEnds)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_cut");
    const std::filesystem::path file = rising_file(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, file, next_token(), tiny_options()));

    // 64 bytes keep ids 0 to 9; batch 1 reads ids 8 to 12, which end at byte 76.
    std::filesystem::resize_file(file, 64);

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));
    SA_EXPECT_FAILURE(dataset->read(1, batch), StatusCode::InvalidFile, "rising.satk",
                      "ends at byte 64", "end at byte 76");
}

/** @brief Tests that a batch index past the last batch stops the process. */
TEST_F(Medium_LanguageDatasetsTokenDataset, ABatchOutsideTheEpochStopsTheProcess)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_index");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, rising_file(scratch), next_token(), tiny_options()));

    Data::Batch batch;
    SA_EXPECT_FATAL((void)dataset->read(2, batch), "TokenDataset", "2 batches", "batch 2");
}

/** @brief Tests that a second read into the same batch writes the tensors of the first. */
TEST_F(Medium_LanguageDatasetsTokenDataset, AReadReusesTheTensorsItFilledBefore)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_reuse");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, rising_file(scratch), next_token(), tiny_options()));

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));
    const TokenID* const first = batch.inputs.data_as<TokenID>();
    const TokenID* const first_labels = batch.targets[0].values.data_as<TokenID>();

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, batch)));
    EXPECT_EQ(batch.inputs.data_as<TokenID>(), first);
    EXPECT_EQ(batch.targets[0].values.data_as<TokenID>(), first_labels);
    EXPECT_EQ(first[0], 8);
}

/** @brief Tests that a read leaves alone a tensor something else still references. */
TEST_F(Medium_LanguageDatasetsTokenDataset, AReadDoesNotWriteATensorSomethingElseHolds)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_shared");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, rising_file(scratch), next_token(), tiny_options()));

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));
    const SushiBLAS::Tensor kept = batch.inputs;

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, batch)));
    EXPECT_NE(batch.inputs.data_as<TokenID>(), kept.data_as<TokenID>());
    EXPECT_EQ(kept.data_as<TokenID>()[0], 0);
    EXPECT_EQ(batch.inputs.data_as<TokenID>()[0], 8);
}

/** @brief Tests that a token dataset is read through the Data::Dataset interface. */
TEST_F(Medium_LanguageDatasetsTokenDataset, ItIsADatasetATrainerCanRead)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_seam");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> made,
        TokenDataset::make(*engine, rising_file(scratch), next_token(), tiny_options()));
    const Data::Dataset& dataset = *made;

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset.read(1, batch)));
    ASSERT_EQ(batch.targets.size(), dataset.targets().size());
    EXPECT_EQ(batch.targets[0].role, dataset.targets()[0].role);
    EXPECT_EQ(batch.inputs.shape[0], dataset.batch_size());
}

/** @brief Tests that a range read holds the windows the whole read holds at those rows. */
TEST_F(Medium_LanguageDatasetsTokenDataset, ARangeReadHoldsTheRowsOfTheWholeBatch)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_rows");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, rising_file(scratch), next_token(), tiny_options()));

    const Data::RowRange rows{1, 1};
    Data::Batch whole;
    Data::Batch part;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, whole)));
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read_rows(1, rows, part)));

    EXPECT_EQ(part.inputs.shape[0], rows.count);
    EXPECT_TRUE(Support::holds_rows_of(whole, dataset->batch_size(), rows, part));
}

/** @brief Tests that a range read reads no window outside it, so a cut past it is not seen. */
TEST_F(Medium_LanguageDatasetsTokenDataset, ARangeReadReadsNoWindowOutsideIt)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_rows_cut");
    const std::filesystem::path file = rising_file(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, file, next_token(), tiny_options()));

    // 76 bytes keep ids 0 to 12: window 2 ends there and window 3, ids 12 to 16, does not.
    std::filesystem::resize_file(file, 76);

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read_rows(1, Data::RowRange{0, 1}, batch)));
    EXPECT_EQ(batch.inputs.shape[0], 1);
    SA_EXPECT_FAILURE(dataset->read(1, batch), StatusCode::InvalidFile, "rising.satk");
}

/** @brief Tests that a range outside the token dataset is refused with a Status, not a stop. */
TEST_P(Medium_LanguageDatasetsTokenDatasetRange, RegressionABadRangeIsRefusedWithAStatus)
{
    const RangeCase& row = GetParam();
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_bad_rows");
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, rising_file(scratch), next_token(), tiny_options()));
    Data::Batch batch;

    SA_EXPECT_FAILURE((dataset->read_rows(row.index, row.rows, batch)),
                      SushiAI::StatusCode::InvalidDescription, row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_LanguageDatasetsTokenDatasetRange,
    ::testing::Values(
        RangeCase{"RowsPastTheBatch", 1, Data::RowRange{1, 2}, "Rows 1 to 3", "of batch 1",
                  "a batch of 2"},
        RangeCase{"ABatchOutOfRange", 99, Data::RowRange{0, 1}, "Batch 99 is out of range", "",
                  ""}),
    label_of<RangeCase>);
