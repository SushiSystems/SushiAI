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
 * @file test_token_dataset.cpp
 * @brief Tests the dataset that reads each batch of windows from a token file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
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
} // namespace

/** @brief Tests that the dataset reports its batch size, shapes, input type and batches. */
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

    const TokenID* const inputs = batch.inputs.data_as<TokenID>();
    const TokenID* const labels = batch.targets[0].values.data_as<TokenID>();
    const std::vector<TokenID> expected = {8, 9, 10, 11, 12, 13, 14, 15};

    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(inputs[i], expected[i]) << i;
        EXPECT_EQ(labels[i], expected[i] + 1) << i;
    }
}

/** @brief Tests, over every batch of overlapping windows, that a label is the next token. */
TEST_F(Medium_LanguageDatasetsTokenDataset, EveryLabelIsTheTokenAfterItsInput)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_next");

    // A stride of 3 gives the windows at 0, 3, 6, 9, 12 and 15: three batches of two.
    TokenDatasetOptions options = tiny_options();
    options.stride = 3;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, rising_file(scratch), next_token(), options));
    ASSERT_EQ(dataset->batch_count(), 3u);

    Data::Batch batch;
    for (std::size_t index = 0; index < dataset->batch_count(); ++index)
    {
        ASSERT_TRUE(SaTest::status_is_ok(dataset->read(index, batch)));
        const TokenID* const inputs = batch.inputs.data_as<TokenID>();
        const TokenID* const labels = batch.targets[0].values.data_as<TokenID>();

        for (std::size_t row = 0; row < 2; ++row)
        {
            const std::size_t first = (index * 2 + row) * 3;

            for (std::size_t t = 0; t < 4; ++t)
            {
                // The file holds the id p at position p.
                EXPECT_EQ(inputs[row * 4 + t], static_cast<TokenID>(first + t));
                EXPECT_EQ(labels[row * 4 + t], static_cast<TokenID>(first + t + 1));

                if (t + 1 < 4)
                    EXPECT_EQ(labels[row * 4 + t], inputs[row * 4 + t + 1]);
            }
        }
    }
}

/** @brief Tests that make refuses options no batch can be built from. */
TEST_F(Medium_LanguageDatasetsTokenDataset, MakeRefusesOptionsOutsideTheirBounds)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_refusals");
    const std::filesystem::path file = rising_file(scratch);

    SA_EXPECT_FAILURE(TokenDataset::make(*engine, file, nullptr, tiny_options()),
                      StatusCode::InvalidDescription, "TokenDataset", "tiny", "no collator");

    TokenDatasetOptions options = tiny_options();
    options.batch_size = 0;
    SA_EXPECT_FAILURE(TokenDataset::make(*engine, file, next_token(), options),
                      StatusCode::InvalidDescription, "batch size of 0");

    options = tiny_options();
    options.stride = 0;
    SA_EXPECT_FAILURE(TokenDataset::make(*engine, file, next_token(), options),
                      StatusCode::InvalidDescription, "TokenDataset", "tiny", "stride of 0");

    // 20 ids hold four windows of 5 at a stride of 4, fewer than one batch of five.
    options = tiny_options();
    options.batch_size = 5;
    SA_EXPECT_FAILURE(TokenDataset::make(*engine, file, next_token(), options),
                      StatusCode::InvalidDescription, "rising.satk", "4 windows of 5",
                      "one batch of 5");

    SA_EXPECT_FAILURE(TokenDataset::make(*engine, file, next_token(19), tiny_options()),
                      StatusCode::Incompatible, "rising.satk", "vocabulary of 20", "one of 19");

    SA_EXPECT_FAILURE(TokenDataset::make(*engine, scratch.path() / "absent.satk", next_token(),
                                         tiny_options()),
                      StatusCode::FileNotFound, "absent.satk");
}

/** @brief Tests that an id above the vocabulary of the file is a failure naming its offset. */
TEST_F(Medium_LanguageDatasetsTokenDataset, AnIDOutsideTheVocabularyNamesItsFileOffset)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_bad_id");
    const std::filesystem::path file = rising_file(scratch);

    // Byte 8 holds the vocabulary size; 5 leaves the id 5, at byte 24 + 5 * 4, outside it.
    Support::overwrite_byte(file, 8, '\x05');

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, file, next_token(), tiny_options()));

    Data::Batch batch;
    SA_EXPECT_FAILURE(dataset->read(0, batch), StatusCode::InvalidFile, "rising.satk",
                      "id 5", "byte 44", "[0, 5)");
}

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
TEST_F(Medium_LanguageDatasetsTokenDataset, RegressionABadRangeIsRefusedWithAStatus)
{
    const Support::ScratchDirectory scratch("sushiai_test_language_token_dataset_bad_rows");
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TokenDataset> dataset,
        TokenDataset::make(*engine, rising_file(scratch), next_token(), tiny_options()));
    const std::int64_t size = dataset->batch_size();
    Data::Batch batch;
    SA_EXPECT_FAILURE((dataset->read_rows(1, Data::RowRange{size - 1, 2}, batch)),
                      SushiAI::StatusCode::InvalidDescription,
                      "Rows " + std::to_string(size - 1) + " to " + std::to_string(size + 1),
                      "of batch 1", "a batch of " + std::to_string(size));
    SA_EXPECT_FAILURE((dataset->read_rows(99, Data::RowRange{0, 1}, batch)),
                      SushiAI::StatusCode::InvalidDescription, "Batch 99 is out of range");
}
