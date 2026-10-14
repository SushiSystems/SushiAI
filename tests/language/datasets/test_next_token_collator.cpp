/**************************************************************************/
/* test_next_token_collator.cpp                                           */
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
 * @file test_next_token_collator.cpp
 * @brief Tests the collator of the next-token task.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/next_token_collator.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/layer_rows/minimal_case.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Datasets;
using Language::Text::TokenID;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides an engine and the batches these tests collate into. */
    class Unit_LanguageDatasetsNextTokenCollator : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns a batch of an input and a labels tensor of [rows, length]. */
            Data::Batch batch_of(std::int64_t rows, std::int64_t length, DType input_type)
            {
                Data::Batch batch;

                Result<SushiBLAS::Tensor> inputs =
                    allocate_tensor(*engine, Shape{rows, length}, input_type);
                Result<SushiBLAS::Tensor> labels =
                    allocate_tensor(*engine, Shape{rows, length}, DType::Int32);
                EXPECT_TRUE(inputs.status().is_ok());
                EXPECT_TRUE(labels.status().is_ok());

                batch.inputs = std::move(inputs).value();
                batch.targets.push_back(Data::BatchTarget{std::string(Data::K_ROLE_LABELS),
                                                          std::move(labels).value()});
                return batch;
            }
    };

    /** @brief Returns the ids as the span a collator reads. */
    SushiRuntime::span<const TokenID> span_of(const std::vector<TokenID>& ids)
    {
        return SushiRuntime::span<const TokenID>(ids.data(), ids.size());
    }
} // namespace

/** @brief Tests that the collator states its task, its window and the labels role. */
TEST_F(Unit_LanguageDatasetsNextTokenCollator, ItDescribesTheTask)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenCollator> collator,
                            NextTokenCollator::make(3, 50));

    EXPECT_EQ(collator->task(), K_TASK_NEXT_TOKEN);
    EXPECT_EQ(collator->window_length(), 3);
    EXPECT_EQ(collator->tokens_per_window(), 4);
    EXPECT_EQ(collator->vocabulary_size(), 50);

    ASSERT_EQ(collator->targets().size(), 1u);
    const Data::TargetDescription& labels = collator->targets()[0];
    EXPECT_EQ(labels.role, Data::K_ROLE_LABELS);
    EXPECT_EQ(labels.sample_shape, (Shape{3}));
    EXPECT_EQ(labels.dtype, DType::Int32);
    EXPECT_EQ(labels.classes, 50);
}

/** @brief Tests that labels[n][t] is the id that follows inputs[n][t] in its window. */
TEST_F(Unit_LanguageDatasetsNextTokenCollator, LabelsHoldTheNextTokenOfEachPosition)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenCollator> collator,
                            NextTokenCollator::make(3, 50));
    Data::Batch batch = batch_of(2, 3, DType::Int32);

    const std::vector<TokenID> tokens = {10, 11, 12, 13, 20, 21, 22, 23};
    ASSERT_TRUE(SaTest::status_is_ok(collator->collate(span_of(tokens), batch)));

    const TokenID* const inputs = batch.inputs.data_as<TokenID>();
    const TokenID* const labels = batch.targets[0].values.data_as<TokenID>();
    const std::vector<TokenID> expected_inputs = {10, 11, 12, 20, 21, 22};
    const std::vector<TokenID> expected_labels = {11, 12, 13, 21, 22, 23};

    for (std::size_t i = 0; i < expected_inputs.size(); ++i)
    {
        EXPECT_EQ(inputs[i], expected_inputs[i]) << i;
        EXPECT_EQ(labels[i], expected_labels[i]) << i;
    }
}

/** @brief Tests that a window length or a vocabulary size below 1 is refused. */
TEST_F(Unit_LanguageDatasetsNextTokenCollator, MakeRefusesExtentsOutOfBounds)
{
    SA_EXPECT_FAILURE(NextTokenCollator::make(0, 50), StatusCode::InvalidDescription,
                      "NextTokenCollator", "window of 0 positions");
    SA_EXPECT_FAILURE(NextTokenCollator::make(3, 0), StatusCode::InvalidDescription,
                      "vocabulary of 0 tokens");
}

/** @brief Tests that ids that are no whole windows, and a batch of another shape, are refused. */
TEST_F(Unit_LanguageDatasetsNextTokenCollator, CollateRefusesWhatDoesNotFit)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenCollator> collator,
                            NextTokenCollator::make(3, 50));

    Data::Batch batch = batch_of(2, 3, DType::Int32);
    const std::vector<TokenID> ragged = {1, 2, 3, 4, 5, 6, 7};
    SA_EXPECT_FAILURE(collator->collate(span_of(ragged), batch), StatusCode::Incompatible,
                      "NextTokenCollator", "7 ids", "windows of 4");

    const std::vector<TokenID> three_windows(12, 1);
    SA_EXPECT_FAILURE(collator->collate(span_of(three_windows), batch),
                      StatusCode::Incompatible, "inputs", "6 elements", "writes 9");

    Data::Batch floating = batch_of(2, 3, DType::Float32);
    const std::vector<TokenID> two_windows(8, 1);
    SA_EXPECT_FAILURE(collator->collate(span_of(two_windows), floating),
                      StatusCode::Incompatible, "inputs", "int32");

    Data::Batch bare;
    SA_EXPECT_FAILURE(collator->collate(span_of(two_windows), bare), StatusCode::Incompatible,
                      "one role 'labels'");
}
