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
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
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
    class Medium_LanguageDatasetsNextTokenCollator : public SaTest::EngineTest
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

    /** @brief Returns @p count ids of a tensor as a vector, so a check compares them whole. */
    std::vector<TokenID> ids_of(const SushiBLAS::Tensor& tensor, std::size_t count)
    {
        const TokenID* const data = tensor.data_as<TokenID>();
        return std::vector<TokenID>(data, data + count);
    }

    /** @brief Returns a tensor description with no storage, of a type and an element count. */
    SushiBLAS::Tensor description(DType type, std::int64_t elements)
    {
        SushiBLAS::Tensor tensor;
        tensor.dtype = type;
        tensor.num_elements = elements;
        return tensor;
    }

    /** @brief Names the extents make is given and what its refusal says. */
    struct MakeCase
    {
        const char* label;       /**< The name the row prints. */
        std::int64_t window;     /**< The window length make is given. */
        std::int32_t vocabulary; /**< The vocabulary size make is given. */
        const char* message;     /**< A fragment the refusal names. */
    };

    /** @brief Provides the extents make refuses. */
    class Small_LanguageDatasetsNextTokenCollatorMake : public ::testing::TestWithParam<MakeCase>
    {

    };

    /** @brief Names a batch of descriptions, the ids given, and what the refusal says. */
    struct CollateCase
    {
        const char* label;           /**< The name the row prints. */
        std::size_t id_count;        /**< How many ids collate is given. */
        DType input_type;            /**< The type of the input tensor. */
        std::int64_t input_elements; /**< The elements of the input tensor. */
        bool has_labels_role;        /**< Whether the batch holds the labels role. */
        const char* first;           /**< A fragment the refusal names. */
        const char* second;          /**< A second fragment, or an empty text. */
        const char* third;           /**< A third fragment, or an empty text. */
    };

    /** @brief Provides the batches collate refuses; the refusals decide on shapes and types. */
    class Small_LanguageDatasetsNextTokenCollatorRefusal
        : public ::testing::TestWithParam<CollateCase>
    {

    };
} // namespace

/** @brief Tests that the collator states its task, window, ids per window and vocabulary. */
TEST(Small_LanguageDatasetsNextTokenCollator, ItDescribesTheSizesOfTheTask)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenCollator> collator,
                            NextTokenCollator::make(3, 50));

    EXPECT_EQ(collator->task(), K_TASK_NEXT_TOKEN);
    EXPECT_EQ(collator->window_length(), 3);
    EXPECT_EQ(collator->tokens_per_window(), 4);
    EXPECT_EQ(collator->vocabulary_size(), 50);
}

/** @brief Tests that the one role is the labels, an int32 of the window over the vocabulary. */
TEST(Small_LanguageDatasetsNextTokenCollator, ItDescribesTheLabelsTarget)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenCollator> collator,
                            NextTokenCollator::make(3, 50));

    ASSERT_EQ(collator->targets().size(), 1u);
    const Data::TargetDescription& labels = collator->targets()[0];
    EXPECT_EQ(labels.role, Data::K_ROLE_LABELS);
    EXPECT_EQ(labels.sample_shape, (Shape{3}));
    EXPECT_EQ(labels.dtype, DType::Int32);
    EXPECT_EQ(labels.classes, 50);
}

/** @brief Tests that labels[n][t] is the id that follows inputs[n][t] in its window. */
TEST_F(Medium_LanguageDatasetsNextTokenCollator, LabelsHoldTheNextTokenOfEachPosition)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenCollator> collator,
                            NextTokenCollator::make(3, 50));
    Data::Batch batch = batch_of(2, 3, DType::Int32);

    const std::vector<TokenID> tokens = {10, 11, 12, 13, 20, 21, 22, 23};
    ASSERT_TRUE(SaTest::status_is_ok(collator->collate(span_of(tokens), batch)));

    EXPECT_EQ(ids_of(batch.inputs, 6), (std::vector<TokenID>{10, 11, 12, 20, 21, 22}));
    EXPECT_EQ(ids_of(batch.targets[0].values, 6),
              (std::vector<TokenID>{11, 12, 13, 21, 22, 23}));
}

/** @brief Tests that a window length or a vocabulary size below 1 is refused. */
TEST_P(Small_LanguageDatasetsNextTokenCollatorMake, MakeRefusesExtentsOutOfBounds)
{
    const MakeCase& row = GetParam();

    SA_EXPECT_FAILURE(NextTokenCollator::make(row.window, row.vocabulary),
                      StatusCode::InvalidDescription, "NextTokenCollator", row.message);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LanguageDatasetsNextTokenCollatorMake,
                         ::testing::Values(MakeCase{"WindowOfZero", 0, 50, "window of 0 positions"},
                                           MakeCase{"VocabularyOfZero", 3, 0,
                                                    "vocabulary of 0 tokens"}),
                         [](const ::testing::TestParamInfo<MakeCase>& info)
                         {
                             return std::string(info.param.label);
                         });

/** @brief Tests that ids that are no whole windows, and a batch of another shape, are refused. */
TEST_P(Small_LanguageDatasetsNextTokenCollatorRefusal, CollateRefusesWhatDoesNotFit)
{
    const CollateCase& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenCollator> collator,
                            NextTokenCollator::make(3, 50));
    Data::Batch batch;
    batch.inputs = description(row.input_type, row.input_elements);
    if (row.has_labels_role)
    {
        batch.targets.push_back(Data::BatchTarget{std::string(Data::K_ROLE_LABELS),
                                                  description(DType::Int32, 6)});
    }
    const std::vector<TokenID> ids(row.id_count, 1);

    SA_EXPECT_FAILURE(collator->collate(span_of(ids), batch), StatusCode::Incompatible,
                      "NextTokenCollator", row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageDatasetsNextTokenCollatorRefusal,
    ::testing::Values(
        CollateCase{"RaggedIds", 7, DType::Int32, 6, true, "7 ids", "windows of 4", ""},
        CollateCase{"MoreWindowsThanTheBatch", 12, DType::Int32, 6, true, "inputs", "6 elements",
                    "writes 9"},
        CollateCase{"InputsOfAnotherType", 8, DType::Float32, 6, true, "inputs", "int32", ""},
        CollateCase{"NoLabelsRole", 8, DType::Int32, 6, false, "one role 'labels'", "", ""}),
    [](const ::testing::TestParamInfo<CollateCase>& info)
    {
        return std::string(info.param.label);
    });
