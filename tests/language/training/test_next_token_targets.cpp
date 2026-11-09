/**************************************************************************/
/* test_next_token_targets.cpp                                            */
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
 * @file test_next_token_targets.cpp
 * @brief Tests the fill of the next-token target leaf on spans of values.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/training/next_token_targets.hpp>

using namespace SushiAI;
using Language::Training::fill_next_token_targets;

namespace
{
    /** @brief Holds the positions of a batch, N * T. */
    constexpr std::size_t K_POSITIONS = 6;

    /** @brief Holds the vocabulary size, V. */
    constexpr std::int64_t K_VOCABULARY = 5;

    /** @brief Holds the value a leaf starts at, which no fill of these cases writes. */
    constexpr std::int32_t K_UNTOUCHED = -7;

    /** @brief Returns a leaf of @p count elements, every one at K_UNTOUCHED. */
    std::vector<std::int32_t> untouched_leaf(std::size_t count = K_POSITIONS)
    {
        return std::vector<std::int32_t>(count, K_UNTOUCHED);
    }

    /** @brief Fills @p targets from @p labels over @p vocabulary through the span overload. */
    Status fill(const std::vector<std::int32_t>& labels, std::int64_t vocabulary,
                std::vector<std::int32_t>& targets)
    {
        return fill_next_token_targets(
            SushiRuntime::span<const std::int32_t>(labels.data(), labels.size()), vocabulary,
            SushiRuntime::span<std::int32_t>(targets.data(), targets.size()));
    }

    /** @brief Returns a tensor of @p dtype with @p elements elements that holds no storage. */
    SushiBLAS::Tensor shape_of(DType dtype, std::int64_t elements)
    {
        const std::int64_t dims[1] = {elements};
        return SushiBLAS::Tensor::make_shape_only(SushiRuntime::span<const std::int64_t>(dims, 1),
                                                  dtype);
    }

    /** @brief Names labels the fill refuses and the parts of the message that says why. */
    struct RefusedLabels
    {
        std::vector<std::int32_t> labels; /**< The labels given. */
        std::string first;                /**< The first part of the message. */
        std::string second;               /**< The second part of the message. */
    };

    /** @brief Provides the labels outside the vocabulary. */
    class Small_LanguageTrainingNextTokenTargetsRange
        : public ::testing::TestWithParam<RefusedLabels>
    {

    };

    /** @brief Names a fill that does not fit: its labels, targets, vocabulary and message. */
    struct MisfitFill
    {
        std::vector<std::int32_t> labels; /**< The labels given. */
        std::size_t targets;              /**< The elements of the leaf. */
        std::int64_t vocabulary;          /**< V. */
        std::string first;                /**< The first part of the message. */
        std::string second;               /**< The second part of the message. */
    };

    /** @brief Provides the fills that do not fit. */
    class Small_LanguageTrainingNextTokenTargetsFit : public ::testing::TestWithParam<MisfitFill>
    {

    };
} // namespace

/** @brief Tests that position r of label l becomes r * V + l, row after row. */
TEST(Small_LanguageTrainingNextTokenTargets, APositionIndexesItsLabelAmongEveryScore)
{
    std::vector<std::int32_t> targets = untouched_leaf();

    SA_EXPECT_OK(fill({4, 0, 2, 1, 1, 3}, K_VOCABULARY, targets));

    EXPECT_EQ(targets, (std::vector<std::int32_t>{4, 5, 12, 16, 21, 28}));
}

/** @brief Tests that a label outside the vocabulary is refused and the leaf is left alone. */
TEST_P(Small_LanguageTrainingNextTokenTargetsRange, ALabelOutsideTheVocabularyIsRefused)
{
    const RefusedLabels& row = GetParam();
    std::vector<std::int32_t> targets = untouched_leaf();

    SA_EXPECT_FAILURE(fill(row.labels, K_VOCABULARY, targets), StatusCode::Incompatible,
                      row.first, row.second);

    EXPECT_EQ(targets, untouched_leaf());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageTrainingNextTokenTargetsRange,
    ::testing::Values(RefusedLabels{{4, 0, 2, 1, 5, 3}, "position 4 is 5", "holds 5 tokens"},
                      RefusedLabels{{4, 0, -1, 1, 1, 3}, "position 2 is -1", "holds 5 tokens"}));

/** @brief Tests that another count and an unindexable size are refused. */
TEST_P(Small_LanguageTrainingNextTokenTargetsFit, AFillThatDoesNotFitIsRefused)
{
    const MisfitFill& row = GetParam();
    std::vector<std::int32_t> targets = untouched_leaf(row.targets);

    SA_EXPECT_FAILURE(fill(row.labels, row.vocabulary, targets), StatusCode::Incompatible,
                      row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageTrainingNextTokenTargetsFit,
    ::testing::Values(
        MisfitFill{{4, 0, 2, 1}, K_POSITIONS, K_VOCABULARY, "holds 4 labels", "expects 6"},
        MisfitFill{{4, 0, 2, 1, 1, 3}, K_POSITIONS, std::numeric_limits<std::int32_t>::max(),
                   "6 positions", "cannot be indexed"},
        MisfitFill{{4, 0, 2, 1, 1, 3}, K_POSITIONS, 0, "6 positions", "cannot be indexed"}));

/** @brief Tests that labels of another element type than int32 are refused. */
TEST(Small_LanguageTrainingNextTokenTargets, LabelsOfAnotherElementTypeAreRefused)
{
    const SushiBLAS::Tensor real_labels = shape_of(DType::Float32, 6);
    SushiBLAS::Tensor targets = shape_of(DType::Int32, 6);

    SA_EXPECT_FAILURE(fill_next_token_targets(real_labels, K_VOCABULARY, targets),
                      StatusCode::Incompatible, "float32", "int32");
}
