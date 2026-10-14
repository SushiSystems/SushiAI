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
 * @file test_next_token_targets.cpp
 * @brief Tests the fill of the next-token target leaf and the metrics read from the loss.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/training/next_token_targets.hpp>
#include <SushiAI/language/training/perplexity.hpp>
#include <SushiAI/train/objective.hpp>

using namespace SushiAI;
using Language::Training::fill_next_token_targets;
// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the rows of a batch, N. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the positions of a row, T. */
    constexpr std::int64_t K_LENGTH = 3;

    /** @brief Holds the vocabulary size, V. */
    constexpr std::int64_t K_VOCABULARY = 5;

    /** @brief Holds the value a leaf starts at, which no fill of these cases writes. */
    constexpr std::int32_t K_UNTOUCHED = -7;

    /** @brief Provides an engine, [N, T] labels and the [N * T] leaf they fill. */
    class Unit_LanguageTrainingNextTokenTargets : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns an INT32 tensor of @p shape holding @p values. */
            SushiBLAS::Tensor make_ids(const Shape& shape, const std::vector<std::int32_t>& values)
            {
                Result<SushiBLAS::Tensor> made = allocate_tensor(*engine, shape, DType::Int32);
                EXPECT_TRUE(made.status().is_ok());
                EXPECT_EQ(static_cast<std::size_t>(made.value().num_elements), values.size());

                std::int32_t* const ids = made.value().data_as<std::int32_t>();
                for (std::size_t i = 0; i < values.size(); ++i)
                    ids[i] = values[i];
                return std::move(made).value();
            }

            /** @brief Returns the [N * T] leaf with every element at K_UNTOUCHED. */
            SushiBLAS::Tensor make_leaf()
            {
                return make_ids(Shape{K_BATCH * K_LENGTH},
                                std::vector<std::int32_t>(
                                    static_cast<std::size_t>(K_BATCH * K_LENGTH), K_UNTOUCHED));
            }

            /** @brief Returns the elements of the INT32 tensor @p tensor. */
            static std::vector<std::int32_t> read_ids(const SushiBLAS::Tensor& tensor)
            {
                const std::int32_t* const ids = tensor.data_as<std::int32_t>();
                return std::vector<std::int32_t>(ids, ids + tensor.num_elements);
            }
    };
} // namespace

/** @brief Tests that position r of label l becomes r * V + l, row after row. */
TEST_F(Unit_LanguageTrainingNextTokenTargets, APositionIndexesItsLabelAmongEveryScore)
{
    const SushiBLAS::Tensor labels = make_ids(Shape{K_BATCH, K_LENGTH}, {4, 0, 2, 1, 1, 3});
    SushiBLAS::Tensor targets = make_leaf();

    SA_EXPECT_OK(fill_next_token_targets(labels, K_VOCABULARY, targets));

    EXPECT_EQ(read_ids(targets), (std::vector<std::int32_t>{4, 5, 12, 16, 21, 28}));
}

/** @brief Tests that a label outside the vocabulary is refused and the leaf is left alone. */
TEST_F(Unit_LanguageTrainingNextTokenTargets, ALabelOutsideTheVocabularyIsRefused)
{
    SushiBLAS::Tensor targets = make_leaf();

    const SushiBLAS::Tensor high = make_ids(Shape{K_BATCH, K_LENGTH}, {4, 0, 2, 1, 5, 3});
    SA_EXPECT_FAILURE(fill_next_token_targets(high, K_VOCABULARY, targets),
                      StatusCode::Incompatible, "position 4 is 5", "holds 5 tokens");

    const SushiBLAS::Tensor negative = make_ids(Shape{K_BATCH, K_LENGTH}, {4, 0, -1, 1, 1, 3});
    SA_EXPECT_FAILURE(fill_next_token_targets(negative, K_VOCABULARY, targets),
                      StatusCode::Incompatible, "position 2 is -1");

    for (const std::int32_t value : read_ids(targets))
        EXPECT_EQ(value, K_UNTOUCHED);
}

/** @brief Tests that another count, another element type and an unindexable size are refused. */
TEST_F(Unit_LanguageTrainingNextTokenTargets, AFillThatDoesNotFitIsRefused)
{
    SushiBLAS::Tensor targets = make_leaf();

    const SushiBLAS::Tensor short_labels = make_ids(Shape{K_BATCH, 2}, {4, 0, 2, 1});
    SA_EXPECT_FAILURE(fill_next_token_targets(short_labels, K_VOCABULARY, targets),
                      StatusCode::Incompatible, "holds 4 labels", "expects 6");

    SA_ASSERT_OK_AND_ASSIGN(
        const SushiBLAS::Tensor real_labels,
        allocate_tensor(*engine, Shape{K_BATCH, K_LENGTH}, DType::Float32));
    SA_EXPECT_FAILURE(fill_next_token_targets(real_labels, K_VOCABULARY, targets),
                      StatusCode::Incompatible, "float32", "int32");

    const SushiBLAS::Tensor labels = make_ids(Shape{K_BATCH, K_LENGTH}, {4, 0, 2, 1, 1, 3});
    SA_EXPECT_FAILURE(
        fill_next_token_targets(labels, std::numeric_limits<std::int32_t>::max(), targets),
        StatusCode::Incompatible, "6 positions", "cannot be indexed");
    SA_EXPECT_FAILURE(fill_next_token_targets(labels, 0, targets), StatusCode::Incompatible,
                      "cannot be indexed");
}

/** @brief Tests that the perplexity is the exponential of the mean loss. */
TEST(Unit_LanguageTrainingPerplexity, ThePerplexityIsTheExponentialOfTheLoss)
{
    // A uniform guess over 64 tokens costs ln(64) nats and has a perplexity of 64.
    constexpr double K_RELATIVE_TOLERANCE = 1e-12;
    const double uniform = std::log(64.0);

    EXPECT_NEAR(Language::Training::perplexity(0.0), 1.0, 0.0);
    EXPECT_NEAR(Language::Training::perplexity(uniform), 64.0, 64.0 * K_RELATIVE_TOLERANCE);
    EXPECT_TRUE(std::isinf(Language::Training::perplexity(1e6)));
}

/** @brief Tests that the two metrics are named loss and perplexity and read as bare numbers. */
TEST(Unit_LanguageTrainingPerplexity, TheMetricsAreTheLossAndThePerplexity)
{
    const std::array<Train::Metric, 2> metrics =
        Language::Training::next_token_metrics(std::log(8.0));

    EXPECT_EQ(metrics[0].name, "loss");
    EXPECT_NEAR(metrics[0].value, std::log(8.0), 0.0);
    EXPECT_FALSE(metrics[0].is_fraction);
    EXPECT_EQ(metrics[1].name, "perplexity");
    EXPECT_NEAR(metrics[1].value, 8.0, 8.0 * 1e-12);
    EXPECT_FALSE(metrics[1].is_fraction);
}
