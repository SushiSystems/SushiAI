/**************************************************************************/
/* test_next_token_inputs.cpp                                             */
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
 * @file test_next_token_inputs.cpp
 * @brief Tests the host check of the token ids a batch feeds to the embedding.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/training/next_token_inputs.hpp>

using namespace SushiAI;
using Language::Training::check_next_token_inputs;

namespace
{
    /** @brief Holds the rows of the embedding table, V. */
    constexpr std::int64_t K_VOCABULARY = 5;

    /** @brief Checks @p ids against K_VOCABULARY through the span overload. */
    Status check(const std::vector<std::int32_t>& ids)
    {
        return check_next_token_inputs(
            SushiRuntime::span<const std::int32_t>(ids.data(), ids.size()), K_VOCABULARY);
    }

    /** @brief Names ids the check refuses and the parts of the message that says why. */
    struct RefusedIds
    {
        std::vector<std::int32_t> ids; /**< The ids given. */
        std::string first;             /**< The first part of the message. */
        std::string second;            /**< The second part of the message. */
    };

    /** @brief Provides the ids outside the table. */
    class Small_LanguageTrainingNextTokenInputs : public ::testing::TestWithParam<RefusedIds>
    {
    };
} // namespace

/** @brief Tests that ids from 0 to V - 1 pass. */
TEST(Small_LanguageTrainingNextTokenInputsAccepted, EveryRowOfTheTableIsAnId)
{
    SA_EXPECT_OK(check({0, 4, 2, 1, 3, 0}));
    SA_EXPECT_OK(check({}));
}

/** @brief Tests that an id outside the table is refused, naming its position and the table. */
TEST_P(Small_LanguageTrainingNextTokenInputs, AnIdOutsideTheTableIsRefused)
{
    const RefusedIds& row = GetParam();

    SA_EXPECT_FAILURE(check(row.ids), StatusCode::Incompatible, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageTrainingNextTokenInputs,
    ::testing::Values(RefusedIds{{0, 3, 1, 1, 5, 2}, "position 4 is 5", "holds 5 rows"},
                      RefusedIds{{0, 3, -1, 1, 4, 2}, "position 2 is -1", "holds 5 rows"}));

/** @brief Tests that a tensor that is not int32 is refused before its elements are read. */
TEST(Small_LanguageTrainingNextTokenInputsAccepted, ATensorOfAnotherTypeIsRefused)
{
    const std::int64_t dims[1] = {6};
    const SushiBLAS::Tensor floats = SushiBLAS::Tensor::make_shape_only(
        SushiRuntime::span<const std::int64_t>(dims, 1), DType::Float32);

    SA_EXPECT_FAILURE(check_next_token_inputs(floats, K_VOCABULARY), StatusCode::Incompatible,
                      "inputs are", "read as");
}
