/**************************************************************************/
/* test_in_memory_window_logits.cpp                                      */
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
 * @file test_in_memory_window_logits.cpp
 * @brief Tests that InMemoryWindowLogits scores the successor of each id and keeps its windows.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/in_memory_window_logits.hpp>
#include <SushiAI/language/text/token.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;
using Language::Text::TokenID;

namespace
{
    /** @brief Holds T, the window of the source under test. */
    constexpr std::int64_t K_WINDOW = 3;

    /** @brief Holds V, the vocabulary of the source under test. */
    constexpr std::int64_t K_VOCABULARY = 5;

    /** @brief Names a position of the window {1, 4, 0} and the token that scores 1 there. */
    struct HotCase
    {
        std::size_t position; /**< The position of the window. */
        std::size_t hot;      /**< The successor of the id at the position, modulo V. */
    };

    /** @brief Returns the successors of 1, 4 and 0 over five tokens: 2, 0 and 1. */
    std::vector<HotCase> hot_cases()
    {
        return {{0, 2}, {1, 0}, {2, 1}};
    }

    /** @brief Views ids as the span a source reads. */
    SushiRuntime::span<const TokenID> span_of(const std::vector<TokenID>& ids)
    {
        return SushiRuntime::span<const TokenID>(ids.data(), ids.size());
    }

    /** @brief Provides the positions and their hot tokens. */
    class Small_InMemoryWindowLogitsHot : public ::testing::TestWithParam<HotCase>
    {

    };
} // namespace

/** @brief Tests that a position scores 1 at the successor of its id and 0 elsewhere. */
TEST_P(Small_InMemoryWindowLogitsHot, APositionScoresTheSuccessorOfItsId)
{
    const HotCase& row = GetParam();
    InMemoryWindowLogits source(K_WINDOW, K_VOCABULARY);
    const std::vector<TokenID> ids{1, 4, 0};

    SA_EXPECT_OK(source.run(span_of(ids)));

    const LogitSpan scores = source.row(row.position);
    ASSERT_EQ(scores.size(), static_cast<std::size_t>(K_VOCABULARY));
    EXPECT_EQ(scores[row.hot], 1.0f);
    EXPECT_EQ(scores[(row.hot + 1) % scores.size()], 0.0f);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_InMemoryWindowLogitsHot, ::testing::ValuesIn(hot_cases()));

/** @brief Tests that every window it was run on is kept, in order. */
TEST(Small_InMemoryWindowLogits, TheWindowsItWasRunOnAreKept)
{
    InMemoryWindowLogits source(K_WINDOW, K_VOCABULARY);

    SA_EXPECT_OK(source.run(span_of({1, 2, 3})));
    SA_EXPECT_OK(source.run(span_of({0, 0, 4})));

    EXPECT_EQ(source.windows(), (std::vector<std::vector<TokenID>>{{1, 2, 3}, {0, 0, 4}}));
}

/** @brief Tests that a window of another length is refused. */
TEST(Small_InMemoryWindowLogits, AWindowOfAnotherLengthIsRefused)
{
    InMemoryWindowLogits source(K_WINDOW, K_VOCABULARY);

    SA_EXPECT_FAILURE(source.run(span_of({1, 2})), StatusCode::InvalidDescription, "2 ids");
}
