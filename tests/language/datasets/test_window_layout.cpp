/**************************************************************************/
/* test_window_layout.cpp                                                 */
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
 * @file test_window_layout.cpp
 * @brief Tests where the fixed-length windows of a token file begin.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/window_layout.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Datasets;

namespace
{
    /** @brief Names a layout out of its bounds and what the refusal says. */
    struct LayoutCase
    {
        const char* label;    /**< The name the row prints. */
        WindowLayout layout;  /**< The layout count_windows is given. */
        const char* first;    /**< A fragment the refusal names. */
        const char* second;   /**< A second fragment, or an empty text. */
    };

    /** @brief Provides the layouts count_windows refuses. */
    class Small_LanguageDatasetsWindowLayoutRefusal : public ::testing::TestWithParam<LayoutCase>
    {

    };
} // namespace

/** @brief Tests that 10 ids hold three windows of 4 at a stride of 3. */
TEST(Small_LanguageDatasetsWindowLayout, WindowsAreCountedAtTheStride)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t windows, count_windows(10, WindowLayout{4, 3}));

    EXPECT_EQ(windows, 3u);
}

/** @brief Tests that window i begins at i times the stride. */
TEST(Small_LanguageDatasetsWindowLayout, WindowsBeginOneStrideApart)
{
    const WindowLayout layout{4, 3};

    EXPECT_EQ(first_token_of(0, layout), 0u);
    EXPECT_EQ(first_token_of(1, layout), 3u);
    EXPECT_EQ(first_token_of(2, layout), 6u);
}

/** @brief Tests that a window whose last id would be past the file is not counted. */
TEST(Small_LanguageDatasetsWindowLayout, OnlyWholeWindowsAreCounted)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t exact, count_windows(8, WindowLayout{4, 4}));
    EXPECT_EQ(exact, 2u);

    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t ragged, count_windows(11, WindowLayout{4, 4}));
    EXPECT_EQ(ragged, 2u);

    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t one, count_windows(4, WindowLayout{4, 1}));
    EXPECT_EQ(one, 1u);

    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t none, count_windows(3, WindowLayout{4, 1}));
    EXPECT_EQ(none, 0u);
}

/** @brief Tests that a window or a stride below 1, and a window above the limit, are refused. */
TEST_P(Small_LanguageDatasetsWindowLayoutRefusal, ALayoutOutOfBoundsIsRefused)
{
    const LayoutCase& row = GetParam();

    SA_EXPECT_FAILURE(count_windows(10, row.layout), StatusCode::InvalidDescription,
                      row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageDatasetsWindowLayoutRefusal,
    ::testing::Values(
        LayoutCase{"WindowOfZero", WindowLayout{0, 1}, "count_windows", "window of 0 tokens"},
        LayoutCase{"StrideOfZero", WindowLayout{4, 0}, "stride of 0 tokens", ""},
        LayoutCase{"WindowAboveTheLimit", WindowLayout{K_MAX_WINDOW_TOKENS + 1, 1}, "window of",
                   ""}),
    [](const ::testing::TestParamInfo<LayoutCase>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.label);
    });
