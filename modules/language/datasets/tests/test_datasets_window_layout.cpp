/**************************************************************************/
/* test_datasets_window_layout.cpp                                        */
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
 * @file test_datasets_window_layout.cpp
 * @brief Tests where the fixed-length windows of a token file begin.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/core/status.hpp>
#include <SushiAI/language/datasets/window_layout.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Datasets;

/** @brief Tests that 10 ids hold three windows of 4 at a stride of 3, at 0, 3 and 6. */
TEST(Unit_LanguageDatasetsWindowLayout, WindowsBeginOneStrideApart)
{
    const WindowLayout layout{4, 3};

    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t windows, count_windows(10, layout));
    EXPECT_EQ(windows, 3u);
    EXPECT_EQ(first_token_of(0, layout), 0u);
    EXPECT_EQ(first_token_of(1, layout), 3u);
    EXPECT_EQ(first_token_of(2, layout), 6u);
}

/** @brief Tests that a window whose last id would be past the file is not counted. */
TEST(Unit_LanguageDatasetsWindowLayout, OnlyWholeWindowsAreCounted)
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
TEST(Unit_LanguageDatasetsWindowLayout, ALayoutOutOfBoundsIsRefused)
{
    SA_EXPECT_FAILURE(count_windows(10, WindowLayout{0, 1}), StatusCode::InvalidDescription,
                      "count_windows", "window of 0 tokens");
    SA_EXPECT_FAILURE(count_windows(10, WindowLayout{4, 0}), StatusCode::InvalidDescription,
                      "stride of 0 tokens");
    SA_EXPECT_FAILURE(count_windows(10, WindowLayout{K_MAX_WINDOW_TOKENS + 1, 1}),
                      StatusCode::InvalidDescription, "window of");
}
