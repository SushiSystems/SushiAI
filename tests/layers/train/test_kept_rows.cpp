/**************************************************************************/
/* test_kept_rows.cpp                                                     */
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
 * @file test_kept_rows.cpp
 * @brief Tests that KeptRows keeps every row by default and exactly the flagged rows otherwise.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/train/kept_rows.hpp>

using namespace SushiAI;
using Train::KeptRows;

/** @brief Tests that the default selection keeps every row and states none. */
TEST(Small_KeptRows, TheDefaultKeepsEveryRowWhateverItsIndex)
{
    const KeptRows rows;

    EXPECT_TRUE(rows.keeps(0));
    EXPECT_TRUE(rows.keeps(7));
    EXPECT_TRUE(rows.keeps(100000));
    EXPECT_EQ(rows.size(), 0);
    EXPECT_EQ(rows.count(), 0);
}

/** @brief Tests that a flag list keeps the rows marked 1 and drops the rows marked 0. */
TEST(Small_KeptRows, AFlagListKeepsExactlyTheRowsMarkedOne)
{
    const KeptRows rows(std::vector<std::uint8_t>{1, 0, 1, 1});

    EXPECT_TRUE(rows.keeps(0));
    EXPECT_FALSE(rows.keeps(1));
    EXPECT_TRUE(rows.keeps(2));
    EXPECT_TRUE(rows.keeps(3));
    EXPECT_EQ(rows.size(), 4);
    EXPECT_EQ(rows.count(), 3);
}

/** @brief Tests that any flag other than 0 keeps its row. */
TEST(Small_KeptRows, AnyFlagOtherThanZeroKeepsItsRow)
{
    const KeptRows rows(std::vector<std::uint8_t>{0, 7, 255});

    EXPECT_FALSE(rows.keeps(0));
    EXPECT_TRUE(rows.keeps(1));
    EXPECT_TRUE(rows.keeps(2));
    EXPECT_EQ(rows.count(), 2);
}

/** @brief Tests that a selection that drops every row keeps none and counts none. */
TEST(Small_KeptRows, ASelectionOfZeroFlagsKeepsNoRow)
{
    const KeptRows rows(std::vector<std::uint8_t>{0, 0, 0});

    EXPECT_FALSE(rows.keeps(0));
    EXPECT_FALSE(rows.keeps(2));
    EXPECT_EQ(rows.size(), 3);
    EXPECT_EQ(rows.count(), 0);
}
