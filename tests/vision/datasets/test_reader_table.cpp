/**************************************************************************/
/* test_reader_table.cpp                                                  */
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
 * @file test_reader_table.cpp
 * @brief Tests that every row of the reader table can be called and looked up exactly.
 * @author Mustafa Garip
 */

#include <set>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/vision/datasets/reader_table.hpp>

using namespace SushiAI::Vision::Datasets;

/** @brief Tests that no row of the table leaves its reader null. */
TEST(Small_DatasetsReaderTableRows, EveryRowPointsAtAReader)
{
    for (const ReaderEntry& row : reader_table())
        EXPECT_NE(row.read, nullptr) << row.name;
}

/** @brief Tests that no two rows share a name, so a lookup has one answer. */
TEST(Small_DatasetsReaderTableRows, NoTwoRowsShareAName)
{
    std::set<std::string_view> names;
    for (const ReaderEntry& row : reader_table())
        EXPECT_TRUE(names.insert(row.name).second) << row.name;
}

/** @brief Tests that a lookup matches the whole name in its own case and nothing else. */
TEST(Small_DatasetsReaderTableRows, ALookupMatchesTheWholeNameAndItsCase)
{
    ASSERT_NE(find_reader("yolo_txt"), nullptr);
    EXPECT_EQ(find_reader("YOLO_TXT"), nullptr);
    EXPECT_EQ(find_reader("yolo"), nullptr);
    EXPECT_EQ(find_reader("yolo_txt "), nullptr);
    EXPECT_EQ(find_reader(""), nullptr);
}
