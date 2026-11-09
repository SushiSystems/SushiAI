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
 * @file test_reader_table.cpp
 * @brief Tests that every row of the reader table can be called and looked up exactly.
 * @author Mustafa Garip
 */
#include <cstddef>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/datasets/reader_table.hpp>

using namespace SushiAI::Vision::Datasets;

namespace
{
    /** @brief Runs a check over the index of each row of the reader table. */
    class Small_DatasetsReaderTableRows : public ::testing::TestWithParam<std::size_t>
    {
    };

    /** @brief Names a test after the reader of its row. */
    std::string row_name(const ::testing::TestParamInfo<std::size_t>& info)
    {
        return std::string(reader_table()[info.param].name);
    }

    /** @brief Returns each name that a second row of the reader table repeats. */
    std::vector<std::string> repeated_names()
    {
        std::set<std::string_view> seen;
        std::vector<std::string> repeated;
        for (const ReaderEntry& row : reader_table())
        {
            if (!seen.insert(row.name).second)
                repeated.emplace_back(row.name);
        }

        return repeated;
    }
} // namespace

/** @brief Tests that a row of the table does not leave its reader null. */
TEST_P(Small_DatasetsReaderTableRows, EveryRowPointsAtAReader)
{
    EXPECT_NE(reader_table()[GetParam()].read, nullptr);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsReaderTableRows,
                         ::testing::Range<std::size_t>(0, reader_table().size()), row_name);

/** @brief Tests that no two rows share a name, so a lookup has one answer. */
TEST(Small_DatasetsReaderTableLookup, NoTwoRowsShareAName)
{
    EXPECT_EQ(repeated_names(), std::vector<std::string>{});
}

/** @brief Tests that a lookup matches the whole name in its own case and nothing else. */
TEST(Small_DatasetsReaderTableLookup, ALookupMatchesTheWholeNameAndItsCase)
{
    ASSERT_NE(find_reader("yolo_txt"), nullptr);
    EXPECT_EQ(find_reader("YOLO_TXT"), nullptr);
    EXPECT_EQ(find_reader("yolo"), nullptr);
    EXPECT_EQ(find_reader("yolo_txt "), nullptr);
    EXPECT_EQ(find_reader(""), nullptr);
}
