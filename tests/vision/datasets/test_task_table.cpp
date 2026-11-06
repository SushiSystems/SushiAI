/**************************************************************************/
/* test_task_table.cpp                                                    */
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
 * @file test_task_table.cpp
 * @brief Tests that every row of the task table can be called and looked up exactly.
 * @author Mustafa Garip
 */

#include <set>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/vision/datasets/task_table.hpp>

using namespace SushiAI::Vision::Datasets;

/** @brief Tests that no row of the table leaves its collator factory or its builder null. */
TEST(Small_DatasetsTaskTableRows, EveryRowPointsAtAFactoryAndABuilder)
{
    for (const TaskEntry& row : task_table())
    {
        EXPECT_NE(row.make_collator, nullptr) << row.name;
        EXPECT_NE(row.build, nullptr) << row.name;
    }
}

/** @brief Tests that no two rows share a name, so a lookup has one answer. */
TEST(Small_DatasetsTaskTableRows, NoTwoRowsShareAName)
{
    std::set<std::string_view> names;
    for (const TaskEntry& row : task_table())
        EXPECT_TRUE(names.insert(row.name).second) << row.name;
}

/** @brief Tests that a lookup matches the whole name in its own case and nothing else. */
TEST(Small_DatasetsTaskTableRows, ALookupMatchesTheWholeNameAndItsCase)
{
    ASSERT_NE(find_task("detection"), nullptr);
    EXPECT_EQ(find_task("Detection"), nullptr);
    EXPECT_EQ(find_task("detect"), nullptr);
    EXPECT_EQ(find_task("detection "), nullptr);
    EXPECT_EQ(find_task(""), nullptr);
}
