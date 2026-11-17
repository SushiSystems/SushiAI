/**************************************************************************/
/* test_input_leaves.cpp                                                  */
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
 * @file test_input_leaves.cpp
 * @brief Tests that InputLeafList finds the leaf of a named extra input and lists them in order.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/input_leaves.hpp>

using namespace SushiAI;
using NN::InputLeaf;
using NN::InputLeafList;

/** @brief Tests that a list with no leaf finds no name. */
TEST(Small_InputLeafList, AnEmptyListFindsNoName)
{
    const InputLeafList list;

    EXPECT_EQ(list.find("lengths"), Graph::K_NO_VALUE);
    EXPECT_TRUE(list.leaves().empty());
}

/** @brief Tests that find returns the value declared for the name asked for. */
TEST(Small_InputLeafList, FindReturnsTheValueOfTheNamedLeaf)
{
    InputLeafList list;
    list.add(InputLeaf{"lengths", 7});
    list.add(InputLeaf{"options", 9});

    EXPECT_EQ(list.find("options"), 9u);
    EXPECT_EQ(list.find("lengths"), 7u);
}

/** @brief Tests that a name nobody declared finds no value. */
TEST(Small_InputLeafList, FindOfAnUndeclaredNameReturnsNoValue)
{
    InputLeafList list;
    list.add(InputLeaf{"lengths", 7});

    EXPECT_EQ(list.find("option"), Graph::K_NO_VALUE);
    EXPECT_EQ(list.find("lengths "), Graph::K_NO_VALUE);
}

/** @brief Tests that the leaves come back in the order they were added. */
TEST(Small_InputLeafList, LeavesComeBackInTheOrderTheyWereAdded)
{
    InputLeafList list;
    list.add(InputLeaf{"b", 2});
    list.add(InputLeaf{"a", 1});
    list.add(InputLeaf{"c", 3});

    ASSERT_EQ(list.leaves().size(), 3u);
    EXPECT_EQ(list.leaves()[0].name, "b");
    EXPECT_EQ(list.leaves()[1].name, "a");
    EXPECT_EQ(list.leaves()[2].name, "c");
    EXPECT_EQ(list.leaves()[2].value, 3u);
}
