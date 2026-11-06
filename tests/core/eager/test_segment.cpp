/**************************************************************************/
/* test_segment.cpp                                                       */
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
 * @file test_segment.cpp
 * @brief Tests the default value of Segment and the half-open node range it names.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/graph/ir.hpp>

using SushiAI::Eager::Segment;

/** @brief Tests that a default segment is unnamed, first of its name and holds no node. */
TEST(Small_Segment, ADefaultSegmentIsUnnamedAndEmpty)
{
    const Segment segment;

    EXPECT_TRUE(segment.name.empty());
    EXPECT_EQ(segment.occurrence, 0u);
    EXPECT_EQ(segment.begin, SushiAI::Graph::NodeID{0});
    EXPECT_EQ(segment.end, segment.begin);
}

/** @brief Tests that a segment's node count is end minus begin, as its range is half open. */
TEST(Small_Segment, TheRangeExcludesItsEnd)
{
    const Segment segment{"encoder/block", 2, 4, 9};

    EXPECT_EQ(segment.name, "encoder/block");
    EXPECT_EQ(segment.occurrence, 2u);
    EXPECT_EQ(segment.end - segment.begin, SushiAI::Graph::NodeID{5});
}
