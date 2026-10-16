/**************************************************************************/
/* test_region_names.cpp                                                  */
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
 * @file test_region_names.cpp
 * @brief Tests that RegionNames gives one id per name and direction, whatever the key.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_names.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using SushiAI::Eager::Direction;
using SushiAI::Eager::K_NO_REGION_ID;
using SushiAI::Eager::RegionID;
using SushiAI::Eager::RegionInfo;
using SushiAI::Eager::RegionNames;

/** @brief Tests that a name interned on two steps under two keys keeps its first id. */
TEST(Small_RegionNames, ANameKeepsItsIDWhateverItsKey)
{
    RegionNames names;

    const RegionID first = names.intern(RegionInfo{3, "head#0", Direction::Forward});
    const RegionID again = names.intern(RegionInfo{7, "head#0", Direction::Forward});

    EXPECT_EQ(first, again);
    EXPECT_EQ(names.size(), 1u);
}

/** @brief Tests that the forward and backward regions of one name get two ids. */
TEST(Small_RegionNames, TheTwoDirectionsOfANameAreTwoRegions)
{
    RegionNames names;

    const RegionID forward = names.intern(RegionInfo{1, "head#0", Direction::Forward});
    const RegionID backward = names.intern(RegionInfo{2, "head#0", Direction::Backward});

    EXPECT_NE(forward, backward);
    EXPECT_EQ(names.size(), 2u);
    EXPECT_EQ(names.find("head#0", Direction::Backward), backward);
}

/** @brief Tests that find() answers no id for a name never interned and gives none. */
TEST(Small_RegionNames, FindingAnUnknownNameGivesNoID)
{
    RegionNames names;
    (void)names.intern(RegionInfo{1, "encoder#0", Direction::Forward});

    EXPECT_EQ(names.find("decoder#0", Direction::Forward), K_NO_REGION_ID);
    EXPECT_EQ(names.find("encoder#0", Direction::Backward), K_NO_REGION_ID);
    EXPECT_EQ(names.size(), 1u);
}

/** @brief Tests that the label of an id is its name followed by its direction. */
TEST(Small_RegionNames, ALabelNamesTheRegionAndItsDirection)
{
    RegionNames names;

    const RegionID id = names.intern(RegionInfo{4, "head#0", Direction::Backward});

    EXPECT_EQ(names.label_of(id), "head#0 backward");
}

/** @brief Tests that asking the label of an id never given stops the process, naming the id. */
TEST(Small_RegionNames, TheLabelOfAnIDNeverGivenIsFatal)
{
    RegionNames names;
    (void)names.intern(RegionInfo{1, "head#0", Direction::Forward});

    SA_EXPECT_FATAL((void)names.label_of(5), "Region id 5", "never given");
}
