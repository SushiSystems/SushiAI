/**************************************************************************/
/* test_zone_groups.cpp                                                   */
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
 * @file test_zone_groups.cpp
 * @brief Tests that K_ZONE_GROUPS is the one list of zone groups and that SA_ZONE refuses others.
 * @author Mustafa Garip
 * @see tests/core/profile/test_zone_groups.md
 */

#include <SushiAI/profile/profile.hpp>
#include <SushiAI/profile/zone_groups.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>
#include <gtest/gtest.h>

namespace Profile = SushiAI::Profile;

// Each group of K_ZONE_GROUPS passes the check SA_ZONE runs, or this file does not compile.
SA_REQUIRE_ZONE_GROUP("compile");
SA_REQUIRE_ZONE_GROUP("copy");
SA_REQUIRE_ZONE_GROUP("data");
SA_REQUIRE_ZONE_GROUP("device");
SA_REQUIRE_ZONE_GROUP("eager");
SA_REQUIRE_ZONE_GROUP("eval");
SA_REQUIRE_ZONE_GROUP("host");
SA_REQUIRE_ZONE_GROUP("io");
SA_REQUIRE_ZONE_GROUP("optim");
SA_REQUIRE_ZONE_GROUP("train");

static_assert(!Profile::is_zone_group("Runtime"), "a library group is not a SushiAI group");

/** @brief Tests that the list holds the ten groups of the profiling specification, in order. */
TEST(Small_ZoneGroups, TheListHoldsTheSpecifiedGroups)
{
    const std::array<std::string_view, 10> expected = {
        "compile", "copy", "data", "device", "eager", "eval", "host", "io", "optim", "train"};

    EXPECT_EQ(Profile::K_ZONE_GROUPS, expected);
}

/** @brief Tests that no group is empty or listed twice. */
TEST(Small_ZoneGroups, NoGroupIsEmptyOrRepeated)
{
    for (std::size_t index = 0; index < Profile::K_ZONE_GROUPS.size(); ++index)
    {
        EXPECT_FALSE(Profile::K_ZONE_GROUPS[index].empty());
        const auto first = Profile::K_ZONE_GROUPS.begin();
        EXPECT_EQ(std::count(first, Profile::K_ZONE_GROUPS.end(), Profile::K_ZONE_GROUPS[index]), 1)
            << Profile::K_ZONE_GROUPS[index];
    }
}

/** @brief Tests that is_zone_group accepts each listed group and refuses a near miss. */
TEST(Small_ZoneGroups, IsZoneGroupAcceptsOnlyTheListedGroups)
{
    for (const std::string_view group : Profile::K_ZONE_GROUPS)
        EXPECT_TRUE(Profile::is_zone_group(group)) << group;

    EXPECT_FALSE(Profile::is_zone_group(""));
    EXPECT_FALSE(Profile::is_zone_group("hos"));
    EXPECT_FALSE(Profile::is_zone_group("hosts"));
    EXPECT_FALSE(Profile::is_zone_group("Host"));
    EXPECT_FALSE(Profile::is_zone_group("BLAS"));
}
