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
 * @file test_zone_groups.cpp
 * @brief Tests that K_ZONE_GROUPS is the one list of zone groups and that SA_ZONE refuses others.
 * @author Mustafa Garip
 * @see tests/core/profile/test_zone_groups.md
 */

#include <SushiAI/profile/profile.hpp>
#include <SushiAI/profile/zone_groups.hpp>

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <gtest/gtest.h>

namespace Profile = SushiAI::Profile;

namespace
{
    /** @brief Lists the names that are close to a group but are not one, and must be refused. */
    constexpr std::array<std::string_view, 5> K_NEAR_MISSES = {"", "hos", "hosts", "Host", "BLAS"};

    /** @brief Runs one case per listed group of K_ZONE_GROUPS. */
    class Small_ZoneGroupEntry : public ::testing::TestWithParam<std::string_view>
    {

    };

    /** @brief Runs one case per name of K_NEAR_MISSES. */
    class Small_ZoneGroupNearMiss : public ::testing::TestWithParam<std::string_view>
    {

    };

    /** @brief Names a case after its position in its table. */
    std::string case_index_name(const ::testing::TestParamInfo<std::string_view>& info)
    {
        return std::to_string(info.index);
    }
} // namespace

INSTANTIATE_TEST_SUITE_P(, Small_ZoneGroupEntry, ::testing::ValuesIn(Profile::K_ZONE_GROUPS),
                         case_index_name);

INSTANTIATE_TEST_SUITE_P(, Small_ZoneGroupNearMiss, ::testing::ValuesIn(K_NEAR_MISSES),
                         case_index_name);

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

/**
 * @brief Tests the contract: K_ZONE_GROUPS lists the ten groups of the profiling specification,
 *        in the specified order.
 */
TEST(Small_ZoneGroups, TheContractListsTheSpecifiedGroupsInOrder)
{
    const std::array<std::string_view, 10> expected = {
        "compile", "copy", "data", "device", "eager", "eval", "host", "io", "optim", "train"};

    EXPECT_EQ(Profile::K_ZONE_GROUPS, expected);
}

/** @brief Tests that a listed group is not empty and occurs once in K_ZONE_GROUPS. */
TEST_P(Small_ZoneGroupEntry, IsNonEmptyAndListedOnce)
{
    const std::string_view group = GetParam();

    EXPECT_FALSE(group.empty());
    EXPECT_EQ(std::count(Profile::K_ZONE_GROUPS.begin(), Profile::K_ZONE_GROUPS.end(), group), 1)
        << group;
}

/** @brief Tests that is_zone_group accepts each listed group. */
TEST_P(Small_ZoneGroupEntry, IsZoneGroupAcceptsIt)
{
    EXPECT_TRUE(Profile::is_zone_group(GetParam())) << GetParam();
}

/** @brief Tests that is_zone_group refuses a name that is close to a group but is not one. */
TEST_P(Small_ZoneGroupNearMiss, IsZoneGroupRefusesIt)
{
    EXPECT_FALSE(Profile::is_zone_group(GetParam())) << GetParam();
}
