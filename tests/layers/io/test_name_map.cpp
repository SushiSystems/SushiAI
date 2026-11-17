/**************************************************************************/
/* test_name_map.cpp                                                      */
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
 * @file test_name_map.cpp
 * @brief Tests the reading of a name map from JSON and the drop and rename queries it answers.
 * @author Mustafa Garip
 */

#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/io/name_map.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Names the text in every message. */
    constexpr std::string_view K_ORIGIN = "map.json";
} // namespace

/** @brief Tests that the state key, the rules in file order and the drops are all read. */
TEST(Small_NameMap, ReadsTheStateKeyTheRulesAndTheDrops)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::NameMap map,
        IO::NameMap::parse(R"({"state_dict":"model",)"
                           R"("rename":[{"from":"backbone.","to":"body."},{"from":"","to":"m."}],)"
                           R"("drop":["num_batches_tracked"]})",
                           K_ORIGIN));

    EXPECT_EQ(map.state_key(), "model");
    const std::vector<IO::RenameRule> rules = map.rename_rules();
    ASSERT_EQ(rules.size(), 2u);
    EXPECT_EQ(rules[0].from_prefix, "backbone.");
    EXPECT_EQ(rules[0].to_prefix, "body.");
    EXPECT_EQ(rules[1].from_prefix, "");
    EXPECT_EQ(rules[1].to_prefix, "m.");
    // The first match wins: "backbone.x" takes the first rule, not the second's empty prefix.
    EXPECT_EQ(IO::renamed("backbone.x", rules), "body.x");
    EXPECT_EQ(IO::renamed("head.x", rules), "m.head.x");
}

/** @brief Tests that the empty object is a map that renames and drops nothing and names no key. */
TEST(Small_NameMap, AnEmptyObjectChangesNothing)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::NameMap map, IO::NameMap::parse("{}", K_ORIGIN));

    EXPECT_TRUE(map.state_key().empty());
    EXPECT_TRUE(map.rename_rules().empty());
    EXPECT_FALSE(map.is_dropped("anything"));
}

/** @brief Tests that a drop entry matches the whole name or its last dotted components only. */
TEST(Small_NameMap, ADropEntryMatchesTheNameOrADottedSuffix)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::NameMap map,
        IO::NameMap::parse(R"({"drop":["num_batches_tracked","bn.total"]})", K_ORIGIN));

    EXPECT_TRUE(map.is_dropped("num_batches_tracked"));
    EXPECT_TRUE(map.is_dropped("layer1.0.bn1.num_batches_tracked"));
    EXPECT_TRUE(map.is_dropped("layer1.bn.total"));
    // A longer last component that merely ends in the entry is another tensor.
    EXPECT_FALSE(map.is_dropped("x_num_batches_tracked"));
    EXPECT_FALSE(map.is_dropped("xbn.total"));
    EXPECT_FALSE(map.is_dropped("num_batches_tracked.weight"));
}

/** @brief Tests that a root that is no object is refused, naming the kind it is. */
TEST(Small_NameMap, RefusesARootThatIsNoObject)
{
    SA_EXPECT_FAILURE(IO::NameMap::parse("[1]", K_ORIGIN), StatusCode::InvalidDescription,
                      "map.json", "not an object");
}

/** @brief Tests that a key outside state_dict, rename and drop is refused, naming it. */
TEST(Small_NameMap, RefusesAnUnknownKey)
{
    SA_EXPECT_FAILURE(IO::NameMap::parse(R"({"renames":[]})", K_ORIGIN),
                      StatusCode::InvalidDescription, "renames");
}

/** @brief Tests that an empty state_dict is refused, because leaving the key out says the same. */
TEST(Small_NameMap, RefusesAnEmptyStateKey)
{
    SA_EXPECT_FAILURE(IO::NameMap::parse(R"({"state_dict":""})", K_ORIGIN),
                      StatusCode::InvalidDescription, "state_dict", "empty");
}

/** @brief Tests that a rename rule missing its `to`, or holding another key, names its index. */
TEST(Small_NameMap, RefusesARenameRuleThatIsNotFromAndTo)
{
    SA_EXPECT_FAILURE(IO::NameMap::parse(R"({"rename":[{"from":"a."}]})", K_ORIGIN),
                      StatusCode::InvalidDescription, "rename rule 0", "to");
    SA_EXPECT_FAILURE(
        IO::NameMap::parse(R"({"rename":[{"from":"a.","to":"b."},{"from":"c.","to":"d.","x":1}]})",
                           K_ORIGIN),
        StatusCode::InvalidDescription, "rename rule 1", "x");
    SA_EXPECT_FAILURE(IO::NameMap::parse(R"({"rename":[{"from":"a.","to":3}]})", K_ORIGIN),
                      StatusCode::InvalidDescription, "rename rule 0", "'to'");
}

/** @brief Tests that an empty drop entry, which would match no tensor, is refused. */
TEST(Small_NameMap, RefusesAnEmptyDropEntry)
{
    SA_EXPECT_FAILURE(IO::NameMap::parse(R"({"drop":["a",""]})", K_ORIGIN),
                      StatusCode::InvalidDescription, "drop entry 1", "empty");
}

/** @brief Tests that text that is no JSON is refused as an invalid file, naming the origin. */
TEST(Small_NameMap, RefusesTextThatIsNoJson)
{
    SA_EXPECT_FAILURE(IO::NameMap::parse(R"({"drop":)", K_ORIGIN), StatusCode::InvalidFile,
                      "map.json");
}
