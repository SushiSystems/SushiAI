/**************************************************************************/
/* test_pickle_data.cpp                                                   */
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
 * @file test_pickle_data.cpp
 * @brief Tests the conversion of plain pickle data into a configuration value.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/pickle_data.hpp>
#include <SushiAI/io/pickle_reader.hpp>
#include "archive_bytes.hpp"

using namespace SushiAI;
using namespace std::string_view_literals;

namespace
{
    /** @brief Lists the one global the tests accept. */
    constexpr std::string_view K_ACCEPTED[] = {"torch Size"};

    /** @brief Returns the tree of @p stream, failing the test when it does not read. */
    IO::PickleRef tree_of(std::string_view stream)
    {
        const std::vector<std::byte> bytes = SushiAI::Test::bytes_of(stream);
        Result<IO::PickleRef> read = IO::read_pickle(SushiAI::Test::view_of(bytes), K_ACCEPTED, "data");
        EXPECT_TRUE(read.has_value());
        return read.has_value() ? read.value() : IO::PickleRef();
    }
} // namespace

/** @brief Tests that a dict of a list, a tuple and a whole-number key converts to its shape. */
TEST(Small_PickleData, ConvertsADictOfAListATupleAndAWholeNumberKey)
{
    // {'n': [1, None, True], 0: (2.5,)}; 2.5 is 0x4004000000000000 big-endian.
    const IO::PickleRef tree = tree_of("}("
                                       "X\x01\x00\x00\x00" "n" "(K\x01N\x88l"
                                       "K\x00" "G\x40\x04\x00\x00\x00\x00\x00\x00\x85"
                                       "u."sv);
    ASSERT_NE(tree, nullptr);

    SA_ASSERT_OK_AND_ASSIGN(const Config::Value value, IO::pickle_to_value(*tree, "data"));

    ASSERT_EQ(value.keys().size(), 2u);
    EXPECT_EQ(value.keys()[1], "0");
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& list, value.at("n", "data"));
    ASSERT_EQ(list.as_array("n").value().size(), 3u);
    EXPECT_EQ(list.as_array("n").value()[0].as_int("n").value(), 1);
    EXPECT_TRUE(list.as_array("n").value()[1].is_null());
    EXPECT_TRUE(list.as_array("n").value()[2].as_bool("n").value());
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& tuple, value.at("0", "data"));
    EXPECT_EQ(tuple.as_array("0").value()[0].as_real("0").value(), 2.5);
}

/** @brief Tests that a call inside the data is refused, naming where it sits. */
TEST(Small_PickleData, RefusesACallNamingItsPath)
{
    // {'shape': torch.Size(())}
    const IO::PickleRef tree =
        tree_of("}X\x05\x00\x00\x00" "shape" "ctorch\nSize\n)\x85R" "s."sv);
    ASSERT_NE(tree, nullptr);

    SA_EXPECT_FAILURE(IO::pickle_to_value(*tree, "data"), StatusCode::InvalidDescription,
                      "'.shape'", "not plain data");
}
