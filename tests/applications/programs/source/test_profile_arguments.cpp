/**************************************************************************/
/* test_profile_arguments.cpp                                             */
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
 * @file test_profile_arguments.cpp
 * @brief Tests that read_profile_arguments takes the --profile pair out of a command line.
 * @author Mustafa Garip
 * @see tests/applications/programs/README.md
 */

#include "profile_arguments.hpp"

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using Apps::ProfileArguments;
using Apps::read_profile_arguments;

namespace
{
    /** @brief Reads @p words as a command line without its program name. */
    Result<ProfileArguments> read(const std::vector<const char*>& words)
    {
        return read_profile_arguments(
            SushiRuntime::span<const char* const>(words.data(), words.size()));
    }
} // namespace

/** @brief Tests that a command line without the flag comes back whole. */
TEST(Small_ProfileArguments, ALineWithoutTheFlagIsKept)
{
    SA_ASSERT_OK_AND_ASSIGN(const ProfileArguments read_back, read({"train.json"}));

    EXPECT_EQ(read_back.positional, std::vector<std::string>{"train.json"});
    EXPECT_FALSE(read_back.config.has_value());
}

/** @brief Holds a command line with the pair somewhere in it and what is left of it. */
struct PairCase
{
    /** @brief Names the case. */
    const char* name;

    /** @brief The command line without its program name. */
    std::vector<const char*> words;
};

/** @brief Holds a command line that is refused and the text its refusal carries. */
struct RefusalCase
{
    /** @brief Names the case. */
    const char* name;

    /** @brief The command line without its program name. */
    std::vector<const char*> words;

    /** @brief A part of the refusal's message. */
    const char* message;
};

/** @brief Runs the rows of command lines that carry one --profile pair. */
class Small_ProfileArgumentsPosition : public ::testing::TestWithParam<PairCase> {};

/** @brief Runs the rows of command lines that are refused. */
class Small_ProfileArgumentsRefusal : public ::testing::TestWithParam<RefusalCase> {};

/** @brief Tests that the pair is taken out wherever it stands. */
TEST_P(Small_ProfileArgumentsPosition, ThePairIsTakenOutWhereverItStands)
{
    SA_ASSERT_OK_AND_ASSIGN(const ProfileArguments line, read(GetParam().words));

    EXPECT_EQ(line.positional, (std::vector<std::string>{"m.json", "w.sushi", "e.json"}));
    ASSERT_TRUE(line.config.has_value());
    EXPECT_EQ(line.config->string(), "p.json");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ProfileArgumentsPosition,
    ::testing::Values(
        PairCase{"BeforeThePositionals", {"--profile", "p.json", "m.json", "w.sushi", "e.json"}},
        PairCase{"BetweenThePositionals", {"m.json", "--profile", "p.json", "w.sushi", "e.json"}},
        PairCase{"AfterThePositionals", {"m.json", "w.sushi", "e.json", "--profile", "p.json"}}),
    [](const ::testing::TestParamInfo<PairCase>& info)
    {
        return std::string(info.param.name);
    });

/** @brief Tests that a flag with no path, a flag in place of the path and a repeat are refused. */
TEST_P(Small_ProfileArgumentsRefusal, ALineThatBreaksTheFlagIsRefused)
{
    SA_EXPECT_FAILURE(read(GetParam().words), StatusCode::InvalidDescription,
                      GetParam().message);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ProfileArgumentsRefusal,
    ::testing::Values(
        RefusalCase{"AFlagWithNoPath", {"train.json", "--profile"}, "needs the path"},
        RefusalCase{"AFlagInPlaceOfThePath", {"--profile", "--help"}, "needs the path"},
        RefusalCase{"AFlagGivenTwice", {"--profile", "a.json", "--profile", "b.json"},
                    "given twice"}),
    [](const ::testing::TestParamInfo<RefusalCase>& info)
    {
        return std::string(info.param.name);
    });
