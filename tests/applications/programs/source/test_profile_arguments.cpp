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

/** @brief Tests that the pair is taken out wherever it stands. */
TEST(Small_ProfileArguments, ThePairIsTakenOutWhereverItStands)
{
    SA_ASSERT_OK_AND_ASSIGN(const ProfileArguments before,
                            read({"--profile", "p.json", "m.json", "w.sushi", "e.json"}));
    SA_ASSERT_OK_AND_ASSIGN(const ProfileArguments between,
                            read({"m.json", "--profile", "p.json", "w.sushi", "e.json"}));
    SA_ASSERT_OK_AND_ASSIGN(const ProfileArguments after,
                            read({"m.json", "w.sushi", "e.json", "--profile", "p.json"}));

    const std::vector<std::string> rest = {"m.json", "w.sushi", "e.json"};
    for (const ProfileArguments* line : {&before, &between, &after})
    {
        EXPECT_EQ(line->positional, rest);
        ASSERT_TRUE(line->config.has_value());
        EXPECT_EQ(line->config->string(), "p.json");
    }
}

/** @brief Tests that a flag with no path, a flag in place of the path and a repeat are refused. */
TEST(Small_ProfileArguments, AFlagWithNoPathOrGivenTwiceIsRefused)
{
    SA_EXPECT_FAILURE(read({"train.json", "--profile"}), StatusCode::InvalidDescription,
                      "--profile", "needs the path");
    SA_EXPECT_FAILURE(read({"--profile", "--help"}), StatusCode::InvalidDescription,
                      "needs the path");
    SA_EXPECT_FAILURE(read({"--profile", "a.json", "--profile", "b.json"}),
                      StatusCode::InvalidDescription, "given twice");
}
