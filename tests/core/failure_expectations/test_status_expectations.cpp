/**************************************************************************/
/* test_status_expectations.cpp                                           */
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
 * @file test_status_expectations.cpp
 * @brief Tests the assertions that check the code and the message of a Status.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace
{
    using SushiAI::Result;
    using SushiAI::Status;
    using SushiAI::StatusCode;
    using SushiAI::Test::death_pattern;
    using SushiAI::Test::require_value;
    using SushiAI::Test::status_is_ok;
    using SushiAI::Test::status_names;

    /** @brief Returns an InvalidFile failure whose message is "the header of 'a.bin' is short". */
    Status short_header()
    {
        return Status::from_message(StatusCode::InvalidFile, []()
        {
            return std::string("the header of 'a.bin' is short");
        });
    }
} // namespace

/** @brief Tests that status_is_ok accepts a success and refuses a failure, quoting its code. */
TEST(Small_StatusExpectations, StatusIsOkAcceptsASuccessAndQuotesAFailure)
{
    EXPECT_TRUE(status_is_ok(Status()));

    const ::testing::AssertionResult refused = status_is_ok(short_header());

    EXPECT_FALSE(refused);
    EXPECT_NE(std::string(refused.message()).find("is short"), std::string::npos);
}

/** @brief Tests that status_names accepts the stated code when every needle is in the message. */
TEST(Small_StatusExpectations, StatusNamesAcceptsTheCodeAndEveryNeedle)
{
    EXPECT_TRUE(status_names(short_header(), StatusCode::InvalidFile, {"a.bin", "short"}));
    EXPECT_TRUE(status_names(short_header(), StatusCode::InvalidFile, {}));
}

/** @brief Tests that status_names refuses a success, another code and a needle that is absent. */
TEST(Small_StatusExpectations, StatusNamesRefusesASuccessAnotherCodeAndAMissingNeedle)
{
    EXPECT_FALSE(status_names(Status(), StatusCode::InvalidFile, {}));
    EXPECT_FALSE(status_names(short_header(), StatusCode::FileNotFound, {}));

    const ::testing::AssertionResult missing =
        status_names(short_header(), StatusCode::InvalidFile, {"a.bin", "b.bin"});

    EXPECT_FALSE(missing);
    EXPECT_NE(std::string(missing.message()).find("b.bin"), std::string::npos);
}

/** @brief Tests that death_pattern joins needles with ".*" and escapes the special characters. */
TEST(Small_StatusExpectations, DeathPatternJoinsAndEscapesItsNeedles)
{
    EXPECT_EQ(death_pattern({"rank", "[2, 3]"}), "rank.*\\[2, 3\\]");
    EXPECT_EQ(death_pattern({"a.b", "(c)*"}), "a\\.b.*\\(c\\)\\*");
    EXPECT_EQ(death_pattern({}), "");
}

/** @brief Tests that the macros read a Status and a Result, and bind the value of a success. */
TEST(Small_StatusExpectations, TheMacrosReadAStatusAndAResult)
{
    SA_EXPECT_OK(Status());
    SA_EXPECT_FAILURE(short_header(), StatusCode::InvalidFile, "a.bin");
    SA_EXPECT_FAILURE(Result<int>(short_header()), StatusCode::InvalidFile);

    SA_ASSERT_OK_AND_ASSIGN(const int value, Result<int>(7));

    EXPECT_EQ(value, 7);
}

/** @brief Tests that require_value returns the value or the reference a Result holds. */
TEST(Small_StatusExpectations, RequireValueReturnsWhatTheResultHolds)
{
    EXPECT_EQ(require_value(Result<int>(7)), 7);

    int target = 3;
    int& held = require_value(Result<int&>(target));
    EXPECT_EQ(&held, &target);
}

/** @brief Tests that require_value stops the process with the message of a failure. */
TEST(Small_StatusExpectations, RequireValueStopsTheProcessOnAFailure)
{
    SA_EXPECT_FATAL((void)require_value(Result<int>(short_header())), "fixture is refused",
                    "the header of 'a.bin' is short");
}
