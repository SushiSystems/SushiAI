/**************************************************************************/
/* test_throw_expectations.cpp                                            */
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
 * @file test_throw_expectations.cpp
 * @brief Tests the assertions that check what the message of a thrown Error names.
 * @author Mustafa Garip
 */

#include <stdexcept>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/error.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>

namespace
{
    using SushiAI::Error;
    using SushiAI::Test::capture_error;
    using SushiAI::Test::error_names;
    using SushiAI::Test::ThrowOutcome;

    /** @brief Throws an Error whose message names a layer and a rank. */
    void throw_rank_error()
    {
        throw Error("layer 'conv1' expects rank 4");
    }

    /** @brief Throws a standard exception that is not an Error. */
    void throw_out_of_range()
    {
        throw std::out_of_range("index 9");
    }

    /** @brief Returns without throwing. */
    void do_nothing()
    {

    }
} // namespace

/** @brief Tests that capture_error records an Error and its message. */
TEST(Small_ThrowExpectations, CaptureErrorRecordsAnErrorAndItsMessage)
{
    const ThrowOutcome outcome = capture_error(throw_rank_error);

    EXPECT_TRUE(outcome.threw_error);
    EXPECT_FALSE(outcome.threw_other);
    EXPECT_EQ(outcome.message, "layer 'conv1' expects rank 4");
}

/** @brief Tests that capture_error tells a foreign exception from an Error and from no throw. */
TEST(Small_ThrowExpectations, CaptureErrorTellsAForeignExceptionFromNoThrow)
{
    const ThrowOutcome foreign = capture_error(throw_out_of_range);
    const ThrowOutcome quiet = capture_error(do_nothing);

    EXPECT_FALSE(foreign.threw_error);
    EXPECT_TRUE(foreign.threw_other);
    EXPECT_EQ(foreign.other_what, "index 9");
    EXPECT_FALSE(quiet.threw_error);
    EXPECT_FALSE(quiet.threw_other);
}

/** @brief Tests that error_names accepts an Error naming every needle and refuses the rest. */
TEST(Small_ThrowExpectations, ErrorNamesAcceptsEveryNeedleAndRefusesTheRest)
{
    const ThrowOutcome outcome = capture_error(throw_rank_error);

    EXPECT_TRUE(error_names(outcome, {"conv1", "rank 4"}));
    EXPECT_FALSE(error_names(outcome, {"conv2"}));
    EXPECT_FALSE(error_names(capture_error(do_nothing), {}));
    EXPECT_FALSE(error_names(capture_error(throw_out_of_range), {}));
}

/** @brief Tests that SA_EXPECT_ERROR passes for a statement that throws a naming Error. */
TEST(Small_ThrowExpectations, TheMacroPassesForANamingError)
{
    SA_EXPECT_ERROR(throw_rank_error(), "conv1", "rank 4");
}
