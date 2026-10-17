/**************************************************************************/
/* test_check.cpp                                                         */
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
 * @file test_check.cpp
 * @brief Tests that the fatal checks of check.hpp expand to one statement each.
 * @author Mustafa Garip
 */

#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/failure/check.hpp>

namespace
{
    /** @brief Returns which branch of an unbraced if-else around SA_FATAL_IF ran. */
    int branch_around_fatal_if(bool take_first)
    {
        int branch = 0;
        if (take_first)
            SA_FATAL_IF(false, "never reported");
        else
            branch = 2;

        if (branch == 0)
            branch = 1;
        return branch;
    }

    /** @brief Returns which branch of an unbraced if-else around SA_ASSERT ran. */
    int branch_around_assert(bool take_first)
    {
        int branch = 0;
        if (take_first)
            SA_ASSERT(branch == 0);
        else
            branch = 2;

        if (branch == 0)
            branch = 1;
        return branch;
    }
} // namespace

/** @brief Tests that SA_FATAL_IF is one statement an unbraced if-else can hold. */
TEST(Small_Check, FatalIfIsOneStatementInAnUnbracedIfElse)
{
    EXPECT_EQ(branch_around_fatal_if(true), 1);
    EXPECT_EQ(branch_around_fatal_if(false), 2);
}

/** @brief Tests that SA_ASSERT is one statement an unbraced if-else can hold. */
TEST(Small_Check, AssertIsOneStatementInAnUnbracedIfElse)
{
    EXPECT_EQ(branch_around_assert(true), 1);
    EXPECT_EQ(branch_around_assert(false), 2);
}

/** @brief Tests that the text of an uncomposed report names a fatal check. */
TEST(Small_Check, TheUncomposedReportSaysAFatalCheckFailed)
{
    const std::string_view text = SushiAI::Detail::K_UNCOMPOSED_FATAL_MESSAGE;

    EXPECT_NE(text.find("fatal check failed"), std::string_view::npos);
}
