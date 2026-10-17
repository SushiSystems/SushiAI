/**************************************************************************/
/* test_error.cpp                                                         */
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
 * @file test_error.cpp
 * @brief Tests the Error type and SA_THROW_IF of error.hpp.
 * @author Mustafa Garip
 */

#include <SushiAI/failure/error.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>

namespace
{
    /** @brief Throws through SA_THROW_IF when @p fail holds, naming @p extent; counts the calls. */
    void throw_when(bool fail, int extent, int& reached)
    {
        SA_THROW_IF(fail, "extent {} is not positive", extent);
        ++reached;
    }
} // namespace

/** @brief Tests that Error is caught as a std::runtime_error and keeps its message. */
TEST(Small_Error, AnErrorIsARuntimeErrorWithItsMessage)
{
    static_assert(std::is_base_of<std::runtime_error, SushiAI::Error>::value,
                  "Error derives from std::runtime_error");

    try
    {
        throw SushiAI::Error("a shape mismatch");
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_EQ(std::string(error.what()), "a shape mismatch");
        return;
    }
    ADD_FAILURE() << "SushiAI::Error was not caught as std::runtime_error";
}

/** @brief Tests that SA_THROW_IF passes on a false condition and throws a formatted Error. */
TEST(Small_Error, ThrowIfThrowsAFormattedErrorOnlyWhenItsConditionHolds)
{
    int reached = 0;
    throw_when(false, 3, reached);
    EXPECT_EQ(reached, 1);

    SA_EXPECT_ERROR(throw_when(true, -2, reached), "extent -2 is not positive");
    EXPECT_EQ(reached, 1);
}
