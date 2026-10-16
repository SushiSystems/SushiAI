/**************************************************************************/
/* test_test_environment.cpp                                              */
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
 * @file test_test_environment.cpp
 * @brief Tests that TestEnvironment hands every caller the one runtime it creates.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>

namespace
{
    using SushiAI::Test::TestEnvironment;
} // namespace

/** @brief Tests that two calls to runtime() return the same runtime. */
TEST(Medium_TestEnvironment, EveryCallReturnsTheSameRuntime)
{
    SushiRuntime::Runtime& first = TestEnvironment::runtime();
    SushiRuntime::Runtime& second = TestEnvironment::runtime();

    EXPECT_EQ(&first, &second);
}

/** @brief Tests that the shared runtime accepts a wait with nothing submitted. */
TEST(Medium_TestEnvironment, TheRuntimeWaitsWithNothingSubmitted)
{
    EXPECT_TRUE(TestEnvironment::runtime().wait().has_value());
}
