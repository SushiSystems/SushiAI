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
 * @file test_test_environment.cpp
 * @brief Tests that the runtime TestEnvironment shares accepts a wait.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>

namespace
{
    using SushiAI::Test::TestEnvironment;
} // namespace

/** @brief Tests that the shared runtime accepts a wait with nothing submitted. */
TEST(Medium_TestEnvironment, TheRuntimeWaitsWithNothingSubmitted)
{
    EXPECT_TRUE(TestEnvironment::runtime().wait().has_value());
}
