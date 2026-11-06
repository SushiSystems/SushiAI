/**************************************************************************/
/* test_main.cpp                                                          */
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
 * @file test_main.cpp
 * @brief Defines the main() of a test program, which registers the shared TestEnvironment.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiLog/log/log_calls.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>

/** @brief Runs every registered test under the shared TestEnvironment. */
int main(int argc, char** argv)
{
    // Logging is synchronous so it cannot interleave with GTest's own output.
    SushiLog::set_log_synchronous(true);

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new SushiAI::Test::TestEnvironment());

    return RUN_ALL_TESTS();
}
