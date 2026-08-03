/**************************************************************************/
/* test_main.cpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

// A custom main, so the RuntimeContext is built once for the whole binary
// rather than once per test. GTest::gtest_main is deliberately NOT linked in
// tests/CMakeLists.txt - linking both is a duplicate-symbol error at link
// time, which is the good failure mode, but the reason is worth stating where
// the replacement lives.

#include <gtest/gtest.h>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/core/logger.hpp>
#include "test_environment.hpp"

int main(int argc, char** argv)
{
    // Asynchronous logging interleaves with GTest's own output and makes a
    // failure report unreadable; the sibling suites make the same choice.
    SR_LOGGER_SET_SYNC_MODE(true);
    SB_LOGGER_SET_SYNC_MODE(true);

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new SushiAI::Test::TestEnvironment());

    return RUN_ALL_TESTS();
}
