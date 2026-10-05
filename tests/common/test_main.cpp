/****************************************************************/
/* test_main.cpp                                                */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
