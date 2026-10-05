/****************************************************************/
/* test_main.cpp                                                */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_main.cpp
 * @brief Defines the test binary's main(), which registers the shared TestEnvironment.
 * @author Mustafa Garip
 * @see tests/common/README.md, "test_main.cpp".
 */

#include <gtest/gtest.h>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/core/logger.hpp>
#include "test_environment.hpp"

/** @brief Runs every registered test under the shared TestEnvironment. */
int main(int argc, char** argv)
{
    // Logging is synchronous so it cannot interleave with GTest's own output.
    SR_LOGGER_SET_SYNC_MODE(true);
    SB_LOGGER_SET_SYNC_MODE(true);

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new SushiAI::Test::TestEnvironment());

    return RUN_ALL_TESTS();
}
