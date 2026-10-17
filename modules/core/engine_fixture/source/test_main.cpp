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
