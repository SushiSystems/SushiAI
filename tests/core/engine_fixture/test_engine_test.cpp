/**************************************************************************/
/* test_engine_test.cpp                                                   */
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
 * @file test_engine_test.cpp
 * @brief Tests that EngineTest gives each test an engine and the backend over it.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/engine_fixture/engine_test.hpp>

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Runs the cases on the fixture under test. */
    class Medium_EngineTest : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that SetUp leaves the case an engine. */
TEST_F(Medium_EngineTest, SetUpGivesTheCaseAnEngine)
{
    EXPECT_NE(engine, nullptr);
}

/** @brief Tests that backend hands out the engine of the case. */
TEST_F(Medium_EngineTest, BackendRunsOnTheEngineOfTheCase)
{
    EXPECT_EQ(&backend().engine(), engine);
}
