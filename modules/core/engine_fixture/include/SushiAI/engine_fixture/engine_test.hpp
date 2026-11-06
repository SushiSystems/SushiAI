/**************************************************************************/
/* engine_test.hpp                                                        */
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
 * @file engine_test.hpp
 * @brief Declares EngineTest, the fixture that gives each test an Engine on the shared Runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Provides a fresh Engine per test, on the shared Runtime. */
        class EngineTest : public ::testing::Test
        {
            protected:
                /** @brief Builds the test's Engine on the shared runtime. */
                void SetUp() override
                {
                    engine = std::make_unique<SushiBLAS::Engine>(TestEnvironment::runtime());
                }

                /** @brief Destroys the test's Engine. */
                void TearDown() override
                {
                    engine.reset();
                }

                /** @brief Waits for the device; a test runs the graph it records on itself. */
                void flush()
                {
                    TestEnvironment::runtime().wait().value();
                }

                /** @brief Holds the Engine the test records on. */
                std::unique_ptr<SushiBLAS::Engine> engine;
        };

    } // namespace Test
} // namespace SushiAI
