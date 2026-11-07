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
 * @brief Declares EngineTest, the fixture that gives each test the one Engine of the program.
 * @author Mustafa Garip
 */

#pragma once

#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/execution/runtime_execution_backend.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Gives every test the one Engine and backend of the program, rewound to fresh. */
        class EngineTest : public ::testing::Test
        {
            protected:
                /** @brief Points the test at the program's Engine and rewinds its mode and random stream. */
                void SetUp() override
                {
                    engine = &TestEnvironment::engine();
                    engine->set_mode(SushiBLAS::ExecutionMode::Auto);
                    engine->rng().reseed(0);
                }

                /** @brief Returns the backend of the program, over the runtime and engine. */
                [[nodiscard]] RuntimeExecutionBackend& backend()
                {
                    return TestEnvironment::backend();
                }

                /** @brief Waits for the device; a test runs the graph it records on itself. */
                void flush()
                {
                    TestEnvironment::runtime().wait().value();
                }

                /** @brief Points at the Engine the test records on; the program owns it. */
                SushiBLAS::Engine* engine = nullptr;
        };

    } // namespace Test
} // namespace SushiAI
