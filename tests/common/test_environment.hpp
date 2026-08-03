/**************************************************************************/
/* test_environment.hpp                                                   */
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

#pragma once

/**
 * @file test_environment.hpp
 * @brief The one RuntimeContext the whole test binary shares.
 *
 * A RuntimeContext starts a worker pool, probes the topology and builds SYCL
 * queues. Doing that per test would dominate the suite's runtime and, worse,
 * would repeatedly tear down and restart the thread pool while other work is
 * settling. One process-global context in a GTest Environment, and a fresh
 * Engine per test so no test inherits another's task graph or RNG offset -
 * exactly the split SushiBLAS's own suite uses.
 */

#include <memory>
#include <gtest/gtest.h>
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/SushiRuntime.h>

namespace SushiAI
{
    namespace Test
    {
        /**
         * @class TestEnvironment
         * @brief Owns the process-global RuntimeContext for the whole binary.
         */
        class TestEnvironment : public ::testing::Environment
        {
            public:
                /** @brief Builds the runtime context before the first test. */
                void SetUp() override
                {
                    context_ = std::make_unique<SushiRuntime::Execution::RuntimeContext>();
                }

                /** @brief Tears the runtime context down after the last test. */
                void TearDown() override
                {
                    context_.reset();
                }

                /**
                 * @brief The shared context.
                 * @return A reference valid for the whole run.
                 */
                static SushiRuntime::Execution::RuntimeContext& context()
                {
                    return *context_;
                }

            private:
                static inline std::unique_ptr<SushiRuntime::Execution::RuntimeContext> context_;
        };

    } // namespace Test
} // namespace SushiAI
