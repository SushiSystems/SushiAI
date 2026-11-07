/**************************************************************************/
/* test_environment.hpp                                                   */
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
 * @file test_environment.hpp
 * @brief Declares the GTest environment that owns the Runtime, Engine and backend of a program.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <utility>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/execution/runtime_execution_backend.hpp>
#include <SushiAI/engine_fixture/lazy_shared.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Owns the process-global Runtime for the whole binary. */
        class TestEnvironment : public ::testing::Environment
        {
            public:
                /** @brief Names the call that creates the runtime of the program. */
                using RuntimeFactory = LazyShared<SushiRuntime::Runtime>::Factory;

                /** @brief Creates the runtime with Detail::create_runtime(). */
                TestEnvironment() = default;

                /**
                 * @brief Creates the runtime with @p factory in place of Detail::create_runtime().
                 * @pre runtime() has not been called yet.
                 */
                explicit TestEnvironment(RuntimeFactory factory)
                {
                    runtime_.use_factory(std::move(factory));
                }

                /** @brief Destroys the backend, the Engine and the runtime after the last test. */
                void TearDown() override
                {
                    backend_.reset();
                    engine_.reset();
                    runtime_.release();
                }

                /**
                 * @brief Returns the shared runtime, creating it on the first call.
                 *
                 * A test that never calls it opens no device and starts no worker. A runtime that
                 * cannot be created ends the process with its message.
                 */
                static SushiRuntime::Runtime& runtime() noexcept
                {
                    Result<SushiRuntime::Runtime&> acquired = runtime_.acquire();
                    if (!acquired.has_value())
                    {
                        const std::string_view message = acquired.status().message();
                        std::fprintf(stderr, "cannot create the runtime: %.*s\n",
                                     static_cast<int>(message.size()), message.data());
                        std::abort();
                    }
                    return acquired.value();
                }

                /**
                 * @brief Returns the Engine of the program, built on the shared runtime on first use.
                 * @return The Engine every EngineTest of the program records on.
                 */
                static SushiBLAS::Engine& engine()
                {
                    if (!engine_)
                        engine_.emplace(runtime());
                    return *engine_;
                }

                /**
                 * @brief Returns the backend of the program, built on first use.
                 * @return The RuntimeExecutionBackend every EngineTest of the program runs through.
                 */
                static RuntimeExecutionBackend& backend()
                {
                    if (!backend_)
                        backend_.emplace(runtime(), engine());
                    return *backend_;
                }

            private:
                /** @brief Holds the backend of the program; released before its Engine. */
                static inline std::optional<RuntimeExecutionBackend> backend_;

                /** @brief Holds the Engine of the program; released before the runtime it lives on. */
                static inline std::optional<SushiBLAS::Engine> engine_;

                /** @brief Holds the runtime from the first runtime() call to TearDown(). */
                static inline LazyShared<SushiRuntime::Runtime> runtime_{
                    []()
                    {
                        return Detail::create_runtime();
                    }};

        };

    } // namespace Test
} // namespace SushiAI
