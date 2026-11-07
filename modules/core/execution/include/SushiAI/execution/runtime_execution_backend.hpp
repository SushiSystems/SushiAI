/**************************************************************************/
/* runtime_execution_backend.hpp                                          */
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
 * @file runtime_execution_backend.hpp
 * @brief Declares RuntimeExecutionBackend, the ExecutionBackend whose tasks run on a SushiRuntime
 *        graph.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/execution/runtime_backend.hpp>

namespace SushiAI
{
    /**
     * @class RuntimeExecutionBackend
     * @brief Borrows a runtime and an engine built on it, and records each StepTasks on a graph
     *        of that runtime.
     */
    class RuntimeExecutionBackend final : public RuntimeBackend
    {
        public:
            /**
             * @brief Takes the runtime and the engine built on it.
             * @param runtime Must outlive this backend and every StepTasks it made.
             * @param engine Built on @p runtime; must outlive this backend and its tasks.
             */
            RuntimeExecutionBackend(SushiRuntime::Runtime& runtime,
                                    SushiBLAS::Engine& engine) noexcept
                : runtime_(&runtime), engine_(&engine)
            {

            }

            RuntimeExecutionBackend(const RuntimeExecutionBackend&) = delete;
            RuntimeExecutionBackend& operator=(const RuntimeExecutionBackend&) = delete;
            RuntimeExecutionBackend(RuntimeExecutionBackend&&) = delete;
            RuntimeExecutionBackend& operator=(RuntimeExecutionBackend&&) = delete;
            ~RuntimeExecutionBackend() override = default;

            /** @brief Returns the borrowed engine. */
            [[nodiscard]] SushiBLAS::Engine& engine() noexcept override
            {
                return *engine_;
            }

            /** @brief Returns the borrowed engine. */
            [[nodiscard]] const SushiBLAS::Engine& engine() const noexcept override
            {
                return *engine_;
            }

            /** @brief Returns the borrowed runtime, for the adapters that exist only on one. */
            [[nodiscard]] SushiRuntime::Runtime& runtime() noexcept override
            {
                return *runtime_;
            }

            /** @brief Makes a recording on a new graph of the runtime. */
            [[nodiscard]] Result<std::unique_ptr<StepTasks>> make_tasks() noexcept override;

            /** @brief Returns whether the first device queue has native double precision. */
            [[nodiscard]] bool supports_fp64() const noexcept override;

        private:
            /// @brief The runtime the graphs and the late counts live on; never null.
            SushiRuntime::Runtime* runtime_;

            /// @brief The engine the tasks record on; never null.
            SushiBLAS::Engine* engine_;
    };
} // namespace SushiAI
