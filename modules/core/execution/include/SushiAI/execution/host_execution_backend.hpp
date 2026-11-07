/**************************************************************************/
/* host_execution_backend.hpp                                             */
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
 * @file host_execution_backend.hpp
 * @brief Declares HostExecutionBackend, the ExecutionBackend whose tasks run on the host.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/execution/execution_backend.hpp>

namespace SushiAI
{
    /**
     * @class HostExecutionBackend
     * @brief Owns a host Engine and records each StepTasks on a HostTaskSink of its own.
     */
    class HostExecutionBackend final : public ExecutionBackend
    {
        public:
            /**
             * @brief Builds the host engine.
             * @param layout The memory layout of the tensors the engine creates.
             * @throws std::bad_alloc when the engine cannot be allocated.
             */
            explicit HostExecutionBackend(SushiBLAS::Core::Layout layout =
                                              SushiBLAS::Core::Layout::RowMajor);

            HostExecutionBackend(const HostExecutionBackend&) = delete;
            HostExecutionBackend& operator=(const HostExecutionBackend&) = delete;
            HostExecutionBackend(HostExecutionBackend&&) = delete;
            HostExecutionBackend& operator=(HostExecutionBackend&&) = delete;
            ~HostExecutionBackend() override = default;

            /** @brief Returns the host engine. */
            [[nodiscard]] SushiBLAS::Engine& engine() noexcept override;

            /** @brief Returns the host engine. */
            [[nodiscard]] const SushiBLAS::Engine& engine() const noexcept override;

            /**
             * @brief Makes a recording that keeps its tasks in a HostTaskSink.
             * @pre The tasks do not outlive this backend.
             */
            [[nodiscard]] Result<std::unique_ptr<StepTasks>> make_tasks() noexcept override;

            /** @brief Returns true: the host computes in double precision. */
            [[nodiscard]] bool supports_fp64() const noexcept override;

        private:
            /// @brief The engine the tasks record on and tensors are allocated from.
            SushiBLAS::Engine engine_;
    };
} // namespace SushiAI
