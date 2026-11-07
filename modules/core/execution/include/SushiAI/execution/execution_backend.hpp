/**************************************************************************/
/* execution_backend.hpp                                                  */
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
 * @file execution_backend.hpp
 * @brief Declares ExecutionBackend, the port over the engine a run records on and the tasks it
 *        records.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/execution/step_tasks.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    /**
     * @class ExecutionBackend
     * @brief Gives a caller the engine to allocate and record on, and a StepTasks to record into.
     *
     * The caller that makes the engine picks the adapter; nothing asks an engine what kind it is.
     */
    class ExecutionBackend
    {
        public:
            virtual ~ExecutionBackend() = default;

            /** @brief Returns the engine tensors are allocated from and operations are recorded on. */
            [[nodiscard]] virtual SushiBLAS::Engine& engine() noexcept = 0;

            /** @brief Returns the engine, for a caller that only reads it. */
            [[nodiscard]] virtual const SushiBLAS::Engine& engine() const noexcept = 0;

            /**
             * @brief Makes an empty recording on this backend.
             * @return The recording, or the Status of the failure that kept it from being made.
             */
            [[nodiscard]] virtual Result<std::unique_ptr<StepTasks>> make_tasks() noexcept = 0;

            /** @brief Returns whether the device has native double precision; the host has. */
            [[nodiscard]] virtual bool supports_fp64() const noexcept = 0;
    };
} // namespace SushiAI
