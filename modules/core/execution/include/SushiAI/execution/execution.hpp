/**************************************************************************/
/* execution.hpp                                                          */
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
 * @file execution.hpp
 * @brief Declares the calls that create a runtime and that record on a backend's tasks and run
 *        them, each returning a Status.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <utility>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/execution/step_tasks.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Detail
    {
        /**
         * @brief Creates the runtime an engine is built on.
         * @return The runtime, or the Status of the Error SushiRuntime returned for it.
         */
        [[nodiscard]] Result<SushiRuntime::Runtime> create_runtime(
            const SushiRuntime::RuntimeConfig& config = {}) noexcept;

        /**
         * @brief Records on tasks of its own with @p record, seals them and replays them once.
         * @param backend Makes the tasks.
         * @param record Called once with the recorder of the tasks; it returns Status.
         * @return Success, the failure of @p record with its tasks dropped, the refusal the
         *         recording kept, or the failure of the seal or the replay. A recording of no
         *         task runs nothing and succeeds.
         */
        template<typename Record>
        [[nodiscard]] Status record_and_run(ExecutionBackend& backend, Record&& record) noexcept;

        template<typename Record>
        Status record_and_run(ExecutionBackend& backend, Record&& record) noexcept
        {
            SA_TRY_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());

            const Status recorded = guard(SeamPhase::Record, [&]() -> Status
            {
                return std::forward<Record>(record)(tasks->recorder());
            });
            if (!recorded.is_ok())
                return recorded;

            SA_TRY(tasks->refusal());
            if (tasks->task_count() == 0)
                return Status();

            SA_TRY(tasks->seal());
            SushiRuntime::RunReport report;
            return tasks->replay(report);
        }

    } // namespace Detail
} // namespace SushiAI

/** @brief Short alias for the namespace of SushiRuntime, whose Runtime create_runtime() returns. */
namespace sr = SushiRuntime;
