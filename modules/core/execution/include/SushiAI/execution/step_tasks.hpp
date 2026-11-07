/**************************************************************************/
/* step_tasks.hpp                                                         */
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
 * @file step_tasks.hpp
 * @brief Declares StepTasks, the port over one recording of tasks that replays.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    /**
     * @class StepTasks
     * @brief Holds one recording of tasks and replays it, as often as its owner asks.
     *
     * A backend hands one out through ExecutionBackend::make_tasks(). The recording lives as long
     * as the object does, and the tasks recorded on it are dropped with it.
     */
    class StepTasks
    {
        public:
            virtual ~StepTasks() = default;

            /** @brief Returns the recorder every operation view of the recording records on. */
            [[nodiscard]] virtual SushiBLAS::TaskRecorder& recorder() noexcept = 0;

            /**
             * @brief Returns the recorder of the SushiRuntime graph, for scopes and remote nodes.
             * @return Null when the recording has no SushiRuntime graph.
             */
            [[nodiscard]] virtual SushiRuntime::Recorder* runtime_recorder() noexcept = 0;

            /** @brief Returns how many tasks the recording holds. */
            [[nodiscard]] virtual std::size_t task_count() const noexcept = 0;

            /** @brief Returns the refusal kept while recording, or success when none was kept. */
            [[nodiscard]] virtual Status refusal() const = 0;

            /**
             * @brief Adds a hook that runs before every replay, after the hooks added earlier.
             * @return Success, or OutOfMemory when the hook could not be stored.
             */
            [[nodiscard]] virtual Status before_each_replay(std::function<void()> hook) = 0;

            /**
             * @brief Names the recording, for a report of the runtime.
             * @pre Called before seal().
             */
            [[nodiscard]] virtual Status set_label(std::string label) = 0;

            /**
             * @brief Compiles the recording into the plan its replays run.
             *
             * A recording with no task is never sealed into a plan and replays as nothing.
             * @return Success, or the failure of the compilation.
             */
            [[nodiscard]] virtual Status seal() noexcept = 0;

            /**
             * @brief Runs the hooks, then every task, and keeps the tasks for the next replay.
             * @param report Refilled by a replay that runs on the runtime; left as it was by one
             *               that runs on the host.
             * @return Success, also for a recording with no task, or the failure of the replay;
             *         a failed task is a DeviceFailed naming its index and name.
             */
            [[nodiscard]] virtual Status replay(SushiRuntime::RunReport& report) noexcept = 0;
    };
} // namespace SushiAI
