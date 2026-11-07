/**************************************************************************/
/* step_recording.hpp                                                     */
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
 * @file step_recording.hpp
 * @brief Declares StepRecording, the two recorders a step recorder records through.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/failure/check.hpp>

namespace SushiBLAS
{
    class TaskRecorder;
} // namespace SushiBLAS

namespace SushiRuntime
{
    class Recorder;
} // namespace SushiRuntime

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @brief Hands a step recorder the task recorder of a graph and the graph's own
         *        runtime recorder, which the task recorder records on.
         *
         * Holds both by address and owns neither; both outlive the recording. A host recording
         * holds the task recorder alone, and a detached one holds neither, for a recorder that
         * records no task.
         */
        class StepRecording final
        {
            public:
                /** @brief Takes @p tasks and @p graph, the recorder @p tasks was built on. */
                StepRecording(SushiBLAS::TaskRecorder& tasks,
                              SushiRuntime::Recorder& graph) noexcept
                    : tasks_(&tasks), graph_(&graph)
                {

                }

                /**
                 * @brief Takes @p tasks and @p graph, the runtime recorder of the step or null.
                 *
                 * A null @p graph is a recording with no SushiRuntime graph, as on the host kind:
                 * graph() stops the process on it, as it does on a detached recording.
                 */
                StepRecording(SushiBLAS::TaskRecorder& tasks,
                              SushiRuntime::Recorder* graph) noexcept
                    : tasks_(&tasks), graph_(graph)
                {

                }

                /**
                 * @brief Returns a recording that holds no recorder.
                 *
                 * A StepRecorder that records no task, such as InMemoryStepRecorder, can be
                 * driven with it and touches no device.
                 */
                [[nodiscard]] static StepRecording detached() noexcept
                {
                    return StepRecording();
                }

                /** @brief Reports whether this recording holds no recorder. */
                [[nodiscard]] bool is_detached() const noexcept
                {
                    return tasks_ == nullptr;
                }

                /**
                 * @brief Returns the recorder of SushiBLAS operations and packs.
                 * @pre The recording is not detached; asking a detached one stops the process.
                 */
                [[nodiscard]] SushiBLAS::TaskRecorder& tasks() const noexcept
                {
                    SA_FATAL_IF(tasks_ == nullptr,
                                "A detached StepRecording holds no task recorder");
                    return *tasks_;
                }

                /**
                 * @brief Returns the runtime recorder of the graph, for scopes and remote nodes.
                 * @pre The recording holds a runtime recorder; asking a detached or a host one
                 *      stops the process.
                 */
                [[nodiscard]] SushiRuntime::Recorder& graph() const noexcept
                {
                    SA_FATAL_IF(graph_ == nullptr,
                                "This StepRecording holds no runtime recorder");
                    return *graph_;
                }

            private:
                /** @brief Builds the detached recording. */
                StepRecording() noexcept : tasks_(nullptr), graph_(nullptr)
                {

                }

                /// @brief The task recorder; null only for a detached recording.
                SushiBLAS::TaskRecorder* tasks_;

                /// @brief The runtime recorder tasks_ records on; null when detached or host.
                SushiRuntime::Recorder* graph_;
        };

    } // namespace Graph
} // namespace SushiAI
