/**************************************************************************/
/* test_graph.hpp                                                         */
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

/**
 * @file test_graph.hpp
 * @brief Declares TestGraph, a recording a test owns with the recorder that records on it.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/execution/step_tasks.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Test
    {
        /**
         * @brief Owns one StepTasks of a backend, so a test records on a recording of its own
         *        and runs it.
         *
         * The recording runs once; a test that records again builds a second TestGraph.
         */
        class TestGraph final
        {
            public:
                /**
                 * @brief Makes the tasks the test records on.
                 * @param backend Makes the tasks; it must outlive this object. A backend that
                 *                cannot make tasks stops the process.
                 */
                explicit TestGraph(ExecutionBackend& backend) noexcept
                    : tasks_(backend.make_tasks().value())
                {

                }

                TestGraph(const TestGraph&) = delete;
                TestGraph& operator=(const TestGraph&) = delete;
                TestGraph(TestGraph&&) = delete;
                TestGraph& operator=(TestGraph&&) = delete;
                ~TestGraph() = default;

                /** @brief Returns the recorder every operation of the test records through. */
                [[nodiscard]] SushiBLAS::TaskRecorder& recorder() noexcept
                {
                    return tasks_->recorder();
                }

                /**
                 * @brief Returns the runtime recorder of the tasks, which recorder() records on.
                 * @pre The backend is a RuntimeExecutionBackend; a host recording has none.
                 */
                [[nodiscard]] SushiRuntime::Recorder& graph_recorder() noexcept
                {
                    return *tasks_->runtime_recorder();
                }

                /** @brief Seals and replays every task recorded and returns when they finish. */
                [[nodiscard]] Status run() noexcept
                {
                    SA_TRY(tasks_->seal());
                    SushiRuntime::RunReport report;
                    return tasks_->replay(report);
                }

            private:
                /// @brief The recording the test records on.
                std::unique_ptr<StepTasks> tasks_;
        };
    } // namespace Test
} // namespace SushiAI
