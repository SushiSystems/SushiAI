/**************************************************************************/
/* step_tasks_contract.hpp                                                */
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
 * @file step_tasks_contract.hpp
 * @brief Declares the checks every ExecutionBackend passes through the StepTasks it makes, run by
 *        one suite per adapter.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/random/random.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/execution/step_tasks.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Holds the element count of the vectors the contract fills. */
        inline constexpr std::int64_t K_STEP_TASKS_ELEMENTS = 4;

        /** @brief Holds the checks that define a StepTasks; an adapter's suite calls each one. */
        struct StepTasksContract
        {
            /** @brief A task recorded on the tasks writes its values when they replay. */
            static void expect_recorded_task_writes_its_values_on_replay(ExecutionBackend& backend)
            {
                SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor target, filled_vector(backend, 1.0));
                SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());

                record_constant(backend, *tasks, target, 2.0);
                EXPECT_GT(tasks->task_count(), 0u);
                SA_EXPECT_OK(tasks->seal());
                SA_EXPECT_OK(replay_once(*tasks));

                EXPECT_EQ(SushiAI::Test::read_tensor(target), vector_of(2.0));
            }

            /** @brief A second replay runs the tasks again, over what the first one left. */
            static void expect_second_replay_runs_the_tasks_again(ExecutionBackend& backend)
            {
                SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor target, filled_vector(backend, 1.0));
                SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());

                record_constant(backend, *tasks, target, 2.0);
                SA_EXPECT_OK(tasks->seal());
                SA_EXPECT_OK(replay_once(*tasks));
                EXPECT_EQ(SushiAI::Test::read_tensor(target), vector_of(2.0));

                const std::vector<double> ones = vector_of(1.0);
                SushiAI::Test::write_tensor(
                    target, SushiRuntime::span<const double>(ones.data(), ones.size()));
                SA_EXPECT_OK(replay_once(*tasks));
                EXPECT_EQ(SushiAI::Test::read_tensor(target), vector_of(2.0));
            }

            /** @brief Hooks run before the tasks of every replay, in the order they were added. */
            static void expect_hooks_run_before_the_tasks_in_the_order_added(
                ExecutionBackend& backend)
            {
                SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor target, filled_vector(backend, 1.0));
                SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());

                std::vector<std::string> order;
                std::vector<double> seen;
                SA_EXPECT_OK(tasks->before_each_replay([&]()
                {
                    order.push_back("first");
                    seen.push_back(SushiAI::Test::read_tensor(target).front());
                }));
                SA_EXPECT_OK(tasks->before_each_replay([&]()
                {
                    order.push_back("second");
                }));

                record_constant(backend, *tasks, target, 3.0);
                SA_EXPECT_OK(tasks->seal());
                SA_EXPECT_OK(replay_once(*tasks));
                SA_EXPECT_OK(replay_once(*tasks));

                // The first replay's hook sees the values before any task ran.
                EXPECT_EQ(order, (std::vector<std::string>{"first", "second", "first", "second"}));
                EXPECT_EQ(seen, (std::vector<double>{1.0, 3.0}));
            }

            /** @brief Tasks that hold no task have a task count of zero and replay as success. */
            static void expect_recording_with_no_task_replays_as_success(ExecutionBackend& backend)
            {
                SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());

                std::size_t hook_calls = 0;
                SA_EXPECT_OK(tasks->before_each_replay([&]()
                {
                    ++hook_calls;
                }));

                EXPECT_EQ(tasks->task_count(), 0u);
                SA_EXPECT_OK(tasks->refusal());
                SA_EXPECT_OK(tasks->seal());
                SA_EXPECT_OK(replay_once(*tasks));
                EXPECT_EQ(hook_calls, 0u);
            }

            /** @brief record_and_run calls its work once and runs what the work recorded. */
            static void expect_record_and_run_runs_its_work_once(ExecutionBackend& backend)
            {
                SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor target, filled_vector(backend, 1.0));

                std::size_t calls = 0;
                SA_EXPECT_OK(::SushiAI::Detail::record_and_run(backend,
                                                    [&](SushiBLAS::TaskRecorder& recorder)
                {
                    ++calls;
                    SushiBLAS::RandomOps random(recorder, backend.engine().rng());
                    random.constant(target, 4.0);
                    return Status();
                }));

                EXPECT_EQ(calls, 1u);
                EXPECT_EQ(SushiAI::Test::read_tensor(target), vector_of(4.0));
            }

            private:
                /** @brief Returns K_STEP_TASKS_ELEMENTS copies of @p value. */
                static std::vector<double> vector_of(double value)
                {
                    return std::vector<double>(static_cast<std::size_t>(K_STEP_TASKS_ELEMENTS),
                                               value);
                }

                /** @brief Allocates a float32 vector whose elements are all @p value. */
                static Result<SushiBLAS::Tensor> filled_vector(ExecutionBackend& backend,
                                                               double value)
                {
                    const std::int64_t dims[1] = {K_STEP_TASKS_ELEMENTS};
                    SA_TRY_ASSIGN(SushiBLAS::Tensor tensor,
                                  allocate_tensor(backend.engine(),
                                                  SushiRuntime::span<const std::int64_t>(dims, 1)));

                    const std::vector<double> values = vector_of(value);
                    SushiAI::Test::write_tensor(
                        tensor, SushiRuntime::span<const double>(values.data(), values.size()));
                    return std::move(tensor);
                }

                /** @brief Records a task that sets every element of @p target to @p value. */
                static void record_constant(ExecutionBackend& backend, StepTasks& tasks,
                                            SushiBLAS::Tensor& target, double value)
                {
                    SushiBLAS::RandomOps random(tasks.recorder(), backend.engine().rng());
                    random.constant(target, value);
                }

                /** @brief Replays @p tasks once with a report of its own. */
                static Status replay_once(StepTasks& tasks)
                {
                    SushiRuntime::RunReport report;
                    return tasks.replay(report);
                }
        };
    } // namespace Test
} // namespace SushiAI
