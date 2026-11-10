/**************************************************************************/
/* compiled_step.hpp                                                      */
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
 * @file compiled_step.hpp
 * @brief Declares the CompiledStep class, which lowers, records and replays one step on the
 *        tasks of an ExecutionBackend.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/execution/step_tasks.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/node_range.hpp>
#include <SushiAI/graph/replay_prelude.hpp>
#include <SushiAI/graph/run_counter.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/tensor_pool.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @brief Checks that @p graph may be compiled.
         * @return Success, or InvalidDescription when an emitter refused part of the graph and
         *         the caller dropped the refusal.
         */
        [[nodiscard]] Status check_compilable(const Graph& graph) noexcept;

        /**
         * @class CompiledStep
         * @brief Owns the tasks of one step and replays them.
         */
        class CompiledStep final
        {
            public:
                /**
                 * @brief Lowers the nodes @p nodes of @p graph, then @p trailing, and compiles.
                 * @param backend Must outlive the step, as must @p pool and @p trailing.
                 * @param selector Picks the form of a fusible node; null selects by device.
                 * @param nodes The nodes the step is made of; every node by default.
                 * @param label Names the runtime graph; empty leaves it unnamed.
                 * @return The step, or the first refusal of lowering, recording or compilation.
                 */
                [[nodiscard]] static Result<std::unique_ptr<CompiledStep>> compile(
                    ExecutionBackend& backend, const Graph& graph, TensorPool& pool,
                    StepRecorder* trailing = nullptr, const FusionSelector* selector = nullptr,
                    NodeRange nodes = {}, std::string_view label = {}) noexcept;

                CompiledStep(const CompiledStep&) = delete;
                CompiledStep& operator=(const CompiledStep&) = delete;
                CompiledStep(CompiledStep&&) = delete;
                CompiledStep& operator=(CompiledStep&&) = delete;
                ~CompiledStep() = default;

                /**
                 * @brief Replays the step once and waits until it has finished.
                 *
                 * Each replay prelude is called with run_count() before the step is submitted.
                 * @return Success, also for an empty step, or the failure of the run.
                 */
                [[nodiscard]] Status run() noexcept;

                /**
                 * @brief Replays the step once, as run() does, but calls each replay prelude
                 *        with @p run_index instead of run_count().
                 *
                 * run_count() still counts the run.
                 * @return What run() returns.
                 */
                [[nodiscard]] Status run_at(std::uint64_t run_index) noexcept;

                /** @brief Returns how many replay preludes the lowered nodes registered. */
                [[nodiscard]] std::size_t replay_prelude_count() const noexcept
                {
                    return prelude_count_;
                }

                /** @brief Returns whether a trailing recorder added work to this step. */
                [[nodiscard]] bool has_trailing_work() const noexcept
                {
                    return trailing_ != nullptr;
                }

                /**
                 * @brief Returns how many times the step has been sealed.
                 * @return 1 from compile() on, and 0 for an empty step.
                 */
                [[nodiscard]] std::size_t compile_count() const noexcept
                {
                    return compile_count_;
                }

                /** @brief Returns the index the next run() gives its replay preludes. */
                [[nodiscard]] std::size_t run_count() const noexcept
                {
                    return run_count_.count();
                }

                /**
                 * @brief Sets run_count() to what the step of a saved run had reached.
                 * @param count The runs the saved step had finished.
                 * @return InvalidDescription, with run_count(), once the step has run or
                 *         has been restored to a count above 0.
                 */
                [[nodiscard]] Status restore_run_count(std::size_t count) noexcept;

                /** @brief Returns whether the graph lowered to no runnable work. */
                [[nodiscard]] bool empty() const noexcept
                {
                    return empty_;
                }

                /** @brief Returns how many SushiBLAS tasks the step is made of. */
                [[nodiscard]] std::size_t operation_count() const noexcept
                {
                    return operations_;
                }

                /**
                 * @brief Returns the origin label of each task the step recorded, in order.
                 *
                 * The tasks of a node carry "<node id>:<operation>"; those of a trailing recorder
                 * carry "step.trailing". Empty when the backend keeps no labels: only the host
                 * backend does.
                 */
                [[nodiscard]] std::vector<std::string> task_origins() const
                {
                    return tasks_->task_origins();
                }

                /** @brief Returns what the selector chose for each fused node. */
                [[nodiscard]] SushiRuntime::span<const FusionSelection> selections() const noexcept
                {
                    return SushiRuntime::span<const FusionSelection>(selections_.data(),
                                                                     selections_.size());
                }

                /**
                 * @brief Returns the report the last run() of a step that holds work refilled.
                 * @return An empty report marked failed after a run() that failed.
                 */
                [[nodiscard]] const SushiRuntime::RunReport& last_report() const noexcept
                {
                    return report_;
                }

            private:
                /**
                 * @brief Builds an empty step over @p tasks.
                 * @param tasks The recording the step records into and replays; never null.
                 */
                CompiledStep(std::unique_ptr<StepTasks> tasks, StepRecorder* trailing) noexcept;

                /**
                 * @brief Adds the hooks tasks_ calls before each replay: one per prelude, in
                 *        the order given, then one for the trailing recorder.
                 * @return Success, or OutOfMemory when a hook could not be stored.
                 */
                [[nodiscard]] Status add_replay_hooks(std::vector<ReplayPrelude> preludes);

                /// @brief The recorder of the trailing work, or null; not owned.
                StepRecorder* trailing_;

                /// @brief The recording the step records into, seals once and replays; never null.
                std::unique_ptr<StepTasks> tasks_;

                /// @brief The report each run() refills.
                SushiRuntime::RunReport report_;

                /// @brief The fusion decisions of the lowering, in execution order.
                std::vector<FusionSelection> selections_;

                /// @brief The number of replay preludes tasks_ holds as hooks.
                std::size_t prelude_count_ = 0;

                /// @brief The index the replay preludes of the coming replay are called with.
                std::uint64_t prelude_index_ = 0;

                /// @brief The runs finished, counted from what restore_run_count() set.
                RunCounter run_count_;

                /// @brief The number of tasks the lowering and the trailing recorder recorded.
                std::size_t operations_ = 0;

                /// @brief How many times the step was sealed: 1 for a step that holds work.
                std::size_t compile_count_ = 0;

                /// @brief Whether nothing was recorded on tasks_.
                bool empty_ = false;
        };

    } // namespace Graph
} // namespace SushiAI
