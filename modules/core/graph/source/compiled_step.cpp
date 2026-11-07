/**************************************************************************/
/* compiled_step.cpp                                                      */
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
 * @file compiled_step.cpp
 * @brief Implements the lowering, the recording and the replay of CompiledStep.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/compiled_step.hpp>

#include <cstdint>
#include <string>
#include <utility>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/graph/lowering.hpp>
#include <SushiAI/profile/profile.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Holds the origin of the tasks the trailing recorder adds. */
            constexpr std::string_view K_TRAILING_ORIGIN = "step.trailing";
        } // namespace

        CompiledStep::CompiledStep(std::unique_ptr<StepTasks> tasks,
                                   StepRecorder* trailing) noexcept
            : trailing_(trailing), tasks_(std::move(tasks))
        {

        }

        Status CompiledStep::add_replay_hooks(std::vector<ReplayPrelude> preludes)
        {
            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                const std::uint64_t* run_index = &prelude_index_;
                for (ReplayPrelude& prelude : preludes)
                {
                    // A run() replays one step, so each replay calls a prelude once.
                    SA_TRY(tasks_->before_each_replay([run_index, work = std::move(prelude)]()
                    {
                        work(*run_index);
                    }));
                    ++prelude_count_;
                }

                if (trailing_)
                {
                    StepRecorder* trailing = trailing_;
                    SA_TRY(tasks_->before_each_replay([trailing]()
                    {
                        trailing->before_replay();
                    }));
                }

                return Status();
            });
        }

        Status check_compilable(const Graph& graph) noexcept
        {
            SA_FAIL_IF(graph.has_refusal(), StatusCode::InvalidDescription,
                       "This graph holds a refused operation and cannot be compiled; the builder "
                       "call that refused it returned the reason");

            return Status();
        }

        Result<std::unique_ptr<CompiledStep>> CompiledStep::compile(
            ExecutionBackend& backend, const Graph& graph, TensorPool& pool,
            StepRecorder* trailing, const FusionSelector* selector, NodeRange nodes,
            std::string_view label) noexcept
        {
            SA_TRY(check_compilable(graph));

            SA_TRY_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());

            SA_TRY_ASSIGN(std::unique_ptr<CompiledStep> step,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<CompiledStep>(
                    new CompiledStep(std::move(tasks), trailing));
            }));

            const DeviceFusionSelector device = DeviceFusionSelector::for_backend(backend);
            Lowering lowering(step->tasks_->recorder(), backend.engine().rng(), pool,
                              selector ? *selector : device);

            Status recorded;
            {
                SA_ZONE("compile", "lower");
                recorded = lowering.record(graph, nodes);
            }
            step->operations_ = lowering.emitted_operations();

            const Status copied = SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate,
                                                         [&]()
            {
                const SushiRuntime::span<const FusionSelection> chosen = lowering.selections();
                step->selections_.assign(chosen.begin(), chosen.end());
                return Status();
            });
            if (recorded.is_ok())
                recorded = copied;

            if (recorded.is_ok() && trailing)
            {
                SushiBLAS::TaskRecorder& task_recorder = step->tasks_->recorder();
                const StepRecording recording(task_recorder, step->tasks_->runtime_recorder());
                const SushiBLAS::TaskRecorder::OriginScope origin =
                    task_recorder.origin(K_TRAILING_ORIGIN);
                const Result<std::size_t> trailed = trailing->record_into_step(recording);
                if (trailed.has_value())
                    step->operations_ += trailed.value();
                else
                    recorded = trailed.status();
            }

            // A step that fails here dies with what it recorded when this function returns.
            SA_TRY(recorded);

            step->empty_ = step->tasks_->task_count() == 0;
            SA_TRY(step->add_replay_hooks(lowering.take_preludes()));

            // The first plan copies the label, so it is set before the seal.
            if (!label.empty())
            {
                SA_TRY_ASSIGN(std::string name,
                              SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                           [&]()
                {
                    return std::string(label);
                }));
                SA_TRY(step->tasks_->set_label(std::move(name)));
            }

            // An empty step is never compiled, so a refused recording is read here and only here.
            SA_TRY(step->tasks_->refusal());

            // A recording with no task has no plan to compile.
            if (!step->empty_)
            {
                SA_ZONE("compile", "runtime");
                SA_TRY(step->tasks_->seal());
                step->compile_count_ = 1;
            }

            SA_TRY(pool.flush_constants(backend));

            return std::move(step);
        }

        Status CompiledStep::restore_run_count(std::size_t count) noexcept
        {
            return run_count_.restore(count);
        }

        Status CompiledStep::run() noexcept
        {
            return run_at(static_cast<std::uint64_t>(run_count_.count()));
        }

        Status CompiledStep::run_at(std::uint64_t run_index) noexcept
        {
            if (empty_)
                return Status();

            prelude_index_ = run_index;
            SA_TRY(tasks_->replay(report_));
            run_count_.advance();

            if (trailing_)
                trailing_->after_replay();

            return Status();
        }

    } // namespace Graph
} // namespace SushiAI
