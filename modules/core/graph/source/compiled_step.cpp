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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file compiled_step.cpp
 * @brief Implements the lowering, the recording and the replay of CompiledStep.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/compiled_step.hpp>

#include <cstdint>
#include <utility>

#include <SushiAI/core/check.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/graph/lowering.hpp>

namespace SushiAI
{
    namespace Graph
    {
        CompiledStep::CompiledStep(SushiBLAS::Engine& engine, StepRecorder* trailing)
            : engine_(&engine), trailing_(trailing), graph_(engine.runtime().graph()),
              recorder_(graph_.recorder(), engine.scratch_pool(), engine.mode())
        {

        }

        Status CompiledStep::add_replay_hooks(std::vector<ReplayPrelude> preludes)
        {
            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                const std::size_t* finished_runs = &run_count_;
                for (ReplayPrelude& prelude : preludes)
                {
                    // A run() replays one step, so the hook's own count is always 0.
                    graph_.before_each_step(
                        [finished_runs, work = std::move(prelude)](std::size_t)
                    {
                        work(static_cast<std::uint64_t>(*finished_runs));
                    });
                    ++prelude_count_;
                }

                if (trailing_)
                {
                    StepRecorder* trailing = trailing_;
                    graph_.before_each_step([trailing](std::size_t)
                    {
                        trailing->before_replay();
                    });
                }

                return Status();
            });
        }

        Result<std::unique_ptr<CompiledStep>> CompiledStep::compile(
            SushiBLAS::Engine& engine, const Graph& graph, TensorPool& pool,
            StepRecorder* trailing, const FusionSelector* selector)
        {
            SA_FAIL_IF(graph.has_refusal(), StatusCode::InvalidDescription,
                       "This graph holds a refused operation and cannot be compiled; the builder "
                       "call that refused it returned the reason");

            SA_TRY_ASSIGN(std::unique_ptr<CompiledStep> step,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<CompiledStep>(new CompiledStep(engine, trailing));
            }));

            const DeviceFusionSelector device = DeviceFusionSelector::for_engine(engine);
            Lowering lowering(step->recorder_, engine.rng(), pool,
                              selector ? *selector : device);

            Status recorded = lowering.record(graph);
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
                const Result<std::size_t> trailed = trailing->record_into_step(step->recorder_);
                if (trailed.has_value())
                    step->operations_ += trailed.value();
                else
                    recorded = trailed.status();
            }

            // A step that fails here dies with what it recorded when this function returns.
            SA_TRY(recorded);

            const SushiRuntime::Recorder& tasks = step->graph_.recorder();
            if (const SushiRuntime::Error* refused = tasks.refusal())
                return SushiAI::Detail::status_of_error(*refused,
                                                        SushiAI::Detail::SeamPhase::Record);

            step->empty_ = tasks.node_count() == 0;
            SA_TRY(step->add_replay_hooks(lowering.take_preludes()));

            SA_TRY(pool.flush_constants());

            return std::move(step);
        }

        Status CompiledStep::restore_run_count(std::size_t count)
        {
            SA_FAIL_IF(run_count_ != 0, StatusCode::InvalidDescription,
                       "The run count of a compiled step is restored before its first run, "
                       "but this step has reached run {}",
                       run_count_);

            run_count_ = count;
            return Status();
        }

        Status CompiledStep::run()
        {
            if (empty_)
                return Status();

            SA_TRY(SushiAI::Detail::run_step(engine_->runtime(), graph_, report_));
            ++run_count_;

            if (trailing_)
                trailing_->after_replay();

            return Status();
        }

    } // namespace Graph
} // namespace SushiAI
