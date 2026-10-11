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
 * @brief Implements the lowering, the single compilation and the replay of CompiledStep.
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
            : engine_(&engine), trailing_(trailing),
              recorder_(graph_, engine.get_scratch_pool(), engine.get_mode())
        {

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
            Lowering lowering(step->recorder_, engine.get_rng(), pool,
                              selector ? *selector : device);

            Status recorded = lowering.record(graph);
            step->operations_ = lowering.emitted_operations();
            step->preludes_ = lowering.take_preludes();

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

            if (!recorded.is_ok())
            {
                const Status dropped = SushiAI::Detail::discard_recorded(step->graph_);
                SA_FATAL_IF(!dropped.is_ok(), "A failed recording could not be dropped: {}",
                            dropped.message());
                return recorded;
            }

            SA_TRY(SushiAI::Detail::compile_graph(step->graph_, step->plan_));
            ++step->compile_count_;
            step->empty_ = step->plan_.roots.empty();

            SA_TRY(pool.flush_constants());

            // The base state is captured after compilation, which sets the in-degrees it reads.
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Execute, [&]()
            {
                step->plan_.capture_base_state();
                return Status();
            }));

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

            // The plan is idle here, so a prelude may write memory its kernels read.
            for (const ReplayPrelude& prelude : preludes_)
                prelude(static_cast<std::uint64_t>(run_count_));

            if (trailing_)
                trailing_->before_replay();

            plan_.rearm();
            SA_TRY(SushiAI::Detail::run_plan(engine_->get_context(), plan_));
            ++run_count_;

            if (trailing_)
                trailing_->after_replay();

            return Status();
        }

    } // namespace Graph
} // namespace SushiAI
