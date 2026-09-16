/**************************************************************************/
/* compiled_step.cpp                                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file compiled_step.cpp
 * @brief Implements CompiledStep lowerings, graph compilation, and replays.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/compiled_step.hpp>

namespace SushiAI
{
    namespace Graph
    {
        CompiledStep::CompiledStep(SushiBLAS::Engine& engine, const Graph& graph,
                                   TensorPool& pool, IStepRecorder* trailing,
                                   const IFusionSelector* selector)
            : engine_(&engine), trailing_(trailing)
        {
            Lowering lowering(engine, pool, selector);
            lowering.record(graph);
            operations_ = lowering.emitted_operations();

            const SushiRuntime::span<const FusionSelection> chosen = lowering.selections();
            selections_.assign(chosen.begin(), chosen.end());

            // Trailing device work is recorded into this plan before compilation.
            if (trailing_) operations_ += trailing_->record_into_step();

            // Single step compilation drains the builder before constant fills are flushed.
            plan_ = engine.get_graph().compile();
            ++compile_count_;

            empty_ = plan_.roots.empty();

            pool.flush_constants();

            // Base state capture must follow graph compilation to arm in-degree sentinels.
            plan_.capture_base_state();
        }

        void CompiledStep::run()
        {
            if (empty_) return;

            // Host state refresh executes between device synchronization and dispatch.
            if (trailing_) trailing_->before_replay();

            plan_.rearm();
            engine_->get_context().get_execution_engine().execute(plan_);
            engine_->get_context().wait_all();
            ++run_count_;

            if (trailing_) trailing_->after_replay();
        }

    } // namespace Graph
} // namespace SushiAI
