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

            // After the lowering and before the compile, so the trailing work
            // joins this plan rather than a second one. Its ordering against
            // the graph comes from the byte ranges each task declares, not from
            // the order it was recorded in.
            if (trailing_) operations_ += trailing_->record_into_step();

            // The one and only compile. It drains the engine's builder, which
            // is what makes the constant flush below safe to run afterwards -
            // the fills land in a fresh, empty builder rather than being tacked
            // onto the step that was about to be compiled.
            plan_ = engine.get_graph().compile();
            ++compile_count_;

            empty_ = plan_.roots.empty();

            pool.flush_constants();

            // Never before the capture: an empty base_in_degree leaves the
            // sentinel armed at zero, so the very first run reports complete
            // before any node has been dispatched. See the header.
            plan_.capture_base_state();
        }

        void CompiledStep::run()
        {
            if (empty_) return;

            // Between the last wait_all() and this dispatch, so a recorder may
            // write into memory its own kernels read without a race and without
            // a dependency edge to express - see step_recorder.hpp.
            if (trailing_) trailing_->before_replay();

            plan_.rearm();
            engine_->get_context().get_execution_engine().execute(plan_);
            engine_->get_context().wait_all();
            ++run_count_;

            if (trailing_) trailing_->after_replay();
        }

    } // namespace Graph
} // namespace SushiAI
