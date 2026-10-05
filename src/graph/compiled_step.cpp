/****************************************************************/
/* compiled_step.cpp                                            */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
