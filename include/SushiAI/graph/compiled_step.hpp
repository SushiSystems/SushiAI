/****************************************************************/
/* compiled_step.hpp                                            */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file compiled_step.hpp
 * @brief Declares the CompiledStep class for lowering, compiling, and replaying a step.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/execution/execution_engine.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering.hpp>
#include <SushiAI/graph/step_recorder.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class CompiledStep
         * @brief Manages the lowered and compiled execution plan for a graph.
         */
        class CompiledStep
        {
            public:
                /**
                 * @brief Lowers @p graph, compiles it once, and arms it for execution.
                 * @param engine Recording engine; must outlive this step.
                 * @param graph The graph to lower.
                 * @param pool Memory pool behind values; must outlive this step.
                 * @param trailing Optional extra recorder for trailing replay work.
                 * @param selector Fusion selector, or null for default.
                 */
                CompiledStep(SushiBLAS::Engine& engine, const Graph& graph, TensorPool& pool,
                             IStepRecorder* trailing = nullptr,
                             const IFusionSelector* selector = nullptr);

                CompiledStep(const CompiledStep&) = delete;
                CompiledStep& operator=(const CompiledStep&) = delete;
                CompiledStep(CompiledStep&&) = delete;
                CompiledStep& operator=(CompiledStep&&) = delete;

                /**
                 * @brief Re-arms the plan, executes it, and waits for completion.
                 */
                void run();

                /** @brief Whether extra non-IR work was folded into this plan. */
                bool has_trailing_work() const noexcept { return trailing_ != nullptr; }

                /**
                 * @brief How many times this step has been compiled.
                 *
                 * One, always, for the life of the object - that is the point.
                 * Exposed so a test can hold the guarantee to account.
                 */
                std::size_t compile_count() const noexcept { return compile_count_; }

                /** @brief How many times run() has executed the plan. */
                std::size_t run_count() const noexcept { return run_count_; }

                /** @brief Whether the graph lowered to no runnable work at all. */
                bool empty() const noexcept { return empty_; }

                /** @brief How many SushiBLAS calls the step is made of. */
                std::size_t operation_count() const noexcept { return operations_; }

                /** @brief What the selector chose for each fused node; see Lowering. */
                SushiRuntime::span<const FusionSelection> selections() const noexcept
                {
                    return SushiRuntime::span<const FusionSelection>(selections_.data(),
                                                                     selections_.size());
                }

                /**
                 * @brief The compiled plan, for profiling.
                 *
                 * Exposed only so a caller can hand it to
                 * @c IExecutionEngine::reset_profiling / @c collect_profiling.
                 * Do not compile, rearm or execute it yourself.
                 */
                SushiRuntime::Execution::CompiledGraph& plan() noexcept { return plan_; }

            private:
                SushiBLAS::Engine* engine_;
                IStepRecorder* trailing_;
                SushiRuntime::Execution::CompiledGraph plan_;
                std::vector<FusionSelection> selections_;
                std::size_t compile_count_ = 0;
                std::size_t run_count_ = 0;
                std::size_t operations_ = 0;
                bool empty_ = false;
        };

    } // namespace Graph
} // namespace SushiAI
