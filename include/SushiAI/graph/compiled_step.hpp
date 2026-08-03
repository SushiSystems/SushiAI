/**************************************************************************/
/* compiled_step.hpp                                                      */
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

#pragma once

/**
 * @file compiled_step.hpp
 * @brief One training step, lowered and compiled once, replayed for ever.
 *
 * The compile-once/replay contract is three calls in a fixed order with two
 * ways to get it wrong, so it gets a type rather than a comment:
 *
 * 1. @c TaskGraph::compile() @e drains the builder. Call it twice and the
 *    second call returns an empty plan whose execute() throws for want of a
 *    root node - so the second compile has to be impossible, not discouraged.
 * 2. @c CompiledGraph::execute() consumes each node's in-degree as it releases
 *    successors, so a replay must @c rearm() first.
 * 3. @c rearm() before @c capture_base_state() is the sharp one. The snapshot
 *    is empty at that point, so the per-node loop restores nothing - but the
 *    sentinel's in-degree is still written, as zero. A sentinel that starts
 *    ready fires the moment the run begins, so @c execute() returns
 *    "complete" before a single kernel has run, and the step silently does
 *    nothing. Loss stays flat and nothing reports an error.
 *
 * Doing both in the constructor makes (3) unrepresentable: there is no state
 * of this object in which the base state has not been captured.
 *
 * @c compile_count() exists so a test can assert the guarantee rather than
 * trust it. A training loop that recompiles per step still converges - it is
 * just quietly paying graph-analysis cost on every step, which is exactly the
 * kind of regression that hides for months.
 *
 * @section trailing Work that has no IR node
 *
 * A step is not only what the IR describes. The optimizer's parameter update
 * has no node in the graph, and used to be recorded and flushed on its own
 * every step for the reason optimizer.hpp records. Once it is fused - one
 * kernel per parameter, its per-step numbers read from device memory rather
 * than baked into a capture - it is replay-invariant work like any other, so
 * @c CompiledStep takes an optional @c IStepRecorder and folds it into the same
 * plan: recorded after the lowering, before the one compile, and driven around
 * every replay. Nothing about the ordering needs stating, because every
 * SushiBLAS call declares the bytes it touches and the runtime's dependency
 * tracker sees that the update reads the gradients the backward pass wrote and
 * writes the parameters the forward pass read.
 */

#include <cstddef>
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/execution/execution_engine.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering.hpp>
#include <SushiAI/graph/step_recorder.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class CompiledStep
         * @brief RAII over lower-compile-arm-replay for one graph.
         *
         * Non-copyable and non-movable: the plan holds raw @c Node* roots into
         * its own owned node list, and the engine's execution path is handed a
         * reference to it, so the object must not change address between the
         * constructor and the last run().
         */
        class CompiledStep
        {
            public:
                /**
                 * @brief Lowers @p graph, compiles it exactly once, and arms it.
                 *
                 * On return the step is ready to run: the plan is compiled, the
                 * in-degree baseline is captured, and every constant operand
                 * the lowering asked for has been filled and waited on.
                 *
                 * @param engine The engine to record into; must outlive this step.
                 * @param graph The graph to lower; only read, and only here.
                 * @param pool The memory behind the graph's values; must outlive this step.
                 * @param trailing Optional extra work to record into the same
                 *        plan and drive around every replay - see the header's
                 *        "Work that has no IR node". Must outlive this step,
                 *        and must record only replay-invariant work; nothing
                 *        here can check that, so the caller owes the promise.
                 * @throws Error If lowering fails - see Lowering::record.
                 */
                CompiledStep(SushiBLAS::Engine& engine, const Graph& graph, TensorPool& pool,
                             IStepRecorder* trailing = nullptr);

                CompiledStep(const CompiledStep&) = delete;
                CompiledStep& operator=(const CompiledStep&) = delete;
                CompiledStep(CompiledStep&&) = delete;
                CompiledStep& operator=(CompiledStep&&) = delete;

                /**
                 * @brief Re-arms the plan, runs it, and waits for the device.
                 *
                 * With a trailing recorder the replay is bracketed by its
                 * @c before_replay() and @c after_replay(), so its per-step
                 * numbers are refreshed while nothing is running.
                 *
                 * A no-op for an empty graph: the execution engine rejects a
                 * plan with no roots, and a graph of only leaves legitimately
                 * has none.
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
                std::size_t compile_count_ = 0;
                std::size_t run_count_ = 0;
                std::size_t operations_ = 0;
                bool empty_ = false;
        };

    } // namespace Graph
} // namespace SushiAI
