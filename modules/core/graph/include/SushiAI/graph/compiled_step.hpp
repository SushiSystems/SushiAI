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
 * @file compiled_step.hpp
 * @brief Declares the CompiledStep class, which lowers, compiles and replays one step.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/execution/execution_engine.hpp>
#include <SushiRuntime/graph/task_graph.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/tensor_pool.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class CompiledStep
         * @brief Owns the task graph of one step, compiles it once and replays its plan.
         */
        class CompiledStep final
        {
            public:
                /**
                 * @brief Lowers @p graph into a new step, compiles it once and arms it.
                 * @param engine Must outlive the step.
                 * @param pool Holds the tensor of each value; must outlive the step.
                 * @param trailing Records extra work behind the graph's; must outlive the step.
                 * @param selector Picks the form of a fusible node; null selects by device.
                 * @return The step, or the first refusal of the recording or the compilation.
                 */
                [[nodiscard]] static Result<std::unique_ptr<CompiledStep>> compile(
                    SushiBLAS::Engine& engine, const Graph& graph, TensorPool& pool,
                    StepRecorder* trailing = nullptr, const FusionSelector* selector = nullptr);

                CompiledStep(const CompiledStep&) = delete;
                CompiledStep& operator=(const CompiledStep&) = delete;
                CompiledStep(CompiledStep&&) = delete;
                CompiledStep& operator=(CompiledStep&&) = delete;
                ~CompiledStep() = default;

                /** @brief Re-arms the plan, runs it and waits until it has finished. */
                [[nodiscard]] Status run();

                /** @brief Returns whether a trailing recorder added work to this step. */
                [[nodiscard]] bool has_trailing_work() const noexcept
                {
                    return trailing_ != nullptr;
                }

                /** @brief Returns how many times this step has been compiled. */
                [[nodiscard]] std::size_t compile_count() const noexcept
                {
                    return compile_count_;
                }

                /** @brief Returns how many times run() has executed the plan. */
                [[nodiscard]] std::size_t run_count() const noexcept
                {
                    return run_count_;
                }

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

                /** @brief Returns what the selector chose for each fused node. */
                [[nodiscard]] SushiRuntime::span<const FusionSelection> selections() const noexcept
                {
                    return SushiRuntime::span<const FusionSelection>(selections_.data(),
                                                                     selections_.size());
                }

                /**
                 * @brief Returns the compiled plan, for the profiling calls of the execution
                 *        engine.
                 * @pre The caller does not compile, re-arm or execute the plan.
                 */
                [[nodiscard]] SushiRuntime::Execution::CompiledGraph& plan() noexcept
                {
                    return plan_;
                }

            private:
                /**
                 * @brief Builds an empty step whose recorder is bound to its own graph.
                 * @param engine Must outlive the step.
                 */
                CompiledStep(SushiBLAS::Engine& engine, StepRecorder* trailing);

                /// @brief The engine whose context runs the plan.
                SushiBLAS::Engine* engine_;

                /// @brief The recorder of the trailing work, or null; not owned.
                StepRecorder* trailing_;

                /// @brief The task graph the step records into and compiles.
                SushiRuntime::Graph::TaskGraph graph_;

                /// @brief The recorder bound to graph_.
                SushiBLAS::TaskRecorder recorder_;

                /// @brief The plan compiled from graph_.
                SushiRuntime::Execution::CompiledGraph plan_;

                /// @brief The fusion decisions of the lowering, in execution order.
                std::vector<FusionSelection> selections_;

                /// @brief The number of times graph_ has been compiled.
                std::size_t compile_count_ = 0;

                /// @brief The number of times run() has executed the plan.
                std::size_t run_count_ = 0;

                /// @brief The number of tasks the lowering and the trailing recorder recorded.
                std::size_t operations_ = 0;

                /// @brief Whether the plan holds no root.
                bool empty_ = false;
        };

    } // namespace Graph
} // namespace SushiAI
