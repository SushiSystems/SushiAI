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
 * @brief Declares the CompiledStep class, which lowers, records and replays one step.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/node_range.hpp>
#include <SushiAI/graph/replay_prelude.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/tensor_pool.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class CompiledStep
         * @brief Owns the SushiRuntime graph of one step and replays it.
         */
        class CompiledStep final
        {
            public:
                /**
                 * @brief Lowers the nodes @p nodes of @p graph into a step and compiles it.
                 * @param engine Must outlive the step, as must @p pool and @p trailing.
                 * @param trailing Records extra work behind the graph's.
                 * @param selector Picks the form of a fusible node; null selects by device.
                 * @param nodes The nodes the step is made of; every node by default.
                 * @return The step, or the first refusal from lowering, recording or compilation.
                 */
                [[nodiscard]] static Result<std::unique_ptr<CompiledStep>> compile(
                    SushiBLAS::Engine& engine, const Graph& graph, TensorPool& pool,
                    StepRecorder* trailing = nullptr, const FusionSelector* selector = nullptr,
                    NodeRange nodes = {});

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
                [[nodiscard]] Status run();

                /**
                 * @brief Replays the step once, as run() does, but calls each replay prelude
                 *        with @p run_index instead of run_count().
                 *
                 * run_count() still counts the run.
                 * @return What run() returns.
                 */
                [[nodiscard]] Status run_at(std::uint64_t run_index);

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
                 * @brief Returns how many times the runtime has compiled this step.
                 * @return 1 from compile() on, and 0 for an empty step.
                 */
                [[nodiscard]] std::size_t compile_count() const noexcept
                {
                    return graph_.compile_count();
                }

                /** @brief Returns the index the next run() gives its replay preludes. */
                [[nodiscard]] std::size_t run_count() const noexcept
                {
                    return run_count_;
                }

                /**
                 * @brief Sets run_count() to what the step of a saved run had reached.
                 * @param count The runs the saved step had finished.
                 * @return InvalidDescription, with run_count(), once the step has run or
                 *         has been restored to a count above 0.
                 */
                [[nodiscard]] Status restore_run_count(std::size_t count);

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
                 * @brief Returns the report the last run() of a step that holds work refilled.
                 * @return An empty report marked failed after a run() that failed; one with no
                 *         node timing before the first run(), and without profiling.
                 */
                [[nodiscard]] const SushiRuntime::RunReport& last_report() const noexcept
                {
                    return report_;
                }

            private:
                /**
                 * @brief Builds an empty step whose recorder is bound to its own graph.
                 * @param engine Must outlive the step.
                 */
                CompiledStep(SushiBLAS::Engine& engine, StepRecorder* trailing);

                /**
                 * @brief Adds the hooks graph_ calls before each replay: one per prelude, in
                 *        the order given, then one for the trailing recorder.
                 * @return Success, or OutOfMemory when a hook could not be stored.
                 */
                [[nodiscard]] Status add_replay_hooks(std::vector<ReplayPrelude> preludes);

                /// @brief The recorder of the trailing work, or null; not owned.
                StepRecorder* trailing_;

                /// @brief Keeps the late-count tasks of graph_; declared first so it dies last.
                SushiBLAS::LateCountHold late_counts_;

                /// @brief The graph the step records into, compiles once and replays.
                SushiRuntime::Graph graph_;

                /// @brief The recorder bound to the recorder of graph_.
                SushiBLAS::TaskRecorder recorder_;

                /// @brief The report each run() refills.
                SushiRuntime::RunReport report_;

                /// @brief The fusion decisions of the lowering, in execution order.
                std::vector<FusionSelection> selections_;

                /// @brief The number of replay preludes graph_ holds as hooks.
                std::size_t prelude_count_ = 0;

                /// @brief The index the replay preludes of the coming replay are called with.
                std::uint64_t prelude_index_ = 0;

                /// @brief The runs finished, counted from what restore_run_count() set.
                std::size_t run_count_ = 0;

                /// @brief The number of tasks the lowering and the trailing recorder recorded.
                std::size_t operations_ = 0;

                /// @brief Whether nothing was recorded on graph_.
                bool empty_ = false;
        };

    } // namespace Graph
} // namespace SushiAI
