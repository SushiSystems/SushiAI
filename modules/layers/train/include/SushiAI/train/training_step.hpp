/**************************************************************************/
/* training_step.hpp                                                      */
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
 * @file training_step.hpp
 * @brief Declares TrainingStep, which compiles one training replay with its optimizer and
 *        its buffer updates.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/node_range.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @enum OptimizerPlacement
         * @brief Whether the parameter update is part of the compiled step.
         */
        enum class OptimizerPlacement : std::uint8_t
        {
            /**
             * @brief Records the update into the same plan as forward and backward.
             * @pre @c Optim::Optimizer::is_replay_invariant() holds.
             */
            InPlan,

            /** @brief Steps the optimizer on its own after every replay, with its own flush. */
            PerStep,
        };

        /**
         * @brief Names what the host reads and writes between two parts of one step.
         *
         * A step with a host stage is compiled in two parts. The first ends at the node that
         * produces the last value of @c reads; the second holds every other node and the
         * trailing work.
         */
        struct HostStage
        {
            /** @brief The values the host reads; an empty list declares no host stage. */
            std::vector<Graph::ValueID> reads;

            /** @brief The leaves the host writes; no node of the first part reads one. */
            std::vector<Graph::ValueID> writes;
        };

        /** @brief States how a training step is compiled. */
        struct TrainingStepOptions
        {
            /** @brief Whether the optimizer's update is recorded into the compiled step. */
            OptimizerPlacement optimizer_placement = OptimizerPlacement::InPlan;

            /** @brief Picks the kernel variant of a fused node; null selects by device. */
            const Graph::FusionSelector* fusion_selector = nullptr;

            /** @brief Names what the host reads and writes inside the step, if anything. */
            HostStage host_stage{};

            /**
             * @brief Records ahead of the optimizer, such as the work that writes its
             *        gradients; null records nothing. Not owned, and outlives the step.
             */
            Graph::StepRecorder* before_update = nullptr;

            /**
             * @brief Records behind the optimizer and the buffer updates, such as the work
             *        that reads the new parameters; null records nothing. Not owned, and
             *        outlives the step.
             * @pre optimizer_placement is InPlan when it is not null.
             */
            Graph::StepRecorder* after_update = nullptr;
        };

        /**
         * @class TrainingStep
         * @brief Owns one compiled training replay, and the recorders behind the work it holds.
         */
        class TrainingStep final
        {
            public:
                /**
                 * @brief Compiles @p graph with the trailing work the options place behind it.
                 * @param engine Must outlive the step, as must @p pool and @p optimizer.
                 * @param updates The buffer updates to blend on every replay; may be empty.
                 * @return The step; InvalidDescription for an InPlan optimizer that is not
                 *         replay invariant, an after_update beside PerStep, or a host stage the
                 *         graph cannot be cut at; or the first refusal of the compilation.
                 */
                [[nodiscard]] static Result<std::unique_ptr<TrainingStep>> make(
                    SushiBLAS::Engine& engine, const Graph::Graph& graph, Graph::TensorPool& pool,
                    Optim::Optimizer& optimizer, std::vector<BufferUpdateSlot> updates,
                    TrainingStepOptions options = {});

                TrainingStep(const TrainingStep&) = delete;
                TrainingStep& operator=(const TrainingStep&) = delete;
                TrainingStep(TrainingStep&&) = delete;
                TrainingStep& operator=(TrainingStep&&) = delete;
                ~TrainingStep() = default;

                /**
                 * @brief Replays the compiled step, then steps the optimizer when it is not in
                 *        the plan.
                 * @return Success, or the failure of the replay or of the optimizer's step.
                 */
                [[nodiscard]] Status run();

                /**
                 * @brief Replays the part before the host stage, calls @p host_work, then
                 *        replays the rest as run() does.
                 * @param host_work Called as host_work() and returns Status; it may read what
                 *        HostStage::reads names and write what HostStage::writes names.
                 * @return Success, InvalidDescription for a step with no host stage, or the
                 *         failure of a replay, of @p host_work or of the optimizer's step.
                 */
                template<typename HostWork>
                [[nodiscard]] Status run(HostWork&& host_work)
                {
                    SA_TRY(run_lead());
                    SA_TRY(std::forward<HostWork>(host_work)());
                    return run_rest();
                }

                /** @brief Returns whether the step was compiled around a host stage. */
                [[nodiscard]] bool has_host_stage() const noexcept
                {
                    return lead_ != nullptr;
                }

                /**
                 * @brief Sets the run count of every part to that of a saved run.
                 * @param count What step().run_count() of the saved run returned.
                 * @return What Graph::CompiledStep::restore_run_count() returns for a part.
                 */
                [[nodiscard]] Status restore_run_count(std::size_t count);

                /** @brief Returns the part before the host stage, or null for a step with none. */
                [[nodiscard]] const Graph::CompiledStep* lead() const noexcept
                {
                    return lead_.get();
                }

                /** @brief Returns whether the optimizer's update is recorded into the step. */
                [[nodiscard]] bool optimizer_is_in_plan() const noexcept
                {
                    return optimizer_in_plan_;
                }

                /** @brief Returns the optimizer whose update the step applies. */
                [[nodiscard]] Optim::Optimizer& optimizer() noexcept
                {
                    return *optimizer_;
                }

                /** @brief Returns the compiled step; behind a host stage, the part after it. */
                [[nodiscard]] Graph::CompiledStep& step() noexcept
                {
                    return *step_;
                }

                /** @brief Returns the compiled step; behind a host stage, the part after it. */
                [[nodiscard]] const Graph::CompiledStep& step() const noexcept
                {
                    return *step_;
                }

            private:
                /**
                 * @brief Stores the optimizer and the slots; the recorders are added by make().
                 * @param optimizer Must outlive the step; not owned.
                 * @param updates The slots the buffer recorder blends.
                 * @param optimizer_in_plan Whether the optimizer is recorded into the step.
                 */
                TrainingStep(Optim::Optimizer& optimizer, std::vector<BufferUpdateSlot> updates,
                             bool optimizer_in_plan) noexcept;

                /**
                 * @brief Replays the part before the host stage.
                 * @return Success, InvalidDescription for a step with no host stage, or the
                 *         failure of the replay.
                 */
                [[nodiscard]] Status run_lead();

                /** @brief Replays step(), then steps an optimizer that is not in the plan. */
                [[nodiscard]] Status run_rest();

                /// @brief The update rule; not owned.
                Optim::Optimizer* optimizer_;

                /// @brief Whether the optimizer is recorded into the step.
                bool optimizer_in_plan_;

                /// @brief Records the buffer updates behind the step's graph.
                BufferUpdateRecorder updates_;

                /// @brief The trailing work: before_update, optimizer, updates, after_update.
                Graph::StepRecorderList recorders_;

                /// @brief The part before the host stage, or null; dies after step_.
                std::unique_ptr<Graph::CompiledStep> lead_;

                /// @brief The compiled step, declared last so it dies first; never null.
                std::unique_ptr<Graph::CompiledStep> step_;
        };

    } // namespace Train
} // namespace SushiAI
