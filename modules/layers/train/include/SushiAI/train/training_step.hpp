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

#include <cstdint>
#include <memory>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
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

        /** @brief States how a training step is compiled. */
        struct TrainingStepOptions
        {
            /** @brief Whether the optimizer's update is recorded into the compiled step. */
            OptimizerPlacement optimizer_placement = OptimizerPlacement::InPlan;

            /** @brief Picks the kernel variant of a fused node; null selects by device. */
            const Graph::FusionSelector* fusion_selector = nullptr;
        };

        /**
         * @class TrainingStep
         * @brief Owns one compiled training replay, and the recorders behind the work it holds.
         */
        class TrainingStep final
        {
            public:
                /**
                 * @brief Compiles @p graph with the optimizer and the buffer updates recorded
                 *        behind it, as the options place them.
                 * @param engine Must outlive the step, as must @p pool and @p optimizer.
                 * @param updates The buffer updates to blend on every replay; may be empty.
                 * @return The step, InvalidDescription when InPlan is asked of an optimizer that
                 *         is not replay invariant, or the first refusal of the compilation.
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

                /** @brief Returns whether the optimizer's update is recorded into the step. */
                [[nodiscard]] bool optimizer_is_in_plan() const noexcept
                {
                    return optimizer_in_plan_;
                }

                /** @brief Returns the compiled step. */
                [[nodiscard]] Graph::CompiledStep& step() noexcept
                {
                    return *step_;
                }

                /** @brief Returns the compiled step. */
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

                /// @brief The update rule; not owned.
                Optim::Optimizer* optimizer_;

                /// @brief Whether the optimizer is recorded into the step.
                bool optimizer_in_plan_;

                /// @brief Records the buffer updates behind the step's graph.
                BufferUpdateRecorder updates_;

                /// @brief The trailing work of the step, in order: optimizer, then updates.
                Graph::StepRecorderList recorders_;

                /// @brief The compiled step, declared last so it dies first; never null.
                std::unique_ptr<Graph::CompiledStep> step_;
        };

    } // namespace Train
} // namespace SushiAI
