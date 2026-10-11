/**************************************************************************/
/* anchor_training_step.hpp                                               */
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
 * @file anchor_training_step.hpp
 * @brief Declares AnchorTrainingStep, the two-pass training step of a model with anchor heads.
 * @author Mustafa Garip
 * @see modules/vision/training/README.md
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/anchor_objective.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Owns a forward-only step and a training step that share one model state.
     * @pre The engine, both pools, the state and the optimizer outlive the step.
     * @see docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md, section 4.8.
     */
    class AnchorTrainingStep final
    {
        public:
            /**
             * @brief Compiles both passes and resolves the tensors the host stage touches.
             * @param probe The graph of build_probe_graph(); @p probe_pool holds no leaf of it.
             * @param built The graph build_training_graph() made with @p objective, already
             *        bound into @p training_pool by bind_parameters() over @p state.
             * @return The step; InvalidDescription for a head or a leaf a graph lacks;
             *         Incompatible for one that is not float32 or not of its head's size.
             */
            [[nodiscard]] static Result<std::unique_ptr<AnchorTrainingStep>> make(
                SushiBLAS::Engine& engine, const AnchorObjective& objective,
                const Train::TracedStep& probe, Graph::TensorPool& probe_pool,
                const Train::TrainingGraph& built, Graph::TensorPool& training_pool,
                const NN::ModelState& state, Optim::Optimizer& optimizer,
                std::vector<Train::BufferUpdateSlot> updates,
                Train::TrainingStepOptions options = {});

            AnchorTrainingStep(const AnchorTrainingStep&) = delete;
            AnchorTrainingStep& operator=(const AnchorTrainingStep&) = delete;
            AnchorTrainingStep(AnchorTrainingStep&&) = delete;
            AnchorTrainingStep& operator=(AnchorTrainingStep&&) = delete;
            ~AnchorTrainingStep() = default;

            /**
             * @brief Runs the forward-only pass, computes and feeds each delta, then trains once.
             * @param inputs A tensor with the rank, extents and dtype of the input leaf.
             * @pre The engine holds no recorded task; the copies run everything recorded.
             * @return The summed cost of the deltas fed; Incompatible for a truth of another
             *         batch or array size; InvalidDescription for a label that names no class;
             *         or the failure of a feed or of a replay.
             */
            [[nodiscard]] Result<Detect::Anchor::HeadCost> run(
                const SushiBLAS::Tensor& inputs, const Detect::GroundTruthBatch& truth);

            /** @brief Returns the delta of each head as the last run() fed it, in head order. */
            [[nodiscard]] SushiRuntime::span<const Detect::Anchor::HeadDelta> deltas()
                const noexcept;

            /** @brief Returns the forward-only step, for the heads the deltas were computed on. */
            [[nodiscard]] const Train::EvaluationStep& evaluation() const noexcept;

            /** @brief Returns the training step, for its counters. */
            [[nodiscard]] const Train::TrainingStep& training() const noexcept;

            /** @brief Returns how many times run() has finished, counted from a restored count. */
            [[nodiscard]] std::size_t run_count() const noexcept;

            /**
             * @brief Sets the run count of both passes to that of a saved run.
             * @param count What run_count() of the saved run returned.
             * @return What Graph::CompiledStep::restore_run_count() returns for either pass.
             */
            [[nodiscard]] Status restore_run_count(std::size_t count);

            /** @brief Returns the objective that describes the heads. */
            [[nodiscard]] const AnchorObjective& objective() const noexcept;

            /** @brief Returns the optimizer whose update run() applies. */
            [[nodiscard]] Optim::Optimizer& optimizer() noexcept;

        private:
            /** @brief Takes the parts make() built; the tensor lists are filled by make(). */
            AnchorTrainingStep(SushiBLAS::Engine& engine, AnchorObjective objective,
                               std::unique_ptr<Train::EvaluationStep> evaluation,
                               std::unique_ptr<Train::TrainingStep> training,
                               SushiBLAS::Tensor& input) noexcept;

            /**
             * @brief Resolves the forward-only output and the delta leaf of one head.
             * @return InvalidDescription for a missing head or leaf; Incompatible for a tensor
             *         that is not float32 or does not hold the head's element count.
             */
            [[nodiscard]] Status resolve_head(std::size_t head, const Train::TrainingGraph& built,
                                              Graph::TensorPool& training_pool);

            /** @brief Returns Incompatible or InvalidDescription for a truth run() cannot use. */
            [[nodiscard]] Status check_truth(const Detect::GroundTruthBatch& truth) const;

            /// @brief Records and runs the copy of each feed; not owned.
            SushiBLAS::Engine* engine_;

            /// @brief Describes the heads and measures the cost.
            AnchorObjective objective_;

            /// @brief Runs the model forward under batch statistics; never null.
            std::unique_ptr<Train::EvaluationStep> evaluation_;

            /// @brief Replays forward, backward and the update; never null.
            std::unique_ptr<Train::TrainingStep> training_;

            /// @brief Points at the input leaf of the training graph, which its pool owns.
            SushiBLAS::Tensor* input_;

            /// @brief Points at each head of the forward-only pass, in head order.
            std::vector<const SushiBLAS::Tensor*> heads_;

            /// @brief Points at each delta leaf of the training graph, in head order.
            std::vector<SushiBLAS::Tensor*> leaves_;

            /// @brief Holds the delta of each head; run() overwrites it.
            std::vector<Detect::Anchor::HeadDelta> deltas_;
    };

} // namespace SushiAI::Vision::Training
