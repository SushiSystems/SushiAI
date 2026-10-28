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
#include <string_view>
#include <vector>

#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/anchor_objective.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Owns a training step that stops after its heads, so the host computes each
     *        delta from the forward pass the step differentiates.
     * @pre The engine, the pool and the optimizer outlive the step.
     * @see docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md, section 4.8.
     */
    class AnchorTrainingStep final
    {
        public:
            /**
             * @brief Compiles the step around its host stage and resolves that stage's tensors.
             * @param built The graph build_training_graph() made with @p objective, already
             *        bound into @p training_pool by bind_parameters().
             * @param options Its host stage is replaced by the heads and the delta leaves.
             * @return The step; InvalidDescription for a head or a leaf the graph lacks;
             *         Incompatible for one that is not float32 or not of its head's size.
             */
            [[nodiscard]] static Result<std::unique_ptr<AnchorTrainingStep>> make(
                SushiBLAS::Engine& engine, const AnchorObjective& objective,
                const Train::TrainingGraph& built, Graph::TensorPool& training_pool,
                Optim::Optimizer& optimizer, std::vector<Train::BufferUpdateSlot> updates,
                Train::TrainingStepOptions options = {}) noexcept;

            AnchorTrainingStep(const AnchorTrainingStep&) = delete;
            AnchorTrainingStep& operator=(const AnchorTrainingStep&) = delete;
            AnchorTrainingStep(AnchorTrainingStep&&) = delete;
            AnchorTrainingStep& operator=(AnchorTrainingStep&&) = delete;
            ~AnchorTrainingStep() = default;

            /**
             * @brief Runs the step up to its heads, feeds each delta, then runs the rest.
             * @param inputs A tensor with the rank, extents and dtype of the input leaf.
             * @pre The engine holds no recorded task; the copies run everything recorded.
             * @return The cost the objective's rule reported for the deltas fed; Incompatible
             *         for a truth of another batch or array size; InvalidDescription for a
             *         label that names no class; or the failure of a feed or of a replay.
             */
            [[nodiscard]] Result<Detect::Anchor::HeadCost> run(
                const SushiBLAS::Tensor& inputs, const Detect::GroundTruthBatch& truth) noexcept;

            /** @brief Returns the delta of each head as the last run() fed it, in head order. */
            [[nodiscard]] SushiRuntime::span<const Detect::Anchor::HeadDelta> deltas()
                const noexcept;

            /**
             * @brief Returns the tensor behind the output the model published under @p name.
             * @return A tensor the pool owns, holding what the last run() computed, or
             *         InvalidDescription for a name the model did not publish.
             */
            [[nodiscard]] Result<const SushiBLAS::Tensor*> output(std::string_view name) const noexcept;

            /** @brief Returns the training step, for its counters. */
            [[nodiscard]] const Train::TrainingStep& training() const noexcept;

            /** @brief Returns how many times run() has finished, counted from a restored count. */
            [[nodiscard]] std::size_t run_count() const noexcept;

            /**
             * @brief Sets the run count of both parts of the step to that of a saved run.
             * @param count What run_count() of the saved run returned.
             * @return What Train::TrainingStep::restore_run_count() returns.
             */
            [[nodiscard]] Status restore_run_count(std::size_t count) noexcept;

            /** @brief Returns the objective that describes the heads. */
            [[nodiscard]] const AnchorObjective& objective() const noexcept;

            /** @brief Returns the optimizer whose update run() applies. */
            [[nodiscard]] Optim::Optimizer& optimizer() noexcept;

        private:
            /** @brief Takes the parts make() built; the tensor lists are filled by make(). */
            AnchorTrainingStep(SushiBLAS::Engine& engine, AnchorObjective objective,
                               NN::ModelOutputs outputs, Graph::TensorPool& pool,
                               SushiBLAS::Tensor& input) noexcept;

            /**
             * @brief Names the head and the delta leaf of every head, in head order.
             * @return The stage; InvalidDescription for a head the model did not publish or a
             *         leaf the graph does not declare.
             */
            [[nodiscard]] Result<Train::HostStage> describe_host_stage(
                const Train::TrainingGraph& built) const;

            /**
             * @brief Resolves the output and the delta leaf of one head of the compiled step.
             * @return InvalidDescription for a leaf with no tensor; Incompatible for a tensor
             *         that is not float32 or does not hold the head's element count.
             */
            [[nodiscard]] Status resolve_head(std::size_t head, const Train::HostStage& stage);

            /**
             * @brief Computes every head's delta and the cost through the objective's rule,
             *        then writes each delta into its leaf.
             */
            [[nodiscard]] Status feed_deltas(const Detect::GroundTruthBatch& truth);

            /** @brief Returns Incompatible or InvalidDescription for a truth run() cannot use. */
            [[nodiscard]] Status check_truth(const Detect::GroundTruthBatch& truth) const;

            /// @brief Records and runs the copy of each feed; not owned.
            SushiBLAS::Engine* engine_;

            /// @brief Describes the heads and measures the cost.
            AnchorObjective objective_;

            /// @brief Names every value the model published in the training graph.
            NN::ModelOutputs outputs_;

            /// @brief Holds the tensor of every value of the training graph; not owned.
            Graph::TensorPool* pool_;

            /// @brief Replays forward, backward and the update around the host stage; never
            ///        null once make() returns.
            std::unique_ptr<Train::TrainingStep> training_;

            /// @brief Points at the input leaf of the training graph, which its pool owns.
            SushiBLAS::Tensor* input_;

            /// @brief Points at each head of the training graph, in head order.
            std::vector<const SushiBLAS::Tensor*> heads_;

            /// @brief Points at each delta leaf of the training graph, in head order.
            std::vector<SushiBLAS::Tensor*> leaves_;

            /// @brief Holds the delta of each head; run() overwrites it.
            std::vector<Detect::Anchor::HeadDelta> deltas_;

            /// @brief Holds a view of each head's values as the host stage reads them.
            std::vector<SushiRuntime::span<const float>> outputs_view_;

            /// @brief Holds the cost the objective's rule reported for the last deltas fed.
            Detect::Anchor::HeadCost cost_;
    };

} // namespace SushiAI::Vision::Training
