/**************************************************************************/
/* free_training_step.hpp                                                 */
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
 * @file free_training_step.hpp
 * @brief Declares FreeTrainingStep, the two-part training step of an anchor-free detector.
 * @author Mustafa Garip
 * @see modules/vision/training/README.md
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/free_assignment_stage.hpp>
#include <SushiAI/vision/training/free_detection_loss_host.hpp>
#include <SushiAI/vision/training/free_objective.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Owns a training step that stops after the one-to-many ports, so the host
     *        assigns the targets from the forward pass the step differentiates.
     * @pre The engine, the pool and the optimizer outlive the step.
     * @see docs/agent/2026_10_25_ULTRALYTICS_DETECTORS/PLAN.md, section 4.
     */
    class FreeTrainingStep final
    {
        public:
            /**
             * @brief Compiles the step around its host stage and resolves that stage's tensors.
             * @param built Built with @p objective and bound into @p training_pool.
             * @param options Its host stage is replaced by the ports and the leaves.
             * @return The step; InvalidDescription for a port or a leaf with no value or
             *         tensor; Incompatible for one not float32 or not of its size; or the
             *         refusal of the stage or the compilation.
             */
            [[nodiscard]] static Result<std::unique_ptr<FreeTrainingStep>> make(
                SushiBLAS::Engine& engine, const FreeObjective& objective,
                const Train::TrainingGraph& built, Graph::TensorPool& training_pool,
                Optim::Optimizer& optimizer, std::vector<Train::BufferUpdateSlot> updates,
                Train::TrainingStepOptions options = {});

            FreeTrainingStep(const FreeTrainingStep&) = delete;
            FreeTrainingStep& operator=(const FreeTrainingStep&) = delete;
            FreeTrainingStep(FreeTrainingStep&&) = delete;
            FreeTrainingStep& operator=(FreeTrainingStep&&) = delete;
            ~FreeTrainingStep() = default;

            /**
             * @brief Runs the step up to the ports, assigns @p truth, writes every leaf, then
             *        runs the rest.
             * @pre The engine holds no recorded task.
             * @return The loss terms the host evaluates over the leaves fed; Incompatible for
             *         a truth of another batch or size; InvalidDescription for a label that
             *         names no class; or the failure of a feed, the stage or a replay.
             */
            [[nodiscard]] Result<FreeLossTerms> run(const SushiBLAS::Tensor& inputs,
                                                    const Detect::GroundTruthBatch& truth);

            /** @brief Returns the stage as the last run() left it. */
            [[nodiscard]] const FreeAssignmentStage& stage() const noexcept;

            /** @brief Returns the training step, for its counters. */
            [[nodiscard]] const Train::TrainingStep& training() const noexcept;

            /** @brief Returns how many times run() has finished, counted from a restored count. */
            [[nodiscard]] std::size_t run_count() const noexcept;

            /**
             * @brief Sets the run count of both parts of the step to that of a saved run.
             * @return What Train::TrainingStep::restore_run_count() returns.
             */
            [[nodiscard]] Status restore_run_count(std::size_t count);

            /** @brief Returns the objective that describes the ports and the leaves. */
            [[nodiscard]] const FreeObjective& objective() const noexcept;

            /** @brief Returns the optimizer whose update run() applies. */
            [[nodiscard]] Optim::Optimizer& optimizer() noexcept;

        private:
            /** @brief Holds one level's ports and leaves in double, as the host loss reads. */
            struct LevelDoubles
            {
                std::vector<double> boxes;                /**< `4 * reg_max` per row. */
                std::vector<double> scores;               /**< C per row. */
                std::vector<double> target_scores;        /**< C per row. */
                std::vector<double> box_mask;             /**< 4 per row. */
                std::vector<double> normalized_weight;    /**< 1 per row. */
                std::vector<double> target_boxes;         /**< 4 per row. */
                std::vector<double> target_distances;     /**< 4 per row; l1_gain above 0. */
                std::vector<double> l1_scale;             /**< 4 per row; l1_gain above 0. */
                std::vector<double> safe_boxes;           /**< 4 per row. */
                std::vector<double> distribution_weights; /**< `4 * reg_max`; reg_max above 1. */
                std::vector<double> anchor_points;        /**< 2 per row. */
            };

            /** @brief Takes the parts make() built; the tensor lists are filled by make(). */
            FreeTrainingStep(SushiBLAS::Engine& engine, FreeObjective objective,
                             FreeAssignmentStage stage, Graph::TensorPool& pool,
                             SushiBLAS::Tensor& input) noexcept;

            /**
             * @brief Names every one-to-many port and every declared leaf.
             * @return The stage; InvalidDescription for a port the model did not publish.
             */
            [[nodiscard]] Result<Train::HostStage> describe_host_stage(
                const Train::TrainingGraph& built);

            /**
             * @brief Resolves the tensors of the ports and the leaves of the compiled step.
             * @return InvalidDescription for a value with no tensor; Incompatible for a tensor
             *         that is not float32 or does not hold its described count.
             */
            [[nodiscard]] Status resolve_tensors(const Train::TrainingGraph& built);

            /** @brief Returns Incompatible or InvalidDescription for a truth run() cannot use. */
            [[nodiscard]] Status check_truth(const Detect::GroundTruthBatch& truth) const;

            /** @brief Assigns @p truth over the ports and writes every leaf and the projection. */
            [[nodiscard]] Status feed_leaves(const Detect::GroundTruthBatch& truth);

            /** @brief Evaluates the loss terms over the ports and the leaves just written. */
            [[nodiscard]] FreeLossTerms measure_terms();

            /// @brief Records and runs the copy of the input; not owned.
            SushiBLAS::Engine* engine_;

            /// @brief Describes the ports, the leaves and the gains.
            FreeObjective objective_;

            /// @brief Decodes the ports, assigns the truth and fills the leaves.
            FreeAssignmentStage stage_;

            /// @brief Holds the tensor of every value of the training graph; not owned.
            Graph::TensorPool* pool_;

            /// @brief Replays the step around the host stage; never null once make() returns.
            std::unique_ptr<Train::TrainingStep> training_;

            /// @brief Points at the input leaf of the training graph, which its pool owns.
            SushiBLAS::Tensor* input_;

            /// @brief Holds the port ids, boxes then scores of each level, finest first.
            std::vector<Graph::ValueID> port_ids_;

            /// @brief Points at the tensor of each entry of port_ids_.
            std::vector<const SushiBLAS::Tensor*> ports_;

            /// @brief Points at the tensor of each stage leaf; null for one the loss skips.
            std::vector<SushiBLAS::Tensor*> leaves_;

            /// @brief Points at the projection leaf; null at reg_max 1.
            SushiBLAS::Tensor* projection_ = nullptr;

            /// @brief Holds a view of each level's ports as the stage reads them.
            std::vector<FreeLevelViews> views_;

            /// @brief Holds one level at a time in double for measure_terms().
            LevelDoubles doubles_;
    };

} // namespace SushiAI::Vision::Training
