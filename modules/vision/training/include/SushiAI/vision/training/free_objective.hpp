/**************************************************************************/
/* free_objective.hpp                                                     */
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
 * @file free_objective.hpp
 * @brief Declares FreeObjective, the training objective of an anchor-free detector.
 * @author Mustafa Garip
 * @see modules/vision/training/README.md
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/training/free_assignment_stage.hpp>
#include <SushiAI/vision/training/free_branch_weights.hpp>
#include <SushiAI/vision/training/free_detection_loss_host.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds the "type" of an objective section that trains an anchor-free detector. */
    inline constexpr std::string_view K_FREE_OBJECTIVE_TYPE = "anchor_free";

    /** @brief Holds the role and debug name of the `[reg_max, 1]` distribution projection. */
    inline constexpr std::string_view K_FREE_PROJECTION_ROLE = "free.projection";

    /** @brief Holds the role of the `[1]` weight of the one-to-many loss, with a one-to-one copy. */
    inline constexpr std::string_view K_FREE_ONE_TO_MANY_WEIGHT_ROLE = "free.one_to_many_weight";

    /** @brief Holds the role of the `[1]` weight of the one-to-one loss, with a one-to-one copy. */
    inline constexpr std::string_view K_FREE_ONE_TO_ONE_WEIGHT_ROLE = "free.one_to_one_weight";

    /**
     * @brief Trains the one-to-many ports of an anchor-free head, and its one-to-one ports when
     *        the model has them, on the leaves its step's host stages assign.
     * @see docs/agent/2026_10_25_ULTRALYTICS_DETECTORS/PLAN.md, section 4.
     * @see docs/agent/2026_10_26_ONE_TO_ONE_LOSS/PLAN.md
     */
    class FreeObjective final : public Train::Objective
    {
        public:
            /**
             * @brief Checks the description and builds the objective.
             * @param batch Images per step, B.
             * @param weights Mix the two losses; read only when the model has a one-to-one copy.
             * @return The objective, or the refusal of FreeAssignmentStage::make(),
             *         validate_free_loss_gains() or validate_free_branch_weights().
             */
            [[nodiscard]] static Result<FreeObjective> make(FreeModelFacts facts,
                                                            Detect::Free::FreeTask task,
                                                            std::int64_t batch,
                                                            FreeLossGains gains,
                                                            FreeBranchWeights weights = {});

            /**
             * @brief Declares one float32 leaf per stage leaf the loss reads, the projection
             *        when reg_max is above 1, and the two branch weights with a one-to-one copy.
             */
            void declare(const Graph::GraphBuilder& builder,
                         Train::ObjectiveLeaves& leaves) const override;

            /**
             * @brief Emits emit_free_detection_loss() over the one-to-many ports and, with a
             *        one-to-one copy, over its ports too, each total times its weight leaf.
             * @return The id of the `[1]` total.
             * @pre @p outputs are the outputs the facts were derived from.
             */
            [[nodiscard]] Result<Graph::ValueID> loss(
                const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                const NN::ModelOutputs& outputs,
                const Train::ObjectiveLeaves& leaves) const override;

            /** @brief Returns K_FREE_OBJECTIVE_TYPE. */
            std::string_view name() const noexcept override;

            /** @brief Returns whether the loss reads leaves of @p kind under the gains. */
            [[nodiscard]] bool reads_leaf(FreeLeafKind kind) const noexcept;

            /** @brief Returns the leaves of both assignment stages, read or not, one-to-many first. */
            [[nodiscard]] const std::vector<FreeLeaf>& stage_leaves() const noexcept;

            /** @brief Returns the facts the traced outputs state. */
            [[nodiscard]] const FreeModelFacts& facts() const noexcept;

            /** @brief Returns the task section the assigner settings come from. */
            [[nodiscard]] const Detect::Free::FreeTask& task() const noexcept;

            /** @brief Returns the images per step, B. */
            [[nodiscard]] std::int64_t batch() const noexcept;

            /** @brief Returns the factor of each loss term. */
            [[nodiscard]] const FreeLossGains& gains() const noexcept;

            /** @brief Returns the weights that mix the two losses. */
            [[nodiscard]] const FreeBranchWeights& branch_weights() const noexcept;

        private:
            /** @brief Stores a description make() has checked. */
            FreeObjective(FreeModelFacts facts, Detect::Free::FreeTask task, std::int64_t batch,
                          FreeLossGains gains, FreeBranchWeights weights,
                          std::vector<FreeLeaf> stage_leaves) noexcept;

            /**
             * @brief Emits the loss of one copy of the head over its ports and its leaves.
             * @return The id of the copy's `[1]` total; the failure of an emission the builder
             *         refuses.
             */
            [[nodiscard]] Result<Graph::ValueID> branch_loss(
                const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                const NN::ModelOutputs& outputs, const Train::ObjectiveLeaves& leaves,
                FreeBranch branch) const;

            /// @brief Holds the levels, ports, reg_max and classes of the model.
            FreeModelFacts facts_;

            /// @brief Holds the assigner settings of both copies.
            Detect::Free::FreeTask task_;

            /// @brief Holds the images per step, B.
            std::int64_t batch_ = 0;

            /// @brief Holds the factor of each loss term.
            FreeLossGains gains_;

            /// @brief Holds the weights that mix the two losses.
            FreeBranchWeights weights_;

            /// @brief Holds every leaf the assignment stages fill, one-to-many first.
            std::vector<FreeLeaf> stage_leaves_;
    };

    /**
     * @brief Appends "box", "class", "distribution" and "l1" of @p terms to @p metrics.
     * @return Success, or the refusal of Train::Metrics::add().
     */
    [[nodiscard]] Status add_free_loss_metrics(const FreeLossTerms& terms,
                                               Train::Metrics& metrics);

} // namespace SushiAI::Vision::Training
