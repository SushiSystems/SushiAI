/**************************************************************************/
/* anchor_objective.hpp                                                   */
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
 * @file anchor_objective.hpp
 * @brief Declares AnchorObjective, the training objective of a model with anchor heads.
 * @author Mustafa Garip
 * @see modules/vision/training/README.md
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/anchor_head.hpp>
#include <SushiAI/vision/training/head_delta_rule.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Trains anchor heads on a delta the host computes and feeds through one leaf a head.
     * @see docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md, section 4.6.
     */
    class AnchorObjective final : public Train::Objective
    {
        public:
            /**
             * @brief Checks the description and builds the objective with Darknet's rule.
             * @param heads One entry per head of @p layout, in the layout's head order.
             * @return The objective, or InvalidDescription for a network side or an extent
             *         below 1, a head count other than the layout's, batches that differ, an
             *         output name that is empty or repeated, or what validate_head and
             *         validate_head_training refuse.
             */
            [[nodiscard]] static Result<AnchorObjective> make(
                Detect::Anchor::NetworkSize network, Detect::Anchor::AnchorLayout layout,
                std::vector<AnchorHead> heads) noexcept;

            /**
             * @brief Checks the description and builds the objective with @p rule.
             * @param rule Computes the deltas and the cost; null is fatal.
             * @return What the other make() returns, or the refusal of @p rule's check().
             */
            [[nodiscard]] static Result<AnchorObjective> make(
                Detect::Anchor::NetworkSize network, Detect::Anchor::AnchorLayout layout,
                std::vector<AnchorHead> heads, std::shared_ptr<const HeadDeltaRule> rule) noexcept;

            /**
             * @brief Declares one float32 leaf per head, in the shape of the head's output.
             * @param leaves Receives each leaf under leaf_role() of its head.
             */
            void declare(const Graph::GraphBuilder& builder,
                         Train::ObjectiveLeaves& leaves) const override;

            /**
             * @brief Emits the sum over the heads of -(1 / batch) * sum(delta * head).
             * @return The id of the [1] root, whose value carries no meaning; or the failure
             *         when @p outputs lacks a head or a head differs from its leaf.
             */
            [[nodiscard]] Result<Graph::ValueID> loss(
                const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                const NN::ModelOutputs& outputs,
                const Train::ObjectiveLeaves& leaves) const override;

            /** @brief Returns "anchor_delta". */
            std::string_view name() const noexcept override;

            /**
             * @brief Writes the delta of every head through the objective's rule.
             * @param outputs Each head's output, in head order; another count is fatal.
             * @param deltas One per head, in head order; another count is fatal.
             * @return The batch's cost, or the refusal of @p truth.
             */
            [[nodiscard]] Result<Detect::Anchor::HeadCost> compute_deltas(
                SushiRuntime::span<const SushiRuntime::span<const float>> outputs,
                const Detect::GroundTruthBatch& truth,
                SushiRuntime::span<Detect::Anchor::HeadDelta> deltas) const noexcept;

            /**
             * @brief Returns measure_head_cost of every head summed, field by field.
             * @param deltas One delta per head, in head order; another count is fatal.
             */
            [[nodiscard]] Detect::Anchor::HeadCost measure_cost(
                SushiRuntime::span<const Detect::Anchor::HeadDelta> deltas) const noexcept;

            /**
             * @brief Returns the role and debug name of a head's leaf, "delta." and its output.
             * @pre @p head is below the head count; a larger index is fatal.
             */
            [[nodiscard]] std::string leaf_role(std::size_t head) const noexcept;

            /** @brief Returns the size the truth's pixel corners are divided by. */
            [[nodiscard]] Detect::Anchor::NetworkSize network() const noexcept;

            /** @brief Returns the anchors of the model and of each head. */
            [[nodiscard]] const Detect::Anchor::AnchorLayout& layout() const noexcept;

            /** @brief Returns the heads, in the layout's head order. */
            [[nodiscard]] const std::vector<AnchorHead>& heads() const noexcept;

            /** @brief Returns the rule that computes the deltas and the cost. */
            [[nodiscard]] const HeadDeltaRule& rule() const noexcept;

        private:
            /** @brief Stores a description make() has checked. */
            AnchorObjective(Detect::Anchor::NetworkSize network,
                            Detect::Anchor::AnchorLayout layout,
                            std::vector<AnchorHead> heads,
                            std::shared_ptr<const HeadDeltaRule> rule) noexcept;

            /// @brief Holds the size of the network input in pixels.
            Detect::Anchor::NetworkSize network_;

            /// @brief Holds every anchor and the anchors of each head.
            Detect::Anchor::AnchorLayout layout_;

            /// @brief Holds one entry per head of layout_, in its order; never empty.
            std::vector<AnchorHead> heads_;

            /// @brief Computes the deltas and the cost; shared by copies, never null.
            std::shared_ptr<const HeadDeltaRule> rule_;
    };

} // namespace SushiAI::Vision::Training
