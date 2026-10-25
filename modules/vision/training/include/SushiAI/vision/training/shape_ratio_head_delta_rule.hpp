/**************************************************************************/
/* shape_ratio_head_delta_rule.hpp                                        */
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
 * @file shape_ratio_head_delta_rule.hpp
 * @brief Declares the head delta rule of the shape-ratio assignment and its level losses.
 * @author Mustafa Garip
 * @see modules/vision/training/README.md
 */

#pragma once

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_assignment.hpp>
#include <SushiAI/vision/training/head_delta_rule.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Assigns the batch to every level with ShapeRatioAssignment, then computes each
     *        level's delta with compute_shape_ratio_delta.
     *
     * The cost's total is the sum of the levels' ShapeRatioLoss::total, its box share the
     * factored box means, non_box the rest, and stored equals total.
     */
    class ShapeRatioHeadDeltaRule final : public HeadDeltaRule
    {
        public:
            /**
             * @brief Constructs the rule with the assignment's settings.
             * @pre validate_shape_ratio_options(@p options) succeeds.
             */
            explicit ShapeRatioHeadDeltaRule(
                const Detect::Anchor::ShapeRatioOptions& options) noexcept;

            /**
             * @brief Checks every head for what the level loss reads.
             * @return Success, or InvalidDescription for a head that is not of logits, a head
             *         whose size form is not SquaredSigmoid, or a layout of more levels than
             *         the objectness balance weighs.
             */
            [[nodiscard]] Status check(
                const Detect::Anchor::AnchorLayout& layout,
                SushiRuntime::span<const AnchorHead> heads) const override;

            /**
             * @brief Assigns and computes every level of @p batch.
             * @return The summed cost, or the first failure of the assignment, the delta or
             *         the loss.
             */
            [[nodiscard]] Result<Detect::Anchor::HeadCost> compute(
                const HeadDeltaBatch& batch,
                SushiRuntime::span<Detect::Anchor::HeadDelta> deltas) const override;

        private:
            /// @brief Assigns the truths of a batch to the slots of one level.
            Detect::Anchor::ShapeRatioAssignment assignment_;
    };

} // namespace SushiAI::Vision::Training
