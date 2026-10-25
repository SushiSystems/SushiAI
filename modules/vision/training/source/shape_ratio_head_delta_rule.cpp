/**************************************************************************/
/* shape_ratio_head_delta_rule.cpp                                        */
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
 * @file shape_ratio_head_delta_rule.cpp
 * @brief Implements ShapeRatioHeadDeltaRule over the shape-ratio assignment and level loss.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/shape_ratio_head_delta_rule.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_delta.hpp>
#include <SushiAI/vision/detect/anchor/target_assignment.hpp>

namespace SushiAI::Vision::Training
{
    ShapeRatioHeadDeltaRule::ShapeRatioHeadDeltaRule(
        const Detect::Anchor::ShapeRatioOptions& options) noexcept
        : assignment_(options)
    {

    }

    Status ShapeRatioHeadDeltaRule::check(const Detect::Anchor::AnchorLayout& layout,
                                          SushiRuntime::span<const AnchorHead> heads) const
    {
        SA_FAIL_IF(static_cast<std::int64_t>(layout.head_anchors.size())
                       > Detect::Anchor::K_MAX_BALANCED_LEVELS,
                   StatusCode::InvalidDescription,
                   "the shape-ratio assignment weighs at most {} levels; the layout has {}",
                   Detect::Anchor::K_MAX_BALANCED_LEVELS, layout.head_anchors.size());

        for (const AnchorHead& head : heads)
        {
            SA_FAIL_IF(head.options.coding != Detect::Anchor::HeadCoding::Logit,
                       StatusCode::InvalidDescription,
                       "the shape-ratio assignment trains heads of logits; head '{}' has "
                       "another coding",
                       head.output);
            SA_FAIL_IF(head.options.size != Detect::Anchor::SizeForm::SquaredSigmoid,
                       StatusCode::InvalidDescription,
                       "the shape-ratio assignment decodes sides as squared sigmoids; head '{}' "
                       "needs 'size': 'squared_sigmoid'",
                       head.output);
        }

        return Status();
    }

    Result<Detect::Anchor::HeadCost> ShapeRatioHeadDeltaRule::compute(
        const HeadDeltaBatch& batch, SushiRuntime::span<Detect::Anchor::HeadDelta> deltas) const
    {
        SA_FATAL_IF(batch.outputs.size() != batch.heads.size()
                        || deltas.size() != batch.heads.size(),
                    "the shape-ratio rule was given {} outputs and {} deltas for {} heads",
                    batch.outputs.size(), deltas.size(), batch.heads.size());

        Detect::Anchor::HeadCost sum;
        std::vector<Detect::Anchor::TargetAssignment> assignments;
        std::vector<float> targets;

        for (std::size_t head = 0; head < batch.heads.size(); ++head)
        {
            const AnchorHead& described = batch.heads[head];
            const std::int64_t level = static_cast<std::int64_t>(head);

            Detect::Anchor::AssignmentInput assigned;
            assigned.layout = batch.layout;
            assigned.head_index = level;
            assigned.shape = described.shape;
            assigned.classes = described.options.num_classes;
            assigned.truth = batch.truth;
            assigned.network = batch.network;
            SA_TRY(assignment_.assign(assigned, assignments));

            Detect::Anchor::ShapeRatioDeltaInput input;
            input.head = batch.outputs[head];
            input.shape = described.shape;
            input.options = described.options;
            input.layout = batch.layout;
            input.head_index = level;
            input.training = &described.training;
            input.network = batch.network;
            input.assignments = SushiRuntime::span<const Detect::Anchor::TargetAssignment>(
                assignments.data(), assignments.size());

            SA_TRY(Detect::Anchor::build_objectness_targets(input, targets));
            SA_TRY_ASSIGN(const Detect::Anchor::ShapeRatioLoss loss,
                          Detect::Anchor::measure_shape_ratio_loss(
                              input, SushiRuntime::span<const float>(targets.data(),
                                                                     targets.size())));
            SA_TRY(Detect::Anchor::compute_shape_ratio_delta(input, deltas[head]));

            const double box = loss.box * Detect::Anchor::derive_shape_ratio_factors(input).box
                             * static_cast<double>(described.shape.batch);
            sum.total += loss.total;
            sum.box += box;
            sum.non_box += loss.total - box;
            sum.stored += loss.total;
        }

        return sum;
    }

} // namespace SushiAI::Vision::Training
