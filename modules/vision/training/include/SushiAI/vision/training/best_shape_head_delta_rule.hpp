/**************************************************************************/
/* best_shape_head_delta_rule.hpp                                         */
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
 * @file best_shape_head_delta_rule.hpp
 * @brief Declares the head delta rule of the best-shape assignment.
 * @author Mustafa Garip
 * @see modules/vision/training/README.md
 */

#pragma once

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/training/head_delta_rule.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Computes each head's delta and cost with compute_head_delta and measure_head_cost. */
    class BestShapeHeadDeltaRule final : public HeadDeltaRule
    {
        public:
            /**
             * @brief Checks that every head decodes its sides as exponentials.
             * @return Success, or InvalidDescription naming a head of another size form.
             */
            [[nodiscard]] Status check(
                const Detect::Anchor::AnchorLayout& layout,
                SushiRuntime::span<const AnchorHead> heads) const override;

            /**
             * @brief Runs compute_head_delta on each head, then sums measure_head_cost.
             * @return The summed cost, or the first failure of compute_head_delta.
             */
            [[nodiscard]] Result<Detect::Anchor::HeadCost> compute(
                const HeadDeltaBatch& batch,
                SushiRuntime::span<Detect::Anchor::HeadDelta> deltas) const override;
    };

} // namespace SushiAI::Vision::Training
