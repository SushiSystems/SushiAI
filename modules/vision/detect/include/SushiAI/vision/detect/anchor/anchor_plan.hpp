/**************************************************************************/
/* anchor_plan.hpp                                                        */
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
 * @file anchor_plan.hpp
 * @brief Declares the one call that derives a new model's anchoring from label boxes.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/detect/anchor/anchor_count.hpp>
#include <SushiAI/vision/detect/anchor/anchor_fit.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <SushiAI/vision/detect/anchor/assignment_thresholds.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the options of plan_anchors. */
        struct AnchorPlanOptions
        {
            /// @brief Holds the anchors each head takes; 0 has the count chosen from the boxes.
            std::int64_t per_head = 0;
            /**
             * @brief Holds the options of the count and of its clustering.
             *
             * A per_head above 0 reads the clustering options alone.
             */
            CountOptions count;
            FitOptions fit; /**< Options the fit of the plan is measured with. */
        };

        /** @brief Holds the anchoring of a new model: layout, fit, thresholds and count trials. */
        struct AnchorPlan
        {
            AnchorLayout layout; /**< The derived anchors and the anchors each head takes. */
            AnchorFit fit;       /**< Fit of the layout on the boxes it was derived from. */
            /// @brief Holds the match and ignore thresholds derived from the fit.
            AssignmentThresholds thresholds;
            /// @brief Holds the counts per head that were tried; empty when per_head was given.
            std::vector<CountTrial> trials;
        };

        /**
         * @brief Derives anchors from @p boxes and assigns them to the heads of @p strides.
         * @param boxes Widths and heights as fractions of the network input.
         * @param strides One per head, above zero; head `k` of the layout has `strides[k]`.
         * @return InvalidDescription for a bad option or stride, or no stride; Incompatible
         *         for no box or fewer distinct boxes than a count needs. The same arguments
         *         give the same plan.
         */
        [[nodiscard]] Result<AnchorPlan> plan_anchors(
            SushiRuntime::span<const AnchorSize> boxes,
            SushiRuntime::span<const std::int64_t> strides, const AnchorPlanOptions& options) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
