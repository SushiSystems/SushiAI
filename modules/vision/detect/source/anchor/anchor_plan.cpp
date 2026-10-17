/**************************************************************************/
/* anchor_plan.cpp                                                        */
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
 * @file anchor_plan.cpp
 * @brief Implements the one call that derives a new model's anchoring from label boxes.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/anchor_plan.hpp>

#include <limits>
#include <utility>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_clustering.hpp>
#include <SushiAI/vision/detect/anchor/head_assignment.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Holds the anchors of a plan and the count trials that led to them. */
            struct DerivedAnchors
            {
                std::vector<AnchorSize> anchors; /**< The anchors, in ascending area. */
                std::vector<CountTrial> trials;  /**< Counts tried; empty for a given count. */
            };

            /** @brief Derives the anchors of @p heads heads at a count chosen from @p boxes. */
            Result<DerivedAnchors> derive_at_chosen_count(
                SushiRuntime::span<const AnchorSize> boxes, std::int64_t heads,
                const CountOptions& options)
            {
                SA_TRY_ASSIGN(AnchorCountChoice choice,
                              choose_anchor_count(boxes, heads, options));
                return DerivedAnchors{std::move(choice.clustering.anchors),
                                      std::move(choice.trials)};
            }

            /**
             * @brief Derives @p per_head anchors for each of @p heads heads, trying no count.
             * @pre @p per_head and @p heads are at least 1.
             */
            Result<DerivedAnchors> derive_at_given_count(
                SushiRuntime::span<const AnchorSize> boxes, std::int64_t heads,
                std::int64_t per_head, const ClusteringOptions& options)
            {
                SA_FAIL_IF(per_head > std::numeric_limits<std::int64_t>::max() / heads,
                           StatusCode::InvalidDescription,
                           "per_head {} times {} heads does not fit in 64 bits", per_head, heads);

                ClusteringOptions counted = options;
                counted.count = per_head * heads;
                SA_TRY_ASSIGN(AnchorClustering clustering, cluster_anchors(boxes, counted));
                return DerivedAnchors{std::move(clustering.anchors), {}};
            }
        } // namespace

        Result<AnchorPlan> plan_anchors(SushiRuntime::span<const AnchorSize> boxes,
                                        SushiRuntime::span<const std::int64_t> strides,
                                        const AnchorPlanOptions& options)
        {
            SA_FAIL_IF(strides.empty(), StatusCode::InvalidDescription,
                       "an anchor plan needs at least one stride; got none");
            SA_FAIL_IF(options.per_head < 0, StatusCode::InvalidDescription,
                       "per_head must not be negative; got {}", options.per_head);

            const std::int64_t heads = static_cast<std::int64_t>(strides.size());
            SA_TRY_ASSIGN(DerivedAnchors derived,
                          options.per_head == 0
                              ? derive_at_chosen_count(boxes, heads, options.count)
                              : derive_at_given_count(boxes, heads, options.per_head,
                                                      options.count.clustering));

            AnchorPlan plan;
            SA_TRY_ASSIGN(plan.layout, assign_anchors_to_heads(derived.anchors, strides, {}));
            SA_TRY_ASSIGN(plan.fit, measure_anchor_fit(boxes, plan.layout, options.fit));
            plan.thresholds = derive_assignment_thresholds(plan.fit);
            plan.trials = std::move(derived.trials);
            return plan;
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
