/**************************************************************************/
/* anchor_review.cpp                                                      */
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
 * @file anchor_review.cpp
 * @brief Implements the review of an anchor layout against label boxes and its proposal.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/anchor_review.hpp>

#include <cmath>
#include <vector>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/head_assignment.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Checks that min_gain is finite and not negative and min_recall a share. */
            Status check_review_options(const ReviewOptions& options)
            {
                SA_FAIL_IF(!(std::isfinite(options.min_gain) && options.min_gain >= 0.0),
                           StatusCode::InvalidDescription,
                           "min_gain must be finite and not negative; got {}", options.min_gain);
                SA_FAIL_IF(!(options.min_recall >= 0.0 && options.min_recall <= 1.0),
                           StatusCode::InvalidDescription,
                           "min_recall must lie in [0, 1]; got {}", options.min_recall);
                return Status();
            }

            /** @brief Returns the number of anchor slots of each head of @p layout. */
            std::vector<std::int64_t> count_head_slots(const AnchorLayout& layout)
            {
                std::vector<std::int64_t> counts;
                counts.reserve(layout.head_anchors.size());
                for (const std::vector<std::int64_t>& slots : layout.head_anchors)
                    counts.push_back(static_cast<std::int64_t>(slots.size()));
                return counts;
            }

            /** @brief Returns Propose for a recall below min_recall or a gain above min_gain. */
            AnchorVerdict judge_fits(const AnchorFit& current, const AnchorFit& proposed,
                                     const ReviewOptions& options) noexcept
            {
                const double gain = proposed.mean_best_iou - current.mean_best_iou;
                if (current.best_possible_recall < options.min_recall || gain > options.min_gain)
                    return AnchorVerdict::Propose;
                return AnchorVerdict::Keep;
            }
        } // namespace

        Result<AnchorReview> review_anchors(SushiRuntime::span<const AnchorSize> boxes,
                                            const AnchorLayout& current,
                                            SushiRuntime::span<const std::int64_t> strides,
                                            const ReviewOptions& options) noexcept
        {
            SA_TRY(check_review_options(options));

            AnchorReview review;
            SA_TRY_ASSIGN(review.current, measure_anchor_fit(boxes, current, options.fit));

            ClusteringOptions clustering_options = options.clustering;
            clustering_options.count = static_cast<std::int64_t>(current.anchors.size());
            SA_TRY_ASSIGN(const AnchorClustering clustering,
                          cluster_anchors(boxes, clustering_options));

            const std::vector<std::int64_t> per_head = count_head_slots(current);
            SA_TRY_ASSIGN(review.proposed,
                          assign_anchors_to_heads(clustering.anchors, strides, per_head));
            SA_TRY_ASSIGN(review.proposed_fit,
                          measure_anchor_fit(boxes, review.proposed, options.fit));

            review.verdict = judge_fits(review.current, review.proposed_fit, options);
            return review;
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
