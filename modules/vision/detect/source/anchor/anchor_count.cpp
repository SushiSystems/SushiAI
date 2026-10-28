/**************************************************************************/
/* anchor_count.cpp                                                       */
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
 * @file anchor_count.cpp
 * @brief Implements the choice of the number of anchors each head takes.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/anchor_count.hpp>

#include <cmath>
#include <limits>
#include <utility>

#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /**
             * @brief Derives @p per_head anchors for each of @p heads heads from @p boxes.
             * @pre @p per_head times @p heads fits in 64 bits.
             */
            Result<AnchorClustering> cluster_for_count(SushiRuntime::span<const AnchorSize> boxes,
                                                       std::int64_t heads, std::int64_t per_head,
                                                       const ClusteringOptions& options)
            {
                ClusteringOptions counted = options;
                counted.count = per_head * heads;
                return cluster_anchors(boxes, counted);
            }
        } // namespace

        Result<AnchorCountChoice> choose_anchor_count(SushiRuntime::span<const AnchorSize> boxes,
                                                      std::int64_t heads,
                                                      const CountOptions& options) noexcept
        {
            SA_FAIL_IF(heads < 1, StatusCode::InvalidDescription,
                       "heads must be at least 1; got {}", heads);
            SA_FAIL_IF(options.min_per_head < 1, StatusCode::InvalidDescription,
                       "min_per_head must be at least 1; got {}", options.min_per_head);
            SA_FAIL_IF(options.min_per_head > options.max_per_head,
                       StatusCode::InvalidDescription,
                       "min_per_head must not exceed max_per_head; got {} and {}",
                       options.min_per_head, options.max_per_head);
            SA_FAIL_IF(!(std::isfinite(options.min_gain) && options.min_gain >= 0.0),
                       StatusCode::InvalidDescription,
                       "min_gain must be finite and not negative; got {}", options.min_gain);
            SA_FAIL_IF(options.max_per_head > std::numeric_limits<std::int64_t>::max() / heads,
                       StatusCode::InvalidDescription,
                       "max_per_head {} times {} heads does not fit in 64 bits",
                       options.max_per_head, heads);

            AnchorCountChoice choice;
            choice.per_head = options.min_per_head;
            SA_TRY_ASSIGN(choice.clustering,
                          cluster_for_count(boxes, heads, choice.per_head, options.clustering));
            choice.trials.push_back(CountTrial{choice.per_head, choice.clustering.mean_best_iou});

            while (choice.per_head < options.max_per_head)
            {
                const std::int64_t successor = choice.per_head + 1;
                SA_TRY_ASSIGN(AnchorClustering larger,
                              cluster_for_count(boxes, heads, successor, options.clustering));
                choice.trials.push_back(CountTrial{successor, larger.mean_best_iou});

                if (larger.mean_best_iou - choice.clustering.mean_best_iou < options.min_gain)
                    break;

                choice.per_head = successor;
                choice.clustering = std::move(larger);
            }
            return choice;
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
