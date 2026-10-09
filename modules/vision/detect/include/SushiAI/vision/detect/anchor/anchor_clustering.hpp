/**************************************************************************/
/* anchor_clustering.hpp                                                  */
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
 * @file anchor_clustering.hpp
 * @brief Declares the derivation of anchors from the sizes of label boxes.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the options of cluster_anchors. */
        struct ClusteringOptions
        {
            std::int64_t count = 0; /**< Number of anchors to derive; at least 1. */
            /// @brief Holds the seed of the sequence the first centres are drawn from.
            std::uint64_t seed = 0x5355534849ull;
            /// @brief Holds the most assignment passes the clustering makes; at least 1.
            std::int64_t max_iterations = 300;
            /// @brief Holds the most sweeps the refinement makes; 0 skips the refinement.
            std::int64_t max_refinement_sweeps = 64;
            /// @brief Holds the side at or below which a box is left out; finite, not negative.
            float min_side = 0.0f;
        };

        /** @brief Holds the anchors derived from a set of label boxes and how they came about. */
        struct AnchorClustering
        {
            std::vector<AnchorSize> anchors; /**< The anchors, in ascending area. */
            /// @brief Holds the mean over the boxes used of the best shape_iou with any anchor.
            double mean_best_iou = 0.0;
            /// @brief Holds the same mean for the centres the refinement started from.
            double mean_best_iou_before_refinement = 0.0;
            std::int64_t iterations = 0;        /**< Assignment passes made. */
            std::int64_t refinement_sweeps = 0; /**< Refinement sweeps made. */
            std::int64_t boxes_used = 0;        /**< Boxes the anchors were derived from. */
            /// @brief Holds the number of boxes left out for a side that is not usable.
            std::int64_t boxes_rejected = 0;
        };

        /**
         * @brief Derives options.count anchors from @p boxes.
         * @param boxes Widths and heights as fractions of the network input.
         * @return The anchors; InvalidDescription for an option outside its range; Incompatible
         *         for fewer distinct usable boxes than options.count. The same boxes in the
         *         same order with the same options give the same anchors.
         */
        [[nodiscard]] Result<AnchorClustering> cluster_anchors(
            SushiRuntime::span<const AnchorSize> boxes, const ClusteringOptions& options);

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
