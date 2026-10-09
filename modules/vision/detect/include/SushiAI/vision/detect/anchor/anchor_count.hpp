/**************************************************************************/
/* anchor_count.hpp                                                       */
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
 * @file anchor_count.hpp
 * @brief Declares the choice of the number of anchors each head takes.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/detect/anchor/anchor_clustering.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the options of choose_anchor_count. */
        struct CountOptions
        {
            std::int64_t min_per_head = 1; /**< First count per head tried; at least 1. */
            /// @brief Holds the largest count per head tried; at least min_per_head.
            std::int64_t max_per_head = 5;
            /// @brief Holds the smallest gain in mean best IoU one more anchor per head must bring.
            double min_gain = 0.02;
            /// @brief Holds the options of every clustering tried; its count is not read.
            ClusteringOptions clustering;
        };

        /** @brief Holds one count per head that was tried and the fit it reached. */
        struct CountTrial
        {
            std::int64_t per_head = 0;  /**< Anchors per head of this trial. */
            double mean_best_iou = 0.0; /**< Mean best shape_iou of its clustering. */
        };

        /** @brief Holds the chosen count per head, its clustering and every trial made. */
        struct AnchorCountChoice
        {
            std::int64_t per_head = 0; /**< The chosen number of anchors per head. */
            /// @brief Holds the trials in ascending count, the refused successor last.
            std::vector<CountTrial> trials;
            /// @brief Holds the clustering of per_head times the number of heads anchors.
            AnchorClustering clustering;
        };

        /**
         * @brief Chooses the smallest count per head whose successor gains less than min_gain.
         * @param boxes Widths and heights as fractions of the network input.
         * @param heads Number of heads of the model; at least 1.
         * @return The choice; InvalidDescription for @p heads or an option outside its range;
         *         the failure of cluster_anchors for a count that was tried, Incompatible when
         *         the boxes hold fewer distinct sizes than that count.
         */
        [[nodiscard]] Result<AnchorCountChoice> choose_anchor_count(
            SushiRuntime::span<const AnchorSize> boxes, std::int64_t heads,
            const CountOptions& options);

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
