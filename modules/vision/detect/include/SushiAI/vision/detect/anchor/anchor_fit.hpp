/**************************************************************************/
/* anchor_fit.hpp                                                         */
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
 * @file anchor_fit.hpp
 * @brief Declares the report of how well an anchor layout fits a set of label boxes.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the options of measure_anchor_fit. */
        struct FitOptions
        {
            /// @brief Holds the smallest shape_iou at which an owned anchor recalls a box.
            float recall_iou = 0.25f;
        };

        /** @brief Holds what a set of label boxes measures under an anchor layout. */
        struct AnchorFit
        {
            /// @brief Holds the mean over boxes of the best shape_iou with any anchor.
            double mean_best_iou = 0.0;
            /// @brief Holds the share of boxes whose best anchor a head owns at recall_iou.
            double best_possible_recall = 0.0;
            /// @brief Holds the share of boxes whose best anchor no head owns.
            double unowned_share = 0.0;
            /// @brief Holds the best shape_iou at rank ceil(boxes / 10) in ascending order.
            double best_iou_tenth_percentile = 0.0;
            /// @brief Holds, per anchor, the share of boxes that anchor fits best.
            std::vector<double> anchor_share;
            std::int64_t boxes = 0; /**< Number of boxes measured. */
        };

        /**
         * @brief Measures how well the anchors of @p layout fit @p boxes.
         * @param boxes Widths and heights as fractions of the network input.
         * @return The fit; InvalidDescription for a layout validate_anchor_layout refuses or a
         *         recall_iou outside [0, 1]; Incompatible for an empty box list. Of two anchors
         *         that fit a box equally the lower index is its best anchor.
         */
        [[nodiscard]] Result<AnchorFit> measure_anchor_fit(
            SushiRuntime::span<const AnchorSize> boxes, const AnchorLayout& layout,
            const FitOptions& options);

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI

