/**************************************************************************/
/* anchor_review.hpp                                                      */
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
 * @file anchor_review.hpp
 * @brief Declares the review of an anchor layout against label boxes and its proposal.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/detect/anchor/anchor_clustering.hpp>
#include <SushiAI/vision/detect/anchor/anchor_fit.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Names what a review advises for the layout it was given. */
        enum class AnchorVerdict : std::uint8_t
        {
            Keep,    /**< The layout fits the boxes; the proposal gains too little. */
            Propose, /**< The layout recalls too few boxes, or the proposal fits better. */
        };

        /** @brief Holds the options of review_anchors. */
        struct ReviewOptions
        {
            /// @brief Holds the gain in mean best IoU above which the proposal is advised.
            double min_gain = 0.05;
            /// @brief Holds the best possible recall below which the proposal is advised.
            double min_recall = 0.9;
            FitOptions fit; /**< Options both fits are measured with. */
            /// @brief Holds the options of the proposal's clustering; its count is not read.
            ClusteringOptions clustering;
        };

        /** @brief Holds the fit of a layout, a proposal of the same counts, and the verdict. */
        struct AnchorReview
        {
            AnchorFit current;      /**< Fit of the layout that was reviewed. */
            AnchorLayout proposed;  /**< Layout derived from the boxes, with the same counts. */
            AnchorFit proposed_fit; /**< Fit of the proposed layout on the same boxes. */
            AnchorVerdict verdict = AnchorVerdict::Keep; /**< What the review advises. */
        };

        /**
         * @brief Measures @p current on @p boxes and derives a layout of the same counts.
         * @param boxes Widths and heights as fractions of the network input.
         * @param current Left as it is; the proposal has its anchor count and per-head counts.
         * @param strides One per head of @p current, above zero.
         * @return InvalidDescription for a bad option, layout or stride; Incompatible for no
         *         box, too few distinct boxes, or head counts that do not sum to the anchors.
         */
        [[nodiscard]] Result<AnchorReview> review_anchors(
            SushiRuntime::span<const AnchorSize> boxes, const AnchorLayout& current,
            SushiRuntime::span<const std::int64_t> strides, const ReviewOptions& options);

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
