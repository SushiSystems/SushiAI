/**************************************************************************/
/* assignment_thresholds.hpp                                              */
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
 * @file assignment_thresholds.hpp
 * @brief Declares the match and ignore thresholds of anchor assignment and their defaults.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <SushiAI/vision/detect/anchor/anchor_fit.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the two thresholds that decide which slots a truth box trains. */
        struct AssignmentThresholds
        {
            /// @brief Holds the shape_iou a further anchor of a head must exceed to take a truth.
            float match_threshold = 1.0f;
            /// @brief Holds the box IoU above which an unassigned slot is left out of the cost.
            float ignore_threshold = 0.7f;
        };

        /**
         * @brief Derives the thresholds a dataset with the fit @p fit trains under.
         * @return match_threshold as the tenth percentile of the best shape_iou clamped to
         *         [0.2, 1]; ignore_threshold as the mean best IoU clamped to [0.5, 0.7].
         */
        [[nodiscard]] AssignmentThresholds derive_assignment_thresholds(
            const AnchorFit& fit) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
