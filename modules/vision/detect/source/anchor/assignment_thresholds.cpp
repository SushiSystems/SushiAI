/**************************************************************************/
/* assignment_thresholds.cpp                                              */
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
 * @file assignment_thresholds.cpp
 * @brief Implements the defaults of the match and ignore thresholds from an anchor fit.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/assignment_thresholds.hpp>

#include <algorithm>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Holds the smallest match threshold a fit can give. */
            constexpr double K_MATCH_THRESHOLD_MIN = 0.2;
            /** @brief Holds the largest match threshold a fit can give. */
            constexpr double K_MATCH_THRESHOLD_MAX = 1.0;
            /** @brief Holds the smallest ignore threshold a fit can give. */
            constexpr double K_IGNORE_THRESHOLD_MIN = 0.5;
            /** @brief Holds the largest ignore threshold a fit can give. */
            constexpr double K_IGNORE_THRESHOLD_MAX = 0.7;
        } // namespace

        AssignmentThresholds derive_assignment_thresholds(const AnchorFit& fit) noexcept
        {
            AssignmentThresholds thresholds;
            thresholds.match_threshold = static_cast<float>(std::clamp(
                fit.best_iou_tenth_percentile, K_MATCH_THRESHOLD_MIN, K_MATCH_THRESHOLD_MAX));
            thresholds.ignore_threshold = static_cast<float>(
                std::clamp(fit.mean_best_iou, K_IGNORE_THRESHOLD_MIN, K_IGNORE_THRESHOLD_MAX));
            return thresholds;
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
