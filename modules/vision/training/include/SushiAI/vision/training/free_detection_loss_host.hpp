/**************************************************************************/
/* free_detection_loss_host.hpp                                           */
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

/**
 * @file free_detection_loss_host.hpp
 * @brief Declares a host evaluation of the anchor-free detection loss on double arrays.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>

#include <SushiAI/vision/training/free_loss_gains.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Views one level's ports and leaves, row-major `[B, H, W, *]`. */
    struct FreeLevelLossArrays
    {
        SushiRuntime::span<const double> boxes;                /**< `4 * reg_max` per row. */
        SushiRuntime::span<const double> scores;               /**< C logits per row. */
        SushiRuntime::span<const double> target_scores;        /**< C per row. */
        SushiRuntime::span<const double> box_mask;             /**< 4 per row. */
        SushiRuntime::span<const double> normalized_weight;    /**< 1 per row. */
        SushiRuntime::span<const double> target_boxes;         /**< 4 per row, grid units. */
        SushiRuntime::span<const double> target_distances;     /**< 4 per row; l1_gain > 0. */
        SushiRuntime::span<const double> l1_scale;             /**< 4 per row; l1_gain > 0. */
        SushiRuntime::span<const double> safe_boxes;           /**< 4 per row. */
        SushiRuntime::span<const double> distribution_weights; /**< `4 * reg_max`; reg_max > 1. */
        SushiRuntime::span<const double> anchor_points;        /**< 2 per row, grid units. */
    };

    /** @brief Holds the loss and its terms, each term before its gain. */
    struct FreeLossTerms
    {
        double total = 0.0;        /**< Gained terms times the batch. */
        double box = 0.0;          /**< Sum of (1 - CIoU) * weight. */
        double classes = 0.0;      /**< Cross-entropy over the score sum. */
        double distribution = 0.0; /**< DFL; 0 at reg_max 1. */
        double l1 = 0.0;           /**< L1; 0 at l1_gain 0. */
    };

    /**
     * @brief Evaluates in double what emit_free_detection_loss() emits, row by row.
     * @param levels One entry per level; each array holds its rows times its values per row.
     * @param batch Images per step, which scales the total.
     * @param target_scores_sum The `[1]` leaf's value.
     * @return The terms, or InvalidDescription for an array size that does not fit, or for a
     *         reg_max, class count, batch or score sum that is not positive.
     */
    [[nodiscard]] Result<FreeLossTerms> evaluate_free_detection_loss(
        SushiRuntime::span<const FreeLevelLossArrays> levels, std::int64_t reg_max,
        std::int64_t classes, std::int64_t batch, double target_scores_sum,
        const FreeLossGains& gains);

} // namespace SushiAI::Vision::Training
