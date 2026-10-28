/**************************************************************************/
/* free_detection_loss.hpp                                                */
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
 * @file free_detection_loss.hpp
 * @brief Declares the loss graph of an anchor-free detector's one-to-many outputs.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds the values one level of the loss reads, each `[B, H, W, *]` of one type. */
    struct FreeLevelLossValues
    {
        Graph::ValueID boxes = Graph::K_NO_VALUE;          /**< Port, `4 * reg_max` channels. */
        Graph::ValueID scores = Graph::K_NO_VALUE;         /**< Port, one logit per class. */
        Graph::ValueID target_scores = Graph::K_NO_VALUE;  /**< Leaf, C channels. */
        Graph::ValueID box_mask = Graph::K_NO_VALUE;       /**< Leaf, 4 channels. */
        Graph::ValueID normalized_weight = Graph::K_NO_VALUE; /**< Leaf, 1 channel. */
        Graph::ValueID target_boxes = Graph::K_NO_VALUE;   /**< Leaf, 4 channels, grid units. */
        Graph::ValueID target_distances = Graph::K_NO_VALUE; /**< Leaf, 4 channels. */
        Graph::ValueID l1_scale = Graph::K_NO_VALUE;       /**< Leaf, 4; unread at l1_gain 0. */
        Graph::ValueID safe_boxes = Graph::K_NO_VALUE;     /**< Leaf, 4 channels. */
        Graph::ValueID distribution_weights = Graph::K_NO_VALUE; /**< Leaf; reg_max above 1. */
        Graph::ValueID anchor_points = Graph::K_NO_VALUE;  /**< Leaf, 2 channels, grid units. */
    };

    /** @brief Holds every value the loss reads. */
    struct FreeLossValues
    {
        std::vector<FreeLevelLossValues> levels;                  /**< One per level, in order. */
        Graph::ValueID target_scores_sum = Graph::K_NO_VALUE;     /**< Leaf `[1]`, at least 1. */
        Graph::ValueID projection = Graph::K_NO_VALUE; /**< Leaf `[reg_max, 1]`; reg_max above 1. */
    };

    /** @brief Holds the ids of the loss and of its terms, each `[1]` and before its gain. */
    struct FreeLossTermIDs
    {
        Graph::ValueID total = Graph::K_NO_VALUE;        /**< Gained terms times the batch. */
        Graph::ValueID box = Graph::K_NO_VALUE;          /**< Sum of (1 - CIoU) * weight. */
        Graph::ValueID classes = Graph::K_NO_VALUE;      /**< Cross-entropy over the score sum. */
        Graph::ValueID distribution = Graph::K_NO_VALUE; /**< DFL; K_NO_VALUE at reg_max 1. */
        Graph::ValueID l1 = Graph::K_NO_VALUE;           /**< L1; K_NO_VALUE at l1_gain 0. */
    };

    /**
     * @brief Emits the anchor-free loss over every level, a `[1]` root, or the first refusal.
     * @param facts Gives reg_max, the classes and the level count the values must match.
     * @param values The ports and the leaves FreeAssignmentStage declares, of one real type.
     * @param gains As validate_free_loss_gains() accepts them.
     * @pre The shapes, the types and the gains are as stated; anything else stops the process.
     * @see docs/agent/2026_10_25_ULTRALYTICS_DETECTORS/PLAN.md, section 4.
     */
    [[nodiscard]] Result<FreeLossTermIDs> emit_free_detection_loss(
        const Graph::GraphBuilder& builder, const Graph::Graph& graph,
        const FreeModelFacts& facts, const FreeLossValues& values, const FreeLossGains& gains) noexcept;

} // namespace SushiAI::Vision::Training
