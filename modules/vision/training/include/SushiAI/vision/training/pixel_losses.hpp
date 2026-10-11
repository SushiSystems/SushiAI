/**************************************************************************/
/* pixel_losses.hpp                                                       */
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
 * @file pixel_losses.hpp
 * @brief Declares the loss graphs that compare [R, C] scores with one class per row.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds the term added to a Dice denominator, which keeps an absent class finite. */
    inline constexpr double K_DICE_EPSILON = 1e-6;

    /**
     * @brief Emits minus the sum over rows of weight[r] * log_softmax(logits)[r, label[r]].
     * @param logits [R, C] scores of a real floating type.
     * @param indices The INT32 [R] leaf fill_pixel_targets() writes.
     * @param weights The [R, 1] leaf it writes, of the type of @p logits.
     * @pre The shapes and the types are as stated; anything else stops the process.
     * @return The id of a 1-element value.
     */
    [[nodiscard]] Graph::ValueID emit_pixel_cross_entropy(const Graph::GraphBuilder& builder,
                                                          const Graph::Graph& graph,
                                                          Graph::ValueID logits,
                                                          Graph::ValueID indices,
                                                          Graph::ValueID weights);

    /** @brief Holds the five leaves fill_dice_targets() writes, as values of one graph. */
    struct DiceOperands
    {
        Graph::ValueID score_indices = Graph::K_NO_VALUE; /**< INT32 [R]. */
        Graph::ValueID class_indices = Graph::K_NO_VALUE; /**< INT32 [R]. */
        Graph::ValueID kept = Graph::K_NO_VALUE;          /**< [R, 1]. */
        Graph::ValueID class_pixels = Graph::K_NO_VALUE;  /**< [C]. */
        Graph::ValueID class_weights = Graph::K_NO_VALUE; /**< [C]. */
    };

    /**
     * @brief Emits the sum over classes of weight[c] * (1 - 2 I[c] / (P[c] + Y[c] + eps)),
     *        the terms README.md defines over softmax(logits).
     * @param logits [R, C] scores of a real floating type, which the real operands share.
     * @pre The shapes and the types are as stated, and R * C fits INT32; anything else stops
     *      the process.
     * @return The id of a 1-element value.
     */
    [[nodiscard]] Graph::ValueID emit_soft_dice(const Graph::GraphBuilder& builder,
                                                const Graph::Graph& graph, Graph::ValueID logits,
                                                const DiceOperands& operands);

} // namespace SushiAI::Vision::Training
