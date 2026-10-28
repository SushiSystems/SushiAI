/**************************************************************************/
/* pixel_losses.cpp                                                       */
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
 * @file pixel_losses.cpp
 * @brief Implements the loss graphs that compare [R, C] scores with one class per row.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/pixel_losses.hpp>

#include <cstdint>
#include <string_view>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Names the rank of the [R, C] scores both losses read. */
        constexpr std::int32_t K_ROWS_RANK = 2;

        /**
         * @brief Returns the [R, C] shape of @p logits.
         * @pre @p logits is a rank-2 value of at least one element; anything else stops the
         *      process, naming @p loss.
         */
        [[nodiscard]] Shape require_rows(const Graph::Graph& graph, Graph::ValueID logits,
                                         std::string_view loss)
        {
            // By value: the caller's emissions can reallocate the value vector.
            const Shape shape = graph.value(logits).shape;
            SA_FATAL_IF(shape.rank() != K_ROWS_RANK || shape.num_elements() < 1,
                        "The {} reads [rows, classes] scores; got {}", loss, shape.to_string());

            return shape;
        }

        /** @brief Stops the process unless @p operand has @p shape and the type of @p logits. */
        void require_operand(const Graph::Graph& graph, Graph::ValueID logits,
                             Graph::ValueID operand, const Shape& shape, std::string_view name)
        {
            const Graph::Value& held = graph.value(operand);
            SA_FATAL_IF(held.shape != shape || held.dtype != graph.value(logits).dtype,
                        "The loss operand '{}' is {} {}; the loss reads it as {} {}", name,
                        dtype_name(held.dtype), held.shape.to_string(),
                        dtype_name(graph.value(logits).dtype), shape.to_string());
        }
    } // namespace

    Graph::Traced emit_pixel_cross_entropy(const Graph::GraphBuilder& builder,
                                           const Graph::Graph& graph, Graph::ValueID logits,
                                           Graph::ValueID indices, Graph::ValueID weights)
    {
        const Shape shape = require_rows(graph, logits, "per-pixel cross entropy");
        const std::int64_t rows = shape[0];
        const std::int64_t classes = shape[1];
        require_operand(graph, logits, weights, Shape{rows, 1}, "weights");

        // As a table of rows * classes single values, one index picks one log-probability.
        SA_TRY_ASSIGN(const Graph::ValueID table,
                      builder.reshape(builder.log_softmax(logits), Shape{rows * classes, 1}));
        SA_TRY_ASSIGN(const Graph::ValueID picked, builder.gather_rows(table, indices));

        return builder.scale(builder.sum_rows(builder.mul(picked, weights)), -1.0);
    }

    Graph::Traced emit_soft_dice(const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                                 Graph::ValueID logits, const DiceOperands& operands)
    {
        const Shape shape = require_rows(graph, logits, "soft Dice");
        const std::int64_t rows = shape[0];
        const std::int64_t classes = shape[1];
        require_operand(graph, logits, operands.kept, Shape{rows, 1}, "kept");
        require_operand(graph, logits, operands.class_pixels, Shape{classes}, "class_pixels");
        require_operand(graph, logits, operands.class_weights, Shape{classes}, "class_weights");

        SA_TRY_ASSIGN(const Graph::ValueID probabilities, builder.softmax(logits));

        // As a table of rows * classes single values, one index picks a pixel's own class.
        SA_TRY_ASSIGN(const Graph::ValueID table,
                      builder.reshape(probabilities, Shape{rows * classes, 1}));
        SA_TRY_ASSIGN(const Graph::ValueID picked,
                      builder.gather_rows(table, operands.score_indices));

        // An ignored row is zeroed before the sum, so the class its index names gains nothing.
        SA_TRY_ASSIGN(const Graph::ValueID hits,
                      builder.reshape(builder.scatter_add_rows(builder.mul(picked, operands.kept),
                                                               operands.class_indices, classes),
                                      Shape{classes}));

        // kept transposed times the probabilities sums each class over the kept rows alone.
        SA_TRY_ASSIGN(const Graph::ValueID mass,
                      builder.reshape(builder.matmul(operands.kept, probabilities, true, false),
                                      Shape{classes}));
        SA_TRY_ASSIGN(const Graph::ValueID denominators,
                      builder.add(builder.add(mass, operands.class_pixels),
                                  builder.constant_like(mass, K_DICE_EPSILON)));
        SA_TRY_ASSIGN(const Graph::ValueID ratios,
                      builder.div(builder.scale(hits, 2.0), denominators));
        SA_TRY_ASSIGN(const Graph::ValueID misses,
                      builder.sub(builder.constant_like(ratios, 1.0), ratios));
        SA_TRY_ASSIGN(const Graph::ValueID weighted, builder.mul(misses, operands.class_weights));

        return builder.sum_rows(builder.reshape(weighted, Shape{classes, 1}));
    }

} // namespace SushiAI::Vision::Training
