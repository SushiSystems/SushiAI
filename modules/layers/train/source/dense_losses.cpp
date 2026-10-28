/**************************************************************************/
/* dense_losses.cpp                                                       */
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
 * @file dense_losses.cpp
 * @brief Implements the loss graphs that compare an output with a target element by element.
 * @author Mustafa Garip
 */

#include <SushiAI/train/dense_losses.hpp>

#include <cstdint>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Stops the process unless @p a and @p b hold one shape and one type. */
            void require_same_values(const Graph::Graph& graph, Graph::ValueID a,
                                     Graph::ValueID b)
            {
                const Graph::Value& left = graph.value(a);
                const Graph::Value& right = graph.value(b);
                SA_FATAL_IF(left.shape != right.shape || left.dtype != right.dtype,
                            "A dense loss compares {} {} with {} {}; both must hold one shape "
                            "and one element type",
                            dtype_name(left.dtype), left.shape.to_string(),
                            dtype_name(right.dtype), right.shape.to_string());
            }
        } // namespace

        Graph::Traced emit_mean_of_elements(const Graph::GraphBuilder& builder,
                                            const Graph::Graph& graph,
                                            const Graph::Traced& values) noexcept
        {
            SA_TRY_ASSIGN(const Graph::ValueID held, values);

            // By value: every emission below can reallocate the value vector.
            const std::int64_t elements = graph.value(held).shape.num_elements();
            SA_FATAL_IF(elements < 1, "The mean of a value that holds no element was asked for");

            // As [elements, 1], K_SUM_ROWS adds every element into one.
            const Graph::Traced column = builder.reshape(held, Shape{elements, 1});

            return builder.scale(builder.sum_rows(column), 1.0 / static_cast<double>(elements));
        }

        Graph::Traced emit_logistic_loss(const Graph::GraphBuilder& builder,
                                         const Graph::Graph& graph, Graph::ValueID logits,
                                         Graph::ValueID targets) noexcept
        {
            require_same_values(graph, logits, targets);

            const Graph::Traced positive_part = builder.relu(logits);
            const Graph::Traced product = builder.mul(logits, targets);

            // exp reads a value at or below zero, so it cannot overflow for any logit.
            const Graph::Traced tail = builder.softplus(builder.scale(builder.abs(logits), -1.0));
            const Graph::Traced each = builder.add(builder.sub(positive_part, product), tail);

            return emit_mean_of_elements(builder, graph, each);
        }

        Graph::Traced emit_squared_error(const Graph::GraphBuilder& builder,
                                         const Graph::Graph& graph, Graph::ValueID outputs,
                                         Graph::ValueID targets) noexcept
        {
            require_same_values(graph, outputs, targets);

            const Graph::Traced difference = builder.sub(outputs, targets);

            return emit_mean_of_elements(builder, graph, builder.mul(difference, difference));
        }

    } // namespace Train
} // namespace SushiAI
