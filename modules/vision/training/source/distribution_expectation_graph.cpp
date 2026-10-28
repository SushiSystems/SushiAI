/**************************************************************************/
/* distribution_expectation_graph.cpp                                     */
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
 * @file distribution_expectation_graph.cpp
 * @brief Implements the graph of the expected distances of a box output of reg_max bins per side.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/distribution_expectation_graph.hpp>

#include <cstdint>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Names the rank of an NHWC box output. */
        constexpr std::int32_t K_BOX_RANK = 4;

        /** @brief Names the sides of a box: left, top, right, bottom. */
        constexpr std::int64_t K_SIDES = 4;
    } // namespace

    Graph::Traced emit_distribution_expectation(const Graph::GraphBuilder& builder,
                                                const Graph::Graph& graph, Graph::ValueID box,
                                                Graph::ValueID projection) noexcept
    {
        // By value: the emissions below can reallocate the value vector.
        const Shape shape = graph.value(box).shape;
        SA_FATAL_IF(shape.rank() != K_BOX_RANK || shape[3] < K_SIDES || shape[3] % K_SIDES != 0,
                    "The distribution expectation reads [B, H, W, 4 * reg_max]; got {}",
                    shape.to_string());
        const std::int64_t bins = shape[3] / K_SIDES;
        if (bins == 1)
            return box;

        const Graph::Value& held = graph.value(projection);
        SA_FATAL_IF(held.shape != (Shape{bins, 1}) || held.dtype != graph.value(box).dtype,
                    "The projection is {} {}; the expectation reads it as {} [{}, 1]",
                    dtype_name(held.dtype), held.shape.to_string(),
                    dtype_name(graph.value(box).dtype), bins);

        // Channels side * bins + bin are row-major, so each row below is one side's bins.
        const std::int64_t rows = shape[0] * shape[1] * shape[2] * K_SIDES;
        SA_TRY_ASSIGN(const Graph::ValueID probabilities,
                      builder.softmax(builder.reshape(box, Shape{rows, bins})));
        SA_TRY_ASSIGN(const Graph::ValueID expected, builder.matmul(probabilities, projection));

        return builder.reshape(expected, Shape{shape[0], shape[1], shape[2], K_SIDES});
    }

} // namespace SushiAI::Vision::Training
