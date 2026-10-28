/**************************************************************************/
/* distribution_focal_loss.cpp                                            */
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
 * @file distribution_focal_loss.cpp
 * @brief Implements the distribution focal loss graph of a box output of reg_max bins per side.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/distribution_focal_loss.hpp>

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

        /** @brief Names the fewest bins a side needs for two neighbouring target bins. */
        constexpr std::int64_t K_MIN_BINS = 2;
    } // namespace

    Graph::Traced emit_distribution_focal_loss(const Graph::GraphBuilder& builder,
                                               const Graph::Graph& graph, Graph::ValueID box,
                                               Graph::ValueID weights)
    {
        // By value: the emissions below can reallocate the value vector.
        const Shape shape = graph.value(box).shape;
        SA_FATAL_IF(shape.rank() != K_BOX_RANK || shape[3] < K_SIDES * K_MIN_BINS ||
                        shape[3] % K_SIDES != 0,
                    "The distribution focal loss reads [B, H, W, 4 * reg_max], reg_max of at "
                    "least 2; got {}",
                    shape.to_string());
        const Graph::Value& held = graph.value(weights);
        SA_FATAL_IF(held.shape != shape || held.dtype != graph.value(box).dtype,
                    "The bin weights are {} {}; the loss reads them as {} {}",
                    dtype_name(held.dtype), held.shape.to_string(),
                    dtype_name(graph.value(box).dtype), shape.to_string());

        // Channels side * bins + bin are row-major, so each row below is one side's bins.
        const std::int64_t bins = shape[3] / K_SIDES;
        const std::int64_t rows = shape[0] * shape[1] * shape[2] * K_SIDES;
        SA_TRY_ASSIGN(const Graph::ValueID log_probabilities,
                      builder.log_softmax(builder.reshape(box, Shape{rows, bins})));
        SA_TRY_ASSIGN(const Graph::ValueID weighted,
                      builder.mul(log_probabilities, builder.reshape(weights, Shape{rows, bins})));

        return builder.scale(builder.sum_rows(builder.reshape(weighted, Shape{rows * bins, 1})),
                             -1.0);
    }

} // namespace SushiAI::Vision::Training
