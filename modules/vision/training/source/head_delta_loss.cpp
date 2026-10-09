/**************************************************************************/
/* head_delta_loss.cpp                                                    */
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
 * @file head_delta_loss.cpp
 * @brief Implements emit_head_delta_loss over MUL, RESHAPE, SUM_ROWS and SCALE.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/head_delta_loss.hpp>

#include <cstdint>

#include <SushiAI/core/shape.hpp>

namespace SushiAI::Vision::Training
{
    Graph::ValueId emit_head_delta_loss(const Graph::GraphBuilder& builder, Graph::ValueId head,
                                        Graph::ValueId delta, double scale)
    {
        const Graph::ValueId product = builder.mul(delta, head);

        // By value: the emissions below can reallocate the graph's value vector.
        const std::int64_t elements = builder.graph().value(product).shape.num_elements();

        const Graph::ValueId column = builder.reshape(product, Shape{elements, 1});
        const Graph::ValueId sum = builder.sum_rows(column);

        return builder.scale(sum, -scale);
    }

} // namespace SushiAI::Vision::Training
