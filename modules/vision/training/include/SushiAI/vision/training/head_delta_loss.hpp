/**************************************************************************/
/* head_delta_loss.hpp                                                    */
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
 * @file head_delta_loss.hpp
 * @brief Declares emit_head_delta_loss, the loss root whose gradient at a head is a given delta.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Emits the one-element value -scale * sum(delta * head).
     * @param builder Where the four nodes go: MUL, RESHAPE, SUM_ROWS and SCALE.
     * @param delta A leaf in the shape and element type of @p head; it takes no gradient.
     * @param scale The factor; the gradient at @p head is -scale * delta.
     * @return The id of the [1] result, a SCALE node's first output.
     * @throws Error If @p head and @p delta differ in shape or element type.
     */
    [[nodiscard]] Graph::ValueID emit_head_delta_loss(const Graph::GraphBuilder& builder,
                                                      Graph::ValueID head, Graph::ValueID delta,
                                                      double scale);

} // namespace SushiAI::Vision::Training
