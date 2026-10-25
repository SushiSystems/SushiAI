/**************************************************************************/
/* distribution_expectation_graph.hpp                                     */
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
 * @file distribution_expectation_graph.hpp
 * @brief Declares the graph of the expected distances of a box output of reg_max bins per side.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Emits the `[B, H, W, 4]` expected distances of a `[B, H, W, 4 * reg_max]` box output.
     * @param box Channel `side * reg_max + bin`, sides left, top, right, bottom.
     * @param projection The `[reg_max, 1]` leaf fill_distribution_projection() writes, of the
     *        type of @p box; unread, and may be K_NO_VALUE, when reg_max is 1.
     * @pre The shapes and the types are as stated; anything else stops the process.
     * @return @p box itself when reg_max is 1.
     */
    [[nodiscard]] Graph::ValueID emit_distribution_expectation(const Graph::GraphBuilder& builder,
                                                               const Graph::Graph& graph,
                                                               Graph::ValueID box,
                                                               Graph::ValueID projection);

} // namespace SushiAI::Vision::Training
