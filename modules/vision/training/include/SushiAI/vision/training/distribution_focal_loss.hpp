/**************************************************************************/
/* distribution_focal_loss.hpp                                            */
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
 * @file distribution_focal_loss.hpp
 * @brief Declares the distribution focal loss graph of a box output of reg_max bins per side.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Emits minus the sum of log_softmax(bins of each side) times @p weights.
     * @param box `[B, H, W, 4 * reg_max]`, channel `side * reg_max + bin`; reg_max, its channels
     *        over 4, is at least 2.
     * @param weights The leaf fill_distribution_weights() writes, of the shape and type of @p box.
     * @pre The shapes and the types are as stated; anything else stops the process.
     * @return The id of a 1-element value; the failure of an emission the builder refuses.
     */
    [[nodiscard]] Graph::Traced emit_distribution_focal_loss(const Graph::GraphBuilder& builder,
                                                             const Graph::Graph& graph,
                                                             Graph::ValueID box,
                                                             Graph::ValueID weights);

} // namespace SushiAI::Vision::Training
