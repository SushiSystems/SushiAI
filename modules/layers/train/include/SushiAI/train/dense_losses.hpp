/**************************************************************************/
/* dense_losses.hpp                                                       */
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
 * @file dense_losses.hpp
 * @brief Declares the loss graphs that compare an output with a target element by element.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Emits the mean of every element of @p values.
         * @pre @p values holds at least one element of a real floating type.
         * @return A 1-element value, or the failure of @p values or of an emitter.
         */
        [[nodiscard]] Graph::Traced emit_mean_of_elements(const Graph::GraphBuilder& builder,
                                                          const Graph::Graph& graph,
                                                          const Graph::Traced& values);

        /**
         * @brief Emits the mean of max(x, 0) - x * y + log(1 + exp(-|x|)) over every element:
         *        the cross entropy of sigmoid(x) against y, computed from the logit x.
         * @param logits The values x.
         * @param targets The values y, each in [0, 1].
         * @pre Both hold one shape of at least one element and one real floating type.
         * @return A 1-element value, or the failure of an emitter.
         */
        [[nodiscard]] Graph::Traced emit_logistic_loss(const Graph::GraphBuilder& builder,
                                                       const Graph::Graph& graph,
                                                       Graph::ValueID logits,
                                                       Graph::ValueID targets);

        /**
         * @brief Emits the mean of (x - y) squared over every element.
         * @pre @p outputs and @p targets hold one shape of at least one element and one
         *      real floating type.
         * @return A 1-element value, or the failure of an emitter.
         */
        [[nodiscard]] Graph::Traced emit_squared_error(const Graph::GraphBuilder& builder,
                                                       const Graph::Graph& graph,
                                                       Graph::ValueID outputs,
                                                       Graph::ValueID targets);

    } // namespace Train
} // namespace SushiAI
