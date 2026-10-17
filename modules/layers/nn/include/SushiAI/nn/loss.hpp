/**************************************************************************/
/* loss.hpp                                                               */
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
 * @file loss.hpp
 * @brief Declares loss functions and loss calculation modules for training.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/failure/error.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @struct CrossEntropyLoss
         * @brief Computes mean cross-entropy over a batch from raw logits.
         */
        struct CrossEntropyLoss
        {
            /**
             * @brief Emits the loss for one batch.
             * @param builder Where the nodes go.
             * @param logits An [N, C] value of unnormalised scores.
             * @param one_hot An [N, C] one-hot target matrix.
             * @return A 1-element loss value's id.
             * @throws Error If the two operands disagree in shape or dtype.
             */
            Graph::ValueID forward(const Graph::GraphBuilder& builder,
                                   Graph::ValueID logits,
                                   Graph::ValueID one_hot) const
            {
                return builder.nll_loss(builder.log_softmax(logits), one_hot);
            }
        };

    } // namespace NN
} // namespace SushiAI
