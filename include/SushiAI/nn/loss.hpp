/****************************************************************/
/* loss.hpp                                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file loss.hpp
 * @brief Declares loss functions and loss calculation modules for training.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/core/common.hpp>
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
            Graph::ValueId forward(const Graph::GraphBuilder& builder,
                                   Graph::ValueId logits,
                                   Graph::ValueId one_hot) const
            {
                return builder.nll_loss(builder.log_softmax(logits), one_hot);
            }
        };

    } // namespace NN
} // namespace SushiAI
