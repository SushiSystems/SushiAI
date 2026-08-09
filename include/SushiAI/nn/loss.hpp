/**************************************************************************/
/* loss.hpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file loss.hpp
 * @brief Objectives, expressed in the same IR as the model.
 *
 * Host lane: no SushiBLAS, no SYCL.
 */

#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @struct CrossEntropyLoss
         * @brief Mean cross-entropy over a batch, from raw logits.
         *
         * Emits @c LOG_SOFTMAX followed by @c NLL_LOSS as two nodes rather than
         * one. Keeping them separate leaves the log-probabilities available to
         * anything that wants them (a perplexity metric, a distillation term)
         * and costs nothing, because the differentiation pass recognises the
         * pair and emits the fused (softmax - target) / N gradient anyway -
         * see autograd/rules/loss.cpp. Fusing in the forward direction would buy the same
         * gradient and lose the intermediate.
         *
         * Targets are one-hot rows, not integer class indices. The fused
         * gradient identity depends on each row summing to one; SushiBLAS has
         * no gather, so a one-hot matrix is also the only shape the forward
         * pass can evaluate without one.
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
