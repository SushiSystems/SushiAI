/**************************************************************************/
/* loss.cpp                                                               */
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
 * @file loss.cpp
 * @brief Implements backward differentiation rules for loss functions.
 * @author Mustafa Garip
 */

#include <SushiAI/autograd/rules/loss.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        void log_softmax_backward(const GradientTape&)
        {
            // Standalone log-softmax differentiation requires a row-reduction IR node.
            SA_THROW_IF(true,
                        "LOG_SOFTMAX can only be differentiated as part of the fused "
                        "LOG_SOFTMAX + NLL_LOSS pair in this release: the standalone rule "
                        "needs a row-wise reduction of the incoming gradient, and SushiAI has "
                        "no IR node for one yet (SushiBLAS::ReductionOps::sum_along is the "
                        "kernel it would lower to). Feed its result to NLL_LOSS (see "
                        "NN::CrossEntropyLoss), or add a row-sum operation first.");
        }

        void nll_loss_backward(const GradientTape& tape)
        {
            const Graph::Node& node = tape.node();

            SA_THROW_IF(!tape.output_is_root(0),
                        "NLL_LOSS is differentiable only as the value passed to "
                        "differentiate(); its gradient is seeded as the implicit scalar one, "
                        "so it cannot sit inside a larger expression yet.");

            const Graph::GraphBuilder& builder = tape.builder();
            const Graph::ValueId log_probs = node.inputs.at(0);
            const Graph::ValueId targets = node.inputs.at(1);

            const Graph::Value& value = tape.graph().value(log_probs);
            const double batch = static_cast<double>(value.shape.dim(0));

            const Graph::NodeId producer = value.producer;
            const bool fused = producer != Graph::NO_NODE
                            && tape.graph().node(producer).op == Ops::LOG_SOFTMAX.id;

            if (!fused)
            {
                // loss = -sum(L * Y) / N, so dL is just the targets, scaled.
                if (tape.input_requires_grad(0))
                    tape.contribute_input(0, tape.apply_seed_factor(
                        builder.scale(targets, -1.0 / batch)));
                return;
            }

            // Fused cross-entropy analytic gradient with one-hot targets simplifies to (softmax(X) - Y) / N.
            tape.skip(producer);

            const Graph::ValueId logits = tape.graph().node(producer).inputs.at(0);
            if (!tape.graph().value(logits).requires_grad) return;

            const Graph::ValueId probabilities = builder.exp(log_probs);
            const Graph::ValueId difference = builder.sub(probabilities, targets);
            // Seed loss scale factor at the root gradient before downstream precision casts.
            tape.contribute(logits, tape.apply_seed_factor(
                builder.scale(difference, 1.0 / batch)));
        }

    } // namespace Autograd
} // namespace SushiAI
