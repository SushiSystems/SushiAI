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
        void log_softmax_backward(const GradientTape& tape)
        {
            const Graph::ValueID dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            // The id is copied out before the emission grows the graph.
            const Graph::ValueID log_probs = tape.node().outputs.at(0);

            // The forward emitter checked the shape, so a refusal here is fatal in value().
            tape.contribute_input(
                0, tape.builder().log_softmax_backward(dy, log_probs).value());
        }

        namespace
        {
            /**
             * @brief Returns @p local weighted by the gradient that reaches the loss.
             *
             * At the root the gradient is the implicit one and the seed factor stands in for
             * it. Elsewhere @p dy, one element, is spread over the shape of @p local and
             * multiplied in.
             */
            Graph::ValueID weigh_by_loss_gradient(const GradientTape& tape, Graph::ValueID local)
            {
                if (tape.output_is_root(0))
                    return tape.apply_seed_factor(local);

                const Graph::GraphBuilder& builder = tape.builder();
                const Graph::ValueID dy = tape.output_grad(0);

                // The shape and the type are copied out before the emission grows the graph.
                const Shape shape = tape.graph().value(local).shape;
                const DType dtype = tape.graph().value(local).dtype;

                const Graph::ValueID column =
                    builder.zeros_like(dy, Shape{shape.num_elements(), 1}, dtype);
                const Graph::ValueID spread =
                    builder.bias_add(column, builder.reshape(dy, Shape{1}));
                return builder.mul(builder.reshape(spread, shape), local);
            }
        } // namespace

        void nll_loss_backward(const GradientTape& tape)
        {
            const Graph::Node& node = tape.node();

            if (!tape.output_is_root(0) && tape.output_grad(0) == Graph::K_NO_VALUE) return;

            const Graph::GraphBuilder& builder = tape.builder();
            const Graph::ValueID log_probs = node.inputs.at(0);
            const Graph::ValueID targets = node.inputs.at(1);

            const Graph::Value& value = tape.graph().value(log_probs);
            const double batch = static_cast<double>(value.shape.dim(0));

            const Graph::NodeID producer = value.producer;
            const bool fused = producer != Graph::K_NO_NODE
                            && tape.graph().node(producer).op == Ops::K_LOG_SOFTMAX.id;

            if (!fused)
            {
                // loss = -sum(L * Y) / N, so dL is just the targets, scaled.
                if (tape.input_requires_grad(0))
                    tape.contribute_input(0, weigh_by_loss_gradient(
                        tape, builder.scale(targets, -1.0 / batch)));
                return;
            }

            // Fused cross-entropy analytic gradient with one-hot targets simplifies to (softmax(X) - Y) / N.
            tape.skip(producer);

            const Graph::ValueID logits = tape.graph().node(producer).inputs.at(0);
            if (!tape.graph().value(logits).requires_grad) return;

            const Graph::ValueID probabilities = builder.exp(log_probs);
            const Graph::ValueID difference = builder.sub(probabilities, targets);
            // Seed loss scale factor at the root gradient before downstream precision casts.
            tape.contribute(logits, weigh_by_loss_gradient(
                tape, builder.scale(difference, 1.0 / batch)));
        }

    } // namespace Autograd
} // namespace SushiAI
