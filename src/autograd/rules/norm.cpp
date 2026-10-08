/**************************************************************************/
/* norm.cpp                                                               */
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
 * @file norm.cpp
 * @brief Implements backward differentiation rules for normalization operations.
 * @author Mustafa Garip
 */

#include <SushiAI/autograd/rules/norm.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        void batchnorm2d_backward(const GradientTape& tape)
        {
            // The saved mean, the inverse deviation and the batch variance carry no gradient.
            SA_THROW_IF(tape.output_grad(1) != Graph::K_NO_VALUE
                        || tape.output_grad(2) != Graph::K_NO_VALUE
                        || tape.output_grad(3) != Graph::K_NO_VALUE,
                        "BATCHNORM2D's saved statistics received a gradient. They are a side "
                        "channel for this node's own backward (CV-7), not a differentiable "
                        "result, and this rule has no path to propagate one.");

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;

            // Batch normalization gradients share batch reductions across input, scale, and bias.
            const Graph::Node& node = tape.node();
            const Graph::BatchNormGrads g = tape.builder().batchnorm2d_backward(
                dy, node.inputs[0], node.inputs[1], node.outputs[1], node.outputs[2]);

            if (tape.input_requires_grad(0)) tape.contribute_input(0, g.dx);
            if (tape.input_requires_grad(1)) tape.contribute_input(1, g.dgamma);
            if (tape.input_requires_grad(2)) tape.contribute_input(2, g.dbeta);
        }

        void channel_affine_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;

            const Graph::Node& node = tape.node();

            // Affine input gradient broadcasts channel scaling with zero bias.
            if (tape.input_requires_grad(0))
                tape.contribute_input(0, tape.builder().channel_affine(
                    dy, node.inputs[1],
                    tape.builder().zeros_like(dy, tape.graph().value(node.inputs[2]).shape,
                                              tape.graph().value(dy).dtype)));

            // dgamma and dbeta are per-channel sums over N, H and W.
            if (tape.input_requires_grad(1))
                tape.contribute_input(1, tape.builder().sum_channels(
                    tape.builder().mul(dy, node.inputs[0])));
            if (tape.input_requires_grad(2))
                tape.contribute_input(2, tape.builder().sum_channels(dy));
        }
    }
}
