/**************************************************************************/
/* norm.cpp                                                               */
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

#include <SushiAI/autograd/rules/norm.hpp>

// The two normalisation gradients. BATCHNORM2D's is the one rule in this plan
// whose backward is not a rearrangement of its forward: the statistics depend
// on every element of the batch, so differentiating it produces two reductions
// that no elementwise reading of the forward would suggest.

namespace SushiAI
{
    namespace Autograd
    {
        void batchnorm2d_backward(const GradientTape& tape)
        {
            // Results 1 and 2 are saved_mean and saved_inv_std: a side channel
            // this node's own backward reads, which CV-7 keeps out of the
            // differentiated graph. Nothing differentiable consumes them, so a
            // gradient arriving on either would be silently dropped by the
            // early return below. Enforced rather than assumed, because the
            // day something does consume them the failure is a model that
            // trains slightly wrong and no test that fails.
            SA_THROW_IF(tape.output_grad(1) != Graph::NO_VALUE
                        || tape.output_grad(2) != Graph::NO_VALUE,
                        "BATCHNORM2D's saved statistics received a gradient. They are a side "
                        "channel for this node's own backward (CV-7), not a differentiable "
                        "result, and this rule has no path to propagate one.");

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // One node producing all three gradients, because all three share
            // the two reductions over the batch: splitting them would compute
            // the same sums three times, and adding it to activation_backward
            // (which is [dy, primal] -> [dx]) would break that rule's shape.
            //
            // The saved statistics come from this node's own results, so the
            // gradient can only ever be paired with the normalisation that
            // produced them.
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
            if (dy == Graph::NO_VALUE) return;

            const Graph::Node& node = tape.node();

            // dx = dy * gamma, broadcast per channel - the same shape of
            // operation as the forward, so it is the forward's own emitter
            // with beta set to zero rather than a new identity.
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
