/**************************************************************************/
/* loss.cpp                                                               */
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

#include <SushiAI/autograd/rules/loss.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        void log_softmax_backward(const GradientTape&)
        {
            // Reachable only when a LOG_SOFTMAX result is consumed by
            // something other than NLL_LOSS. The standalone rule is
            // `dx = dy - exp(L) * rowsum(dy)`, so it needs a row-wise sum of
            // the incoming gradient - and a backward rule writes IR, never
            // SushiBLAS calls, so what it is missing is an IR *node* for that
            // sum, not the kernel underneath it.
            //
            // The kernel exists now: SushiBLAS grew `sum_along` alongside the
            // `max_along` that fixed log-softmax's own stabilising shift. What
            // is still absent here is an op identity, a builder entry point, a
            // shape rule and a lowering for a row-wise sum, plus this rule
            // written against them. Named rather than left implied, because
            // the honest blocker moved and a stale message would send the next
            // reader to the wrong repository.
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

            // Fused: differentiate log-softmax and the loss together. The
            // composite gradient with respect to the logits is
            // (softmax(X) - Y) / N, which is both far cheaper than chaining
            // through the seven-op log-softmax recipe and far better
            // conditioned - the p and (1-p) factors that cancel analytically
            // would otherwise be computed and subtracted numerically.
            //
            // Valid because the targets are one-hot: the general form is
            // (p * rowsum(Y) - Y) / N, and rowsum(Y) is 1.
            tape.skip(producer);

            const Graph::ValueId logits = tape.graph().node(producer).inputs.at(0);
            if (!tape.graph().value(logits).requires_grad) return;

            const Graph::ValueId probabilities = builder.exp(log_probs);
            const Graph::ValueId difference = builder.sub(probabilities, targets);
            // The loss scale multiplies *here*, on the seeded gradient and
            // before anything narrows. Two reasons it is this value and not
            // another. It is the last point at which one multiply covers the
            // whole backward pass - every gradient below it is linear in it -
            // so one node buys the scaling of the entire graph. And the
            // cross-entropy tail is kept in fp32 by the precision policy
            // (graph/precision.hpp rule 4), so the factor is applied while
            // there is still exponent range to apply it in; a scale inserted
            // after the first cast to fp16 would overflow the very value it
            // exists to keep out of the subnormals.
            tape.contribute(logits, tape.apply_seed_factor(
                builder.scale(difference, 1.0 / batch)));
        }

    } // namespace Autograd
} // namespace SushiAI
