/****************************************************************/
/* activation.cpp                                               */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file activation.cpp
 * @brief Implements backward differentiation rules for activation operations.
 * @author Mustafa Garip
 */

#include <SushiAI/autograd/rules/activation.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        void activation_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            const Graph::Node& node = tape.node();
            const Ops::Activation* activation = Ops::activation_of(node.op);

            SA_THROW_IF(activation == nullptr,
                        "{} is registered as an activation's backward rule but is not in "
                        "Ops::ACTIVATIONS. Add it there rather than here; that table is what "
                        "the fusion catalog also reads.",
                        Ops::name_of(node.op));

            const Graph::ValueId primal =
                activation->gradient_source == Ops::GradientSource::OUTPUT
                    ? node.outputs.at(0)
                    : node.inputs.at(0);

            // Forward attributes directly to match the operation signature specified in ops.hpp.
            tape.contribute_input(
                0, tape.builder().activation_backward(activation->backward, dy, primal,
                                                       node.attributes));
        }

    } // namespace Autograd
} // namespace SushiAI
