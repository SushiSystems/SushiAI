/**************************************************************************/
/* activation.cpp                                                         */
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
 * @file activation.cpp
 * @brief Implements backward differentiation rules for activation operations.
 * @author Mustafa Garip
 */

#include <SushiAI/autograd/rules/activation.hpp>
#include <SushiAI/failure/error.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        void activation_backward(const GradientTape& tape)
        {
            const Graph::ValueID dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            const Graph::Node& node = tape.node();
            const Ops::Activation* activation = Ops::activation_of(node.op);

            SA_THROW_IF(activation == nullptr,
                        "{} is registered as an activation's backward rule but is not in "
                        "Ops::ACTIVATIONS. Add it there rather than here; that table is what "
                        "the fusion catalog also reads.",
                        Ops::name_of(node.op));

            const Graph::ValueID primal =
                activation->gradient_source == Ops::GradientSource::Output
                    ? node.outputs.at(0)
                    : node.inputs.at(0);

            // Forward attributes directly to match the operation signature specified in ops.hpp.
            tape.contribute_input(
                0, tape.builder().activation_backward(activation->backward, dy, primal,
                                                       node.attributes));
        }

    } // namespace Autograd
} // namespace SushiAI
