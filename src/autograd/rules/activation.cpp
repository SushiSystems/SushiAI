/**************************************************************************/
/* activation.cpp                                                        */
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

            tape.contribute_input(
                0, tape.builder().activation_backward(activation->backward, dy, primal));
        }

    } // namespace Autograd
} // namespace SushiAI
