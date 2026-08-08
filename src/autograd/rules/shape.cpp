/**************************************************************************/
/* shape.cpp                                                              */
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

#include <SushiAI/autograd/rules/shape.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        void reshape_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            const Graph::ValueId x = tape.node().inputs[0];
            tape.contribute_input(0, tape.builder().reshape(dy, tape.graph().values()[x].shape));
        }

        void layout_convert_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // The adjoint of a permutation is the inverse permutation, which
            // here is simply the conversion back to the operand's own layout.
            const Graph::ValueId x = tape.node().inputs[0];
            tape.contribute_input(0, tape.builder().layout_convert(
                dy, tape.graph().values()[x].layout));
        }
    }
}
