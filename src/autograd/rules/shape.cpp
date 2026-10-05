/****************************************************************/
/* shape.cpp                                                    */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file shape.cpp
 * @brief Implements backward differentiation rules for shape and layout operations.
 * @author Mustafa Garip
 */

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
            tape.contribute_input(0, tape.builder().reshape(dy, tape.graph().value(x).shape));
        }

        void layout_convert_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // Permutation adjoint inverts by converting back to the operand layout.
            const Graph::ValueId x = tape.node().inputs[0];
            tape.contribute_input(0, tape.builder().layout_convert(
                dy, tape.graph().value(x).layout));
        }
    }
}
