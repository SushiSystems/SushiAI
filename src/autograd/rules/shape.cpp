/**************************************************************************/
/* shape.cpp                                                              */
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
            if (dy == Graph::K_NO_VALUE) return;

            const Graph::ValueId x = tape.node().inputs[0];
            tape.contribute_input(0, tape.builder().reshape(dy, tape.graph().value(x).shape));
        }

        void layout_convert_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;

            // Permutation adjoint inverts by converting back to the operand layout.
            const Graph::ValueId x = tape.node().inputs[0];
            tape.contribute_input(0, tape.builder().layout_convert(
                dy, tape.graph().value(x).layout));
        }
    }
}
