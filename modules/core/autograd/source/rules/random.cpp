/**************************************************************************/
/* random.cpp                                                             */
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
 * @file random.cpp
 * @brief Implements backward differentiation rules for the operations that draw at each replay.
 * @author Mustafa Garip
 */

#include <SushiAI/autograd/rules/random.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/graph/builder.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        void dropout_backward(const GradientTape& tape)
        {
            SA_FATAL_IF(tape.output_grad(1) != Graph::K_NO_VALUE,
                        "DROPOUT's saved mask received a gradient; it is read by this node's "
                        "own backward and by nothing else");

            const Graph::ValueID dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            // The id is copied out before the emission grows the graph.
            const Graph::ValueID mask = tape.output(1);

            // y = x * mask and the mask is constant in x, so dx = dy * mask of the same run.
            tape.contribute_input(0, tape.builder().mul(dy, mask));
        }

    } // namespace Autograd
} // namespace SushiAI
