/**************************************************************************/
/* elementwise.cpp                                                        */
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
 * @file elementwise.cpp
 * @brief Implements backward differentiation rules for elementwise operations.
 * @author Mustafa Garip
 */

#include <SushiAI/autograd/rules/elementwise.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        void add_backward(const GradientTape& tape)
        {
            const Graph::ValueId dc = tape.output_grad(0);
            if (dc == Graph::NO_VALUE) return;

            if (tape.input_requires_grad(0)) tape.contribute_input(0, dc);
            if (tape.input_requires_grad(1)) tape.contribute_input(1, dc);
        }

        void sub_backward(const GradientTape& tape)
        {
            const Graph::ValueId dc = tape.output_grad(0);
            if (dc == Graph::NO_VALUE) return;

            if (tape.input_requires_grad(0)) tape.contribute_input(0, dc);
            if (tape.input_requires_grad(1)) tape.contribute_input(1, tape.builder().scale(dc, -1.0));
        }

        void mul_backward(const GradientTape& tape)
        {
            const Graph::Node& node = tape.node();
            const Graph::ValueId dc = tape.output_grad(0);
            if (dc == Graph::NO_VALUE) return;

            const Graph::GraphBuilder& builder = tape.builder();
            if (tape.input_requires_grad(0))
                tape.contribute_input(0, builder.mul(dc, node.inputs.at(1)));
            if (tape.input_requires_grad(1))
                tape.contribute_input(1, builder.mul(dc, node.inputs.at(0)));
        }

        void scale_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            tape.contribute_input(0, tape.builder().scale(dy, tape.node().attributes.get(0)));
        }

        void exp_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            // d/dx exp(x) = exp(x), which the forward pass already produced.
            tape.contribute_input(0, tape.builder().mul(dy, tape.node().outputs.at(0)));
        }

        void accumulate_backward(const GradientTape& tape)
        {
            const Graph::ValueId dout = tape.output_grad(0);
            if (dout == Graph::NO_VALUE) return;

            if (tape.input_requires_grad(0)) tape.contribute_input(0, dout);
            if (tape.input_requires_grad(1)) tape.contribute_input(1, dout);
        }

    } // namespace Autograd
} // namespace SushiAI
