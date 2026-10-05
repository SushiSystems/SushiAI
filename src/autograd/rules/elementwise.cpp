/****************************************************************/
/* elementwise.cpp                                              */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
