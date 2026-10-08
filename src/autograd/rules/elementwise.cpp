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

#include <cstddef>

namespace SushiAI
{
    namespace Autograd
    {
        namespace
        {
            /**
             * @brief Sends a gradient to one of two operands of the current node, by a mask.
             * @param tape The tape positioned on the node being differentiated.
             * @param dy The gradient to route.
             * @param mask Holds 1 where operand @p where_set takes @p dy, 0 where the other does.
             * @param where_set Index of the operand served where the mask is 1.
             * @param where_clear Index of the operand served where the mask is 0.
             */
            void route_gradient(const GradientTape& tape, Graph::ValueId dy, Graph::ValueId mask,
                                std::size_t where_set, std::size_t where_clear)
            {
                const Graph::GraphBuilder& builder = tape.builder();
                const Shape shape = tape.graph().value(dy).shape;
                const DType dtype = tape.graph().value(dy).dtype;
                const Graph::ValueId zero = builder.zeros_like(dy, shape, dtype);

                if (tape.input_requires_grad(where_set))
                    tape.contribute_input(where_set, builder.select(mask, dy, zero));
                if (tape.input_requires_grad(where_clear))
                    tape.contribute_input(where_clear, builder.select(mask, zero, dy));
            }
        } // namespace

        void add_backward(const GradientTape& tape)
        {
            const Graph::ValueId dc = tape.output_grad(0);
            if (dc == Graph::K_NO_VALUE) return;

            if (tape.input_requires_grad(0)) tape.contribute_input(0, dc);
            if (tape.input_requires_grad(1)) tape.contribute_input(1, dc);
        }

        void sub_backward(const GradientTape& tape)
        {
            const Graph::ValueId dc = tape.output_grad(0);
            if (dc == Graph::K_NO_VALUE) return;

            if (tape.input_requires_grad(0)) tape.contribute_input(0, dc);
            if (tape.input_requires_grad(1)) tape.contribute_input(1, tape.builder().scale(dc, -1.0));
        }

        void mul_backward(const GradientTape& tape)
        {
            const Graph::Node& node = tape.node();
            const Graph::ValueId dc = tape.output_grad(0);
            if (dc == Graph::K_NO_VALUE) return;

            const Graph::GraphBuilder& builder = tape.builder();
            if (tape.input_requires_grad(0))
                tape.contribute_input(0, builder.mul(dc, node.inputs.at(1)));
            if (tape.input_requires_grad(1))
                tape.contribute_input(1, builder.mul(dc, node.inputs.at(0)));
        }

        void scale_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            tape.contribute_input(0, tape.builder().scale(dy, tape.node().attributes.get(0)));
        }

        void exp_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            // d/dx exp(x) = exp(x), which the forward pass already produced.
            tape.contribute_input(0, tape.builder().mul(dy, tape.node().outputs.at(0)));
        }

        void accumulate_backward(const GradientTape& tape)
        {
            const Graph::ValueId dout = tape.output_grad(0);
            if (dout == Graph::K_NO_VALUE) return;

            if (tape.input_requires_grad(0)) tape.contribute_input(0, dout);
            if (tape.input_requires_grad(1)) tape.contribute_input(1, dout);
        }

        void div_backward(const GradientTape& tape)
        {
            const Graph::ValueId dc = tape.output_grad(0);
            if (dc == Graph::K_NO_VALUE)
                return;
            if (!tape.input_requires_grad(0) && !tape.input_requires_grad(1))
                return;

            const Graph::Node& node = tape.node();
            const Graph::GraphBuilder& builder = tape.builder();
            const Graph::ValueId quotient = builder.div(dc, node.inputs.at(1));

            if (tape.input_requires_grad(0))
                tape.contribute_input(0, quotient);

            // The output c equals a / b, so -(dc / b) * c is -dc * a / b^2 without squaring b.
            if (tape.input_requires_grad(1))
                tape.contribute_input(
                    1, builder.scale(builder.mul(quotient, node.outputs.at(0)), -1.0));
        }

        void minimum_backward(const GradientTape& tape)
        {
            const Graph::ValueId dc = tape.output_grad(0);
            if (dc == Graph::K_NO_VALUE)
                return;
            if (!tape.input_requires_grad(0) && !tape.input_requires_grad(1))
                return;

            // A tie satisfies a <= b, so operand 0 takes the whole gradient there.
            const Graph::Node& node = tape.node();
            const Graph::ValueId first_wins =
                tape.builder().less_equal(node.inputs.at(0), node.inputs.at(1));
            route_gradient(tape, dc, first_wins, 0, 1);
        }

        void maximum_backward(const GradientTape& tape)
        {
            const Graph::ValueId dc = tape.output_grad(0);
            if (dc == Graph::K_NO_VALUE)
                return;
            if (!tape.input_requires_grad(0) && !tape.input_requires_grad(1))
                return;

            // A tie satisfies b <= a, so operand 0 takes the whole gradient there.
            const Graph::Node& node = tape.node();
            const Graph::ValueId first_wins =
                tape.builder().less_equal(node.inputs.at(1), node.inputs.at(0));
            route_gradient(tape, dc, first_wins, 0, 1);
        }

        void abs_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE)
                return;
            if (!tape.input_requires_grad(0))
                return;

            const Graph::GraphBuilder& builder = tape.builder();
            const Graph::ValueId x = tape.node().inputs.at(0);
            const Shape shape = tape.graph().value(x).shape;
            const DType dtype = tape.graph().value(x).dtype;
            const Graph::ValueId zero = builder.zeros_like(x, shape, dtype);

            // sign(x) is [0 <= x] - [x <= 0]; both masks are 1 at x = 0, so the sign is 0 there.
            const Graph::ValueId not_negative = builder.less_equal(zero, x);
            const Graph::ValueId not_positive = builder.less_equal(x, zero);
            const Graph::ValueId sign = builder.sub(not_negative, not_positive);

            tape.contribute_input(0, builder.mul(dy, sign));
        }

        void atan_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE)
                return;
            if (!tape.input_requires_grad(0))
                return;

            const Graph::GraphBuilder& builder = tape.builder();
            const Graph::ValueId x = tape.node().inputs.at(0);
            const Graph::ValueId squared = builder.mul(x, x);
            const Graph::ValueId denominator = builder.add(squared, builder.constant_like(x, 1.0));

            tape.contribute_input(0, builder.div(dy, denominator));
        }

        void select_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE)
                return;
            if (!tape.input_requires_grad(1) && !tape.input_requires_grad(2))
                return;

            // The condition is operand 0 and receives nothing; operands 1 and 2 share dy.
            route_gradient(tape, dy, tape.node().inputs.at(0), 1, 2);
        }

    } // namespace Autograd
} // namespace SushiAI
