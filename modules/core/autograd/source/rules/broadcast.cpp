/**************************************************************************/
/* broadcast.cpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file broadcast.cpp
 * @brief Implements backward differentiation rules for the broadcast arithmetic and the sum along.
 * @author Mustafa Garip
 */

#include <SushiAI/autograd/rules/broadcast.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        namespace
        {
            /**
             * @brief Holds what a broadcast rule reads from its node before the first emission.
             *
             * An emission grows the graph, which may move the node and the values a reference
             * into it points at, so the rule copies what it needs first.
             */
            struct BroadcastNode
            {
                Graph::ValueID x;            /**< @brief The tensor operand. */
                Graph::ValueID vector;       /**< @brief The vector operand. */
                Graph::ValueID result;       /**< @brief The forward result. */
                Shape vector_shape;          /**< @brief The shape of the vector operand. */
                Graph::BroadcastSide side;   /**< @brief The side the vector sits on. */
            };

            /** @brief Copies the operands, the result, the vector's shape and the side. */
            BroadcastNode read_node(const GradientTape& tape) noexcept
            {
                const Graph::Node& node = tape.node();
                return BroadcastNode{tape.input(0), tape.input(1), tape.output(0),
                                     tape.graph().value(tape.input(1)).shape,
                                     Graph::broadcast_side(node.attributes)};
            }

            /** @brief Sums @p gradient down to the vector's shape along the node's side. */
            Graph::Traced sum_to_vector(const GradientTape& tape, const BroadcastNode& node,
                                        const Graph::Traced& gradient)
            {
                return tape.builder().sum_along(gradient, node.vector_shape, node.side);
            }
        } // namespace

        void broadcast_add_backward(const GradientTape& tape) noexcept
        {
            const Graph::ValueID dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;

            const BroadcastNode node = read_node(tape);
            if (tape.input_requires_grad(0)) tape.contribute_input(0, dy);
            if (tape.input_requires_grad(1))
                tape.contribute_input(1, sum_to_vector(tape, node, dy));
        }

        void broadcast_sub_backward(const GradientTape& tape) noexcept
        {
            const Graph::ValueID dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;

            const BroadcastNode node = read_node(tape);
            if (tape.input_requires_grad(0)) tape.contribute_input(0, dy);
            if (tape.input_requires_grad(1))
                tape.contribute_input(
                    1, tape.builder().scale(sum_to_vector(tape, node, dy), -1.0));
        }

        void broadcast_mul_backward(const GradientTape& tape) noexcept
        {
            const Graph::ValueID dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;

            const BroadcastNode node = read_node(tape);
            const Graph::GraphBuilder& builder = tape.builder();
            if (tape.input_requires_grad(0))
                tape.contribute_input(0, builder.broadcast_mul(dy, node.vector, node.side));
            if (tape.input_requires_grad(1))
            {
                const Graph::Traced weighted = builder.mul(dy, node.x);
                tape.contribute_input(1, sum_to_vector(tape, node, weighted));
            }
        }

        void broadcast_div_backward(const GradientTape& tape) noexcept
        {
            const Graph::ValueID dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;

            const BroadcastNode node = read_node(tape);
            const Graph::GraphBuilder& builder = tape.builder();
            if (tape.input_requires_grad(0))
                tape.contribute_input(0, builder.broadcast_div(dy, node.vector, node.side));

            // The result y equals x / v, so -y / v is -x / v^2 without squaring v.
            if (tape.input_requires_grad(1))
            {
                const Graph::Traced quotient =
                    builder.broadcast_div(node.result, node.vector, node.side);
                const Graph::Traced weighted = builder.mul(dy, quotient);
                tape.contribute_input(
                    1, builder.scale(sum_to_vector(tape, node, weighted), -1.0));
            }
        }

        void sum_along_backward(const GradientTape& tape) noexcept
        {
            const Graph::ValueID ds = tape.output_grad(0);
            if (ds == Graph::K_NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            // The shape, the type and the side are copied out before the first emission.
            const Graph::Value& operand = tape.graph().value(tape.input(0));
            const Shape shape = operand.shape;
            const DType dtype = operand.dtype;
            const Graph::BroadcastSide side = Graph::broadcast_side(tape.node().attributes);

            // The adjoint of a sum is a broadcast: dy added to zeros of the operand's shape.
            const Graph::GraphBuilder& builder = tape.builder();
            const Graph::Traced zeros = builder.zeros_like(ds, shape, dtype);
            tape.contribute_input(0, builder.broadcast_add(zeros, ds, side));
        }
    } // namespace Autograd
} // namespace SushiAI
