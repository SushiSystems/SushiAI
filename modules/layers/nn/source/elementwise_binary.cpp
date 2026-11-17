/**************************************************************************/
/* elementwise_binary.cpp                                                 */
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
 * @file elementwise_binary.cpp
 * @brief Implements ElementwiseBinary: the shape rule at construction, the trace and the factories.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/elementwise_binary.hpp>

#include <string>
#include <utility>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Holds the number of operands a binary node takes. */
            constexpr std::size_t K_OPERAND_COUNT = 2;

            /** @brief Returns whether @p prefix is shorter than @p whole and opens it. */
            bool is_leading_prefix(const Shape& prefix, const Shape& whole) noexcept
            {
                if (prefix.rank() >= whole.rank())
                    return false;

                for (int32_t axis = 0; axis < prefix.rank(); ++axis)
                {
                    if (prefix[axis] != whole[axis])
                        return false;
                }

                return true;
            }

            /** @brief Returns the failure of a request the node was built to never make. */
            Graph::Traced unfit(const char* message)
            {
                return Status::from_message(StatusCode::InvalidDescription, [message]()
                {
                    return std::string(message);
                });
            }

            /** @brief Emits the operation over two operands of one shape. */
            Graph::Traced emit_equal(const Graph::GraphBuilder& builder, BinaryOperation operation,
                                     Graph::ValueID first, Graph::ValueID second)
            {
                switch (operation)
                {
                    case BinaryOperation::Mul:
                        return builder.mul(first, second);
                    case BinaryOperation::Maximum:
                        return builder.maximum(first, second);
                    case BinaryOperation::Sub:
                        return builder.sub(first, second);
                    case BinaryOperation::Div:
                        return builder.div(first, second);
                }

                return unfit("the binary operation is unknown");
            }

            /** @brief Emits the operation over @p whole and the leading-prefix @p vector. */
            Graph::Traced emit_broadcast(const Graph::GraphBuilder& builder,
                                         BinaryOperation operation, Graph::ValueID whole,
                                         Graph::ValueID vector)
            {
                constexpr Graph::BroadcastSide K_SIDE = Graph::BroadcastSide::Leading;
                switch (operation)
                {
                    case BinaryOperation::Mul:
                        return builder.broadcast_mul(whole, vector, K_SIDE);
                    case BinaryOperation::Sub:
                        return builder.broadcast_sub(whole, vector, K_SIDE);
                    case BinaryOperation::Div:
                        return builder.broadcast_div(whole, vector, K_SIDE);
                    case BinaryOperation::Maximum:
                        break;
                }

                return unfit("the graph has no broadcast form of this operation");
            }
        } // namespace

        std::string_view binary_row_name(BinaryOperation operation) noexcept
        {
            switch (operation)
            {
                case BinaryOperation::Mul:
                    return "mul";
                case BinaryOperation::Maximum:
                    return "maximum";
                case BinaryOperation::Sub:
                    return "sub";
                case BinaryOperation::Div:
                    return "div";
            }

            return "binary";
        }

        Result<std::unique_ptr<ElementwiseBinary>> ElementwiseBinary::create(
            BinaryOperation operation, ShapeSpan inputs) noexcept
        {
            const std::string_view name = binary_row_name(operation);
            SA_FAIL_IF(inputs.size() != K_OPERAND_COUNT, StatusCode::InvalidDescription,
                       "{} takes {} inputs and was given {}", name, K_OPERAND_COUNT,
                       inputs.size());
            for (std::size_t index = 0; index < inputs.size(); ++index)
            {
                SA_FAIL_IF(inputs[index].rank() < 1, StatusCode::InvalidDescription,
                           "{} input {} has the per-sample shape {}; it needs rank 1 or more",
                           name, index, inputs[index].to_string());
            }

            const Shape& first = inputs[0];
            const Shape& second = inputs[1];
            if (first == second)
                return std::unique_ptr<ElementwiseBinary>(
                    new ElementwiseBinary(operation, Arrangement::Equal, first));

            const bool commutes = operation == BinaryOperation::Mul;
            const bool broadcasts = operation != BinaryOperation::Maximum;
            if (broadcasts && is_leading_prefix(second, first))
                return std::unique_ptr<ElementwiseBinary>(
                    new ElementwiseBinary(operation, Arrangement::BroadcastSecond, first));
            if (commutes && is_leading_prefix(first, second))
                return std::unique_ptr<ElementwiseBinary>(
                    new ElementwiseBinary(operation, Arrangement::BroadcastFirst, second));

            SA_FAIL_IF(true, StatusCode::InvalidDescription,
                       "{} inputs are {} and {}; they must share one shape or the second must be a "
                       "leading prefix of the first (a product also takes the prefix first, and a "
                       "maximum takes equal shapes only)",
                       name, first.to_string(), second.to_string());
        }

        ElementwiseBinary::ElementwiseBinary(BinaryOperation operation, Arrangement arrangement,
                                             const Shape& shape)
            : operation_(operation), arrangement_(arrangement),
              port_{std::string(K_SINGLE_PORT_NAME), shape}
        {

        }

        Status ElementwiseBinary::forward(const Trace& trace, ValueSpan inputs,
                                          std::vector<Graph::ValueID>& outputs)
        {
            SA_TRY(Detail::require_input_count(inputs, K_OPERAND_COUNT));

            const Graph::GraphBuilder& builder = trace.builder();
            Graph::Traced emitted;
            switch (arrangement_)
            {
                case Arrangement::Equal:
                    emitted = emit_equal(builder, operation_, inputs[0], inputs[1]);
                    break;
                case Arrangement::BroadcastSecond:
                    emitted = emit_broadcast(builder, operation_, inputs[0], inputs[1]);
                    break;
                case Arrangement::BroadcastFirst:
                    emitted = emit_broadcast(builder, operation_, inputs[1], inputs[0]);
                    break;
            }

            SA_TRY_ASSIGN(const Graph::ValueID result, emitted);
            outputs.push_back(result);
            return Status();
        }

        std::size_t ElementwiseBinary::input_count() const noexcept
        {
            return K_OPERAND_COUNT;
        }

        void ElementwiseBinary::walk_state(std::string_view, const StateVisitor&)
        {

        }

        SushiRuntime::span<const Port> ElementwiseBinary::ports() const noexcept
        {
            return SushiRuntime::span<const Port>(&port_, 1);
        }

        Result<std::unique_ptr<Node>> make_binary_node(BinaryOperation operation, ShapeSpan inputs,
                                                       const Config::Value& attributes,
                                                       std::string_view context) noexcept
        {
            SA_FAIL_IF(!attributes.keys().empty(), StatusCode::InvalidDescription,
                       "{} has the unknown key '{}'; a {} row takes no key", context,
                       attributes.keys().front(), binary_row_name(operation));

            Result<std::unique_ptr<ElementwiseBinary>> made =
                ElementwiseBinary::create(operation, inputs);
            SA_TRY(SA_ANNOTATE(made.status(), "{}", context));

            return std::unique_ptr<Node>(std::move(made).value());
        }

        Result<std::unique_ptr<Node>> make_mul_node(ShapeSpan inputs,
                                                    const Config::Value& attributes,
                                                    std::string_view context) noexcept
        {
            return make_binary_node(BinaryOperation::Mul, inputs, attributes, context);
        }

        Result<std::unique_ptr<Node>> make_maximum_node(ShapeSpan inputs,
                                                        const Config::Value& attributes,
                                                        std::string_view context) noexcept
        {
            return make_binary_node(BinaryOperation::Maximum, inputs, attributes, context);
        }

        Result<std::unique_ptr<Node>> make_sub_node(ShapeSpan inputs,
                                                    const Config::Value& attributes,
                                                    std::string_view context) noexcept
        {
            return make_binary_node(BinaryOperation::Sub, inputs, attributes, context);
        }

        Result<std::unique_ptr<Node>> make_div_node(ShapeSpan inputs,
                                                    const Config::Value& attributes,
                                                    std::string_view context) noexcept
        {
            return make_binary_node(BinaryOperation::Div, inputs, attributes, context);
        }

    } // namespace NN
} // namespace SushiAI
