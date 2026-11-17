/**************************************************************************/
/* elementwise_binary.hpp                                                 */
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
 * @file elementwise_binary.hpp
 * @brief Declares ElementwiseBinary, the node of the mul, maximum, sub and div rows, and its factories.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Names the operation a binary node applies to its two inputs. */
        enum class BinaryOperation : std::uint8_t
        {
            Mul,     /**< The product, a * b. */
            Maximum, /**< The larger of the two, per element. */
            Sub,     /**< The difference, a - b. */
            Div,     /**< The quotient, a / b. */
        };

        /** @brief Names the row type that spells @p operation in a model file. */
        [[nodiscard]] std::string_view binary_row_name(BinaryOperation operation) noexcept;

        /**
         * @brief Applies one BinaryOperation to its two inputs and holds no state.
         *
         * The inputs share one per-sample shape, or the second one is a leading prefix of the
         * first and is broadcast along the axes it lacks. A product also takes the prefix input
         * first. A maximum takes equal shapes only, since the graph has no broadcast maximum.
         */
        class ElementwiseBinary final : public Node
        {
            public:
                /**
                 * @brief Builds the node over the per-sample shapes of its two inputs.
                 * @param inputs Exactly two shapes of rank 1 or more that are equal, or of which
                 *        the one that is a leading prefix of the other is broadcast.
                 * @return The node, or InvalidDescription when @p inputs does not hold exactly
                 *         two shapes, a rank is 0, or the shapes fit neither rule above.
                 */
                [[nodiscard]] static Result<std::unique_ptr<ElementwiseBinary>> create(
                    BinaryOperation operation, ShapeSpan inputs) noexcept;

                /**
                 * @brief Emits the node of the operation over the two @p inputs and appends it.
                 * @return InvalidDescription unless @p inputs holds two values of the shapes the
                 *         node was built over; @p outputs is unchanged then.
                 */
                [[nodiscard]] Status forward(const Trace& trace, ValueSpan inputs,
                                             std::vector<Graph::ValueID>& outputs) override;

                /** @brief Returns 2, the number of operands. */
                [[nodiscard]] std::size_t input_count() const noexcept override;

                /** @brief Visits nothing. */
                void walk_state(std::string_view prefix, const StateVisitor& visitor) override;

                /** @brief Lists the single port, named K_SINGLE_PORT_NAME. */
                [[nodiscard]] SushiRuntime::span<const Port> ports() const noexcept override;

            private:
                /** @brief Names which input forward() broadcasts, if any. */
                enum class Arrangement : std::uint8_t
                {
                    Equal,           /**< Both inputs have one shape. */
                    BroadcastSecond, /**< The second input is the prefix; the first is whole. */
                    BroadcastFirst,  /**< The first input is the prefix; the second is whole. */
                };

                /** @brief Binds the operation, the arrangement and the shape of the result. */
                ElementwiseBinary(BinaryOperation operation, Arrangement arrangement,
                                  const Shape& shape);

                /** @brief Holds the operation forward() emits. */
                BinaryOperation operation_;

                /** @brief Holds which input forward() broadcasts. */
                Arrangement arrangement_;

                /** @brief Holds the one port forward() appends to. */
                Port port_;
        };

        /**
         * @brief Builds an ElementwiseBinary node of @p operation, which takes no key.
         * @param inputs One per-sample shape per operand.
         * @param attributes The row's own keys; any key is refused.
         * @param context What to call the row in a diagnostic, e.g. "layer 3 'mul'".
         * @return The node, or InvalidDescription when a key is present or the shapes do not fit
         *         ElementwiseBinary::create().
         */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_binary_node(
            BinaryOperation operation, ShapeSpan inputs, const Config::Value& attributes,
            std::string_view context) noexcept;

        /** @brief Builds the node of a "mul" row; see make_binary_node(). */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_mul_node(
            ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

        /** @brief Builds the node of a "maximum" row; see make_binary_node(). */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_maximum_node(
            ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

        /** @brief Builds the node of a "sub" row; see make_binary_node(). */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_sub_node(
            ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

        /** @brief Builds the node of a "div" row; see make_binary_node(). */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_div_node(
            ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

    } // namespace NN
} // namespace SushiAI
