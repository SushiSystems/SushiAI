/**************************************************************************/
/* add.hpp                                                                */
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
 * @file add.hpp
 * @brief Declares Add, which sums two [H, W, C] feature maps elementwise, and its factory.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/node.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @brief Sums its two [H, W, C] inputs elementwise into one [H, W, C] result.
         *
         * The node holds no state; its only data is the port it lists.
         */
        class Add final : public Node
        {
            public:
                /**
                 * @brief Builds the node over the per-sample shapes of its two inputs.
                 * @param inputs Exactly two [H, W, C] shapes, equal in every extent.
                 * @return The node, or InvalidDescription when @p inputs does not hold exactly two
                 *         shapes, an input is not rank 3, or the two inputs differ.
                 */
                [[nodiscard]] static Result<std::unique_ptr<Add>> create(ShapeSpan inputs);

                /**
                 * @brief Emits one K_ADD over the two @p inputs and appends its result.
                 * @return InvalidDescription unless @p inputs holds two values of one shape,
                 *         the shape the node was built over; @p outputs is unchanged then.
                 */
                [[nodiscard]] Status forward(const Trace& trace, ValueSpan inputs,
                                             std::vector<Graph::ValueID>& outputs) override;

                /** @brief Returns 2, the number of operands an addition takes. */
                [[nodiscard]] std::size_t input_count() const noexcept override;

                /** @brief Visits nothing. */
                void walk_state(std::string_view prefix, const StateVisitor& visitor) override;

                /** @brief Lists the single port, named K_SINGLE_PORT_NAME. */
                [[nodiscard]] SushiRuntime::span<const Port> ports() const noexcept override;

            private:
                /**
                 * @brief Binds the port to the shape create() checked both inputs share.
                 * @param shape The shape of both inputs, which is the shape of the result.
                 */
                explicit Add(const Shape& shape);

                /** @brief Holds the one port forward() appends to. */
                Port port_;
        };

        /**
         * @brief Builds an Add node, which takes no key.
         * @param inputs One per-sample [H, W, C] shape per operand, in the order they are summed.
         * @param attributes The row's own keys; any key is refused.
         * @param context What to call the row in a diagnostic, e.g. "layer 3 'add'".
         * @return The node, or InvalidDescription when a key is present or the shapes do not
         *         fit Add::create().
         */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_add_node(
            ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace NN
} // namespace SushiAI
