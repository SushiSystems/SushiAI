/**************************************************************************/
/* concat.hpp                                                             */
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
 * @file concat.hpp
 * @brief Declares Concat, which joins feature maps along the channel axis, and its factory.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/node.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Joins its [H, W, C_i] inputs into one [H, W, sum(C_i)] result; holds no state. */
        class Concat final : public Node
        {
            public:
                /**
                 * @brief Builds the node over the per-sample shapes of its inputs.
                 * @param inputs One [H, W, C] shape per input, in the order they are joined.
                 * @return The node, or InvalidDescription when @p inputs is empty, an input is
                 *         not rank 3, or two inputs disagree on H or W.
                 */
                [[nodiscard]] static Result<std::unique_ptr<Concat>> create(ShapeSpan inputs);

                /**
                 * @brief Emits one CONCAT_CHANNELS over @p inputs and appends its result.
                 * @return InvalidDescription unless @p inputs holds one value per shape the
                 *         node was built over, each of that shape; @p outputs is unchanged then.
                 */
                [[nodiscard]] Status forward(const Trace& trace, ValueSpan inputs,
                                             std::vector<Graph::ValueID>& outputs) override;

                /** @brief Returns the number of shapes the node was built over. */
                [[nodiscard]] std::size_t input_count() const noexcept override;

                /** @brief Visits nothing. */
                void walk_state(std::string_view prefix, const StateVisitor& visitor) override;

                /** @brief Lists the single port, named K_SINGLE_PORT_NAME. */
                [[nodiscard]] SushiRuntime::span<const Port> ports() const noexcept override;

            private:
                /** @brief Binds the input count and the joined shape create() checked. */
                Concat(std::size_t input_count, const Shape& joined);

                /** @brief Holds how many values forward() reads. */
                std::size_t input_count_;

                /** @brief Holds the one port forward() appends to. */
                Port port_;
        };

        /**
         * @brief Builds a Concat node, which takes no key.
         * @param inputs One per-sample [H, W, C] shape per value joined, in the joining order.
         * @return The node, or InvalidDescription when a key is present or the shapes do not
         *         fit Concat::create().
         */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_concat_node(
            ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace NN
} // namespace SushiAI
