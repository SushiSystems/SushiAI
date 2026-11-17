/**************************************************************************/
/* softmax.hpp                                                            */
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
 * @file softmax.hpp
 * @brief Declares Softmax, the layer that turns the values along one axis into probabilities.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Normalises the values along one per-sample axis so that they sum to one. */
        struct Softmax
        {
            /**
             * @brief Holds the per-sample axis the softmax runs over, batch axis excluded.
             *
             * A negative axis counts from the end, so -1 is the last axis.
             */
            int64_t axis = -1;

            /** @brief Lists no state: the layer has no parameter and no buffer. */
            static constexpr auto fields() noexcept
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces the softmax over the last axis, or over @c axis after a permutation
             *        that moves it last and one that moves it back.
             * @param x A float32 or float64 value, batch axis included; a half-precision value
             *        is refused when the axis is not the last.
             * @return The probabilities, shaped like @p x, or @p x after trace.fail() when the
             *         axis or the value does not fit.
             */
            Graph::ValueID forward(const Trace& trace, Graph::ValueID x) noexcept;

            /**
             * @brief Returns the per-sample shape forward() produces, which is @p input.
             * @return InvalidDescription when the axis lies outside the rank of @p input.
             */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        static_assert(TraceModule<Softmax>, "Softmax must satisfy the TraceModule contract");

        /**
         * @brief Builds a Softmax node with the optional key "axis", -1 when absent.
         * @param inputs One per-sample shape of rank 1 or more.
         * @param context What to call the row in a diagnostic, e.g. "layer 3 'softmax'".
         * @return The node, or InvalidDescription when @p inputs, the key or its range does not
         *         fit.
         */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_softmax_node(
            ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

    } // namespace NN
} // namespace SushiAI
