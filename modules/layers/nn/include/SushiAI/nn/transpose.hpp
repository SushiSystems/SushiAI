/**************************************************************************/
/* transpose.hpp                                                          */
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
 * @file transpose.hpp
 * @brief Declares Transpose, the layer that permutes the per-sample axes of its input.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <vector>
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
        /** @brief Permutes the per-sample axes of its input and leaves the batch axis first. */
        struct Transpose
        {
            /**
             * @brief Holds the permutation: output axis i takes the extent of input axis
             *        @c permutation[i], over the per-sample axes, batch axis excluded.
             */
            std::vector<int64_t> permutation;

            /** @brief Lists no state: the layer has no parameter and no buffer. */
            static constexpr auto fields() noexcept
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces the permutation as one @c K_TRANSPOSE node, and as none when it is the
             *        identity.
             * @param x A half, float32 or float64 value, batch axis included.
             * @return The permuted value, or @p x after trace.fail() when the permutation or the
             *         value does not fit.
             */
            Graph::ValueID forward(const Trace& trace, Graph::ValueID x) noexcept;

            /**
             * @brief Returns the per-sample shape forward() produces.
             * @return InvalidDescription unless @c permutation is a permutation of the axes of
             *         @p input.
             */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        static_assert(TraceModule<Transpose>, "Transpose must satisfy the TraceModule contract");

        /**
         * @brief Builds a Transpose node with the required key "perm", a list of axes.
         * @param inputs One per-sample shape of rank 1 or more.
         * @param context What to call the row in a diagnostic, e.g. "layer 3 'transpose'".
         * @return The node, or InvalidDescription when @p inputs or the key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_transpose_node(
            ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

    } // namespace NN
} // namespace SushiAI
