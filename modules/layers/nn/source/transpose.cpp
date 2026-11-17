/**************************************************************************/
/* transpose.cpp                                                          */
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
 * @file transpose.cpp
 * @brief Implements Transpose's trace form, its shape rule and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/transpose.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/permuted_copy.hpp>
#include <SushiAI/nn/node_attributes.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Names the key that holds the permutation of a "transpose" row. */
            constexpr std::string_view K_PERM_KEY = "perm";

            /** @brief Lists every key a "transpose" row accepts. */
            constexpr std::string_view K_TRANSPOSE_KEYS[] = {K_PERM_KEY};

            /** @brief Returns InvalidDescription unless @p layer permutes @p rank axes. */
            [[nodiscard]] Status check_permutation(const Transpose& layer, int64_t rank)
            {
                SA_FAIL_IF(static_cast<int64_t>(layer.permutation.size()) != rank ||
                               !Graph::is_axis_permutation(SushiRuntime::span<const int64_t>(
                                   layer.permutation.data(), layer.permutation.size())),
                           StatusCode::InvalidDescription,
                           "Transpose lists {} axes, which is no permutation of the {} per-sample "
                           "axes of its input",
                           layer.permutation.size(), rank);

                return Status();
            }

            /** @brief Traces the permutation of @p x, or returns why it does not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_transpose(const Transpose& layer,
                                                                 const Trace& trace,
                                                                 Graph::ValueID x)
            {
                const Shape shape = trace.builder().graph().value(x).shape;
                SA_TRY(check_permutation(layer, shape.rank() - 1));

                // The identity moves nothing, so it emits no node.
                bool is_identity = true;
                for (std::size_t axis = 0; axis < layer.permutation.size(); ++axis)
                {
                    if (layer.permutation[axis] != static_cast<int64_t>(axis))
                        is_identity = false;
                }

                if (is_identity)
                    return x;

                SA_TRY_ASSIGN(const Graph::ValueID moved,
                              trace.builder().transpose(
                                  x, SushiRuntime::span<const int64_t>(layer.permutation.data(),
                                                                       layer.permutation.size())));
                return moved;
            }
        } // namespace

        Graph::ValueID Transpose::forward(const Trace& trace, Graph::ValueID x) noexcept
        {
            return trace.value_or_fail(trace_transpose(*this, trace, x), x);
        }

        Result<Shape> Transpose::output_shape(const Shape& input) const noexcept
        {
            SA_TRY(check_permutation(*this, input.rank()));

            std::vector<int64_t> extents;
            for (const int64_t axis : permutation)
                extents.push_back(input[static_cast<int32_t>(axis)]);

            return Shape::make(SushiRuntime::span<const int64_t>(extents.data(), extents.size()));
        }

        Result<std::unique_ptr<Node>> make_transpose_node(ShapeSpan inputs,
                                                          const Config::Value& attributes,
                                                          std::string_view context) noexcept
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_TRANSPOSE_KEYS),
                                               context));
            SA_FAIL_IF(inputs.size() != 1, StatusCode::InvalidDescription,
                       "{} reads one input, but it was given {}", context, inputs.size());
            SA_FAIL_IF(inputs[0].rank() < 1, StatusCode::InvalidDescription,
                       "{} was given a per-sample input of {}, but it reads an input of rank 1 or "
                       "more",
                       context, inputs[0].to_string());

            const std::string named = std::string(context) + " '" + std::string(K_PERM_KEY) + "'";
            SA_TRY_ASSIGN(const Config::Value& stated, attributes.at(K_PERM_KEY, context));
            SA_TRY_ASSIGN(const std::vector<Config::Value>& entries, stated.as_array(named));

            Transpose layer{};
            for (std::size_t index = 0; index < entries.size(); ++index)
            {
                SA_TRY_ASSIGN(const int64_t axis,
                              entries[index].as_int(std::string(context) + " '" +
                                                    std::string(K_PERM_KEY) + "[" +
                                                    std::to_string(index) + "]'"));
                layer.permutation.push_back(axis);
            }
            SA_TRY(SA_ANNOTATE(layer.output_shape(inputs[0]).status(), "{}", named));

            return Detail::wrap_block(std::move(layer), inputs[0]);
        }

    } // namespace NN
} // namespace SushiAI
