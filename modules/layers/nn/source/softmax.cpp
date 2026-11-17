/**************************************************************************/
/* softmax.cpp                                                            */
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
 * @file softmax.cpp
 * @brief Implements Softmax's trace form, its shape rule and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/softmax.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
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
            /** @brief Names the key that holds the axis of a "softmax" row. */
            constexpr std::string_view K_AXIS_KEY = "axis";

            /** @brief Lists every key a "softmax" row accepts. */
            constexpr std::string_view K_SOFTMAX_KEYS[] = {K_AXIS_KEY};

            /** @brief Returns the position of @p axis among @p rank axes, or -1 out of range. */
            int64_t resolve_axis(int64_t axis, int64_t rank) noexcept
            {
                if (axis < -rank || axis >= rank)
                    return -1;

                return axis < 0 ? axis + rank : axis;
            }

            /** @brief Returns the permutation of @p rank per-sample axes that moves @p axis last. */
            std::vector<int64_t> move_last(int64_t axis, int64_t rank)
            {
                std::vector<int64_t> permutation;
                for (int64_t position = 0; position < rank; ++position)
                {
                    if (position != axis)
                        permutation.push_back(position);
                }
                permutation.push_back(axis);

                return permutation;
            }

            /** @brief Traces the softmax of @p x, or returns why it does not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_softmax(const Softmax& layer,
                                                               const Trace& trace,
                                                               Graph::ValueID x)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                const Shape shape = builder.graph().value(x).shape;
                const int64_t sample_rank = shape.rank() - 1;
                const int64_t axis = resolve_axis(layer.axis, sample_rank);
                SA_FAIL_IF(axis < 0, StatusCode::InvalidDescription,
                           "Softmax over the axis {} needs a per-sample rank above it; the value "
                           "is {}",
                           layer.axis, shape.to_string());

                if (axis == sample_rank - 1)
                {
                    SA_TRY_ASSIGN(const Graph::ValueID last, builder.softmax(x));
                    return last;
                }

                // The transposes keep the batch axis first and permute the per-sample axes.
                const std::vector<int64_t> forward_permutation = move_last(axis, sample_rank);
                const auto inverse = Graph::inverse_axis_permutation(
                    SushiRuntime::span<const int64_t>(forward_permutation.data(),
                                                      forward_permutation.size()));
                SA_TRY_ASSIGN(const Graph::ValueID moved,
                              builder.transpose(x, SushiRuntime::span<const int64_t>(
                                                       forward_permutation.data(),
                                                       forward_permutation.size())));
                SA_TRY_ASSIGN(const Graph::ValueID normalised, builder.softmax(moved));
                SA_TRY_ASSIGN(const Graph::ValueID restored,
                              builder.transpose(normalised, SushiRuntime::span<const int64_t>(
                                                                inverse.data(),
                                                                forward_permutation.size())));
                return restored;
            }
        } // namespace

        Graph::ValueID Softmax::forward(const Trace& trace, Graph::ValueID x) noexcept
        {
            return trace.value_or_fail(trace_softmax(*this, trace, x), x);
        }

        Result<Shape> Softmax::output_shape(const Shape& input) const noexcept
        {
            SA_FAIL_IF(resolve_axis(axis, input.rank()) < 0, StatusCode::InvalidDescription,
                       "Softmax over the axis {} needs a per-sample rank above it; the input is {}",
                       axis, input.to_string());

            return input;
        }

        Result<std::unique_ptr<Node>> make_softmax_node(ShapeSpan inputs,
                                                        const Config::Value& attributes,
                                                        std::string_view context) noexcept
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_SOFTMAX_KEYS),
                                               context));
            SA_FAIL_IF(inputs.size() != 1, StatusCode::InvalidDescription,
                       "{} reads one input, but it was given {}", context, inputs.size());
            SA_FAIL_IF(inputs[0].rank() < 1, StatusCode::InvalidDescription,
                       "{} was given a per-sample input of {}, but it reads an input of rank 1 or "
                       "more",
                       context, inputs[0].to_string());

            Softmax layer{};
            if (attributes.has(K_AXIS_KEY))
            {
                SA_TRY_ASSIGN(const Config::Value& stated, attributes.at(K_AXIS_KEY, context));
                SA_TRY_ASSIGN(layer.axis, stated.as_int(std::string(context) + " 'axis'"));
            }
            SA_TRY(SA_ANNOTATE(layer.output_shape(inputs[0]).status(), "{} '{}'", context,
                               K_AXIS_KEY));

            return Detail::wrap_block(std::move(layer), inputs[0]);
        }

    } // namespace NN
} // namespace SushiAI
