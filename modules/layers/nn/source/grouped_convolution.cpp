/**************************************************************************/
/* grouped_convolution.cpp                                                */
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
 * @file grouped_convolution.cpp
 * @brief Implements the trace of a grouped convolution as one convolution per group.
 * @author Mustafa Garip
 */

#include "grouped_convolution.hpp"

#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace Detail
        {
            namespace
            {
                /**
                 * @brief Traces the elements [@p offset, @p offset + @p count) of @p whole, as
                 *        a [@p shape] value.
                 * @param whole Any value; its elements are read in memory order.
                 */
                Graph::Traced slice_elements(const Graph::GraphBuilder& builder,
                                             Graph::ValueID whole, int64_t offset, int64_t count,
                                             const Shape& shape)
                {
                    const int64_t total = builder.graph().value(whole).shape.num_elements();
                    SA_TRY_ASSIGN(const Graph::ValueID flat,
                                  builder.reshape(whole, Shape{1, 1, 1, total}));
                    SA_TRY_ASSIGN(const Graph::ValueID part,
                                  builder.slice_channels(flat, offset, count));
                    return builder.reshape(part, shape);
                }
            } // namespace

            Graph::Traced trace_grouped_convolution(const Graph::GraphBuilder& builder,
                                                    Graph::ValueID x, Graph::ValueID weight,
                                                    Graph::ValueID bias,
                                                    const Graph::ConvGeometry& geometry,
                                                    int64_t out_height, int64_t out_width,
                                                    int64_t groups)
            {
                const Shape filters = builder.graph().value(weight).shape;
                const int64_t outputs = filters[0];
                const int64_t group_outputs = outputs / groups;
                const int64_t group_inputs = builder.graph().value(x).shape[3] / groups;
                const int64_t group_weight = group_outputs * filters[1] * filters[2] * filters[3];
                const Shape group_filters{group_outputs, filters[1], filters[2], filters[3]};

                std::vector<Graph::ValueID> parts;
                for (int64_t group = 0; group < groups; ++group)
                {
                    SA_TRY_ASSIGN(const Graph::ValueID slice,
                                  builder.slice_channels(x, group * group_inputs, group_inputs));
                    SA_TRY_ASSIGN(const Graph::ValueID taps,
                                  slice_elements(builder, weight, group * group_weight,
                                                 group_weight, group_filters));

                    Graph::ValueID offsets = Graph::K_NO_VALUE;
                    if (bias != Graph::K_NO_VALUE)
                    {
                        SA_TRY_ASSIGN(offsets,
                                      slice_elements(builder, bias, group * group_outputs,
                                                     group_outputs, Shape{group_outputs}));
                    }

                    SA_TRY_ASSIGN(const Graph::ValueID convolved,
                                  builder.conv2d_with_extents(slice, taps, offsets, geometry,
                                                              out_height, out_width));
                    parts.push_back(convolved);
                }

                return builder.concat_channels(
                    SushiRuntime::span<const Graph::ValueID>(parts.data(), parts.size()));
            }
        } // namespace Detail
    } // namespace NN
} // namespace SushiAI
