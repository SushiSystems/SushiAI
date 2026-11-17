/**************************************************************************/
/* shape.cpp                                                              */
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
 * @file shape.cpp
 * @brief Defines the host rules over rows and channels: reshape, transpose, gather, slice, concat,
 *        affine.
 * @author Mustafa Garip
 */

#include "rules.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/graph/permuted_copy.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Returns the last extent of @p shape, or 1 for a rank-0 shape. */
                int64_t last_extent(const Shape& shape) noexcept
                {
                    return shape.rank() == 0 ? 1 : shape[shape.rank() - 1];
                }
            } // namespace

            Status host_reshape(const HostNodeContext& context) noexcept
            {
                const std::vector<double>& x = context.input(0);
                std::copy(x.begin(), x.end(), context.output(0).begin());
                return Status();
            }

            Status host_transpose(const HostNodeContext& context) noexcept
            {
                const std::vector<double>& x = context.input(0);
                const Shape& source = context.input_shape(0);
                const Shape& result = context.output_value(0).shape;
                const int32_t rank = source.rank();

                // The step in the operand of each per-sample axis, then of each result axis.
                std::array<int64_t, static_cast<std::size_t>(K_MAX_RANK)> steps{};
                int64_t running = 1;
                for (int32_t axis = rank; axis-- > 0;)
                {
                    steps[static_cast<std::size_t>(axis)] = running;
                    running *= source[axis];
                }

                const auto permutation = permutation_of(
                    context.node(), static_cast<std::size_t>(rank - 1));
                std::array<int64_t, static_cast<std::size_t>(K_MAX_RANK)> result_steps{};
                result_steps[0] = steps[0];
                for (int32_t axis = 1; axis < rank; ++axis)
                {
                    const std::size_t taken =
                        static_cast<std::size_t>(permutation[static_cast<std::size_t>(axis - 1)]);
                    result_steps[static_cast<std::size_t>(axis)] = steps[taken + 1];
                }

                std::vector<double>& y = context.output(0);
                for (std::size_t at = 0; at < y.size(); ++at)
                {
                    std::size_t rest = at;
                    int64_t from = 0;
                    for (int32_t axis = rank; axis-- > 0;)
                    {
                        const std::size_t extent = static_cast<std::size_t>(result[axis]);
                        from += static_cast<int64_t>(rest % extent)
                                * result_steps[static_cast<std::size_t>(axis)];
                        rest /= extent;
                    }

                    y[at] = x[static_cast<std::size_t>(from)];
                }

                return Status();
            }

            Status host_gather_rows(const HostNodeContext& context) noexcept
            {
                const std::vector<double>& table = context.input(0);
                const std::vector<double>& indices = context.input(1);
                const int64_t rows = context.input_shape(0)[0];
                const std::size_t width = static_cast<std::size_t>(context.input_shape(0)[1]);
                std::vector<double>& y = context.output(0);

                for (std::size_t at = 0; at < indices.size(); ++at)
                {
                    const int64_t row = static_cast<int64_t>(indices[at]);
                    SA_FAIL_IF(row < 0 || row >= rows, StatusCode::InvalidDescription,
                               "GATHER_ROWS reads row {} of a table of {} rows", row, rows);
                    std::copy_n(table.begin() + static_cast<std::ptrdiff_t>(row) * width, width,
                                y.begin() + static_cast<std::ptrdiff_t>(at * width));
                }

                return Status();
            }

            Status host_slice_channels(const HostNodeContext& context) noexcept
            {
                const std::vector<double>& x = context.input(0);
                const std::size_t channels = static_cast<std::size_t>(
                    last_extent(context.input_shape(0)));
                const std::size_t offset = static_cast<std::size_t>(context.attribute_int(0));
                std::vector<double>& y = context.output(0);
                const std::size_t count = static_cast<std::size_t>(
                    last_extent(context.output_value(0).shape));

                for (std::size_t at = 0; at < y.size(); ++at)
                    y[at] = x[(at / count) * channels + offset + at % count];

                return Status();
            }

            Status host_concat_channels(const HostNodeContext& context) noexcept
            {
                std::vector<double>& y = context.output(0);
                const std::size_t channels = static_cast<std::size_t>(
                    last_extent(context.output_value(0).shape));
                std::size_t offset = 0;

                for (std::size_t part = 0; part < context.input_count(); ++part)
                {
                    const std::vector<double>& x = context.input(part);
                    const std::size_t width = static_cast<std::size_t>(
                        last_extent(context.input_shape(part)));
                    for (std::size_t at = 0; at < x.size(); ++at)
                        y[(at / width) * channels + offset + at % width] = x[at];

                    offset += width;
                }

                return Status();
            }

            Status host_channel_affine(const HostNodeContext& context) noexcept
            {
                const std::vector<double>& x = context.input(0);
                const std::vector<double>& gamma = context.input(1);
                const std::vector<double>& beta = context.input(2);
                std::vector<double>& y = context.output(0);

                for (std::size_t at = 0; at < y.size(); ++at)
                    y[at] = gamma[at % gamma.size()] * x[at] + beta[at % beta.size()];

                return Status();
            }
        } // namespace Detail
    } // namespace Graph
} // namespace SushiAI
