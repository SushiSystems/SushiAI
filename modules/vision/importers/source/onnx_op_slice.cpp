/**************************************************************************/
/* onnx_op_slice.cpp                                                      */
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
 * @file onnx_op_slice.cpp
 * @brief Defines the translation of an ONNX Slice along the channel axis into a slice_channels row.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/axis_translation.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Holds the rank of one sample of an image, [C, H, W]. */
                constexpr std::size_t K_IMAGE_RANK = 3;

                /** @brief Holds the rank of the batch of images the axes are counted in. */
                constexpr std::int64_t K_BATCH_RANK = 4;

                /** @brief Returns the integers of operand @p position, or none when absent. */
                Result<std::vector<std::int64_t>> read_integers(OnnxProgress& progress,
                                                                const Interchange::OnnxNode& node,
                                                                std::size_t position)
                {
                    if (!has_operand(node, position))
                        return std::vector<std::int64_t>();
                    SA_TRY_ASSIGN(const OnnxConstant* operand,
                                  find_operand_constant(progress, node, position));
                    SA_FAIL_IF(operand->is_real || operand->dimensions.size() != 1,
                               StatusCode::InvalidDescription,
                               "{}: operand {} is not a list of integers", describe_node(node),
                               position);
                    return operand->integers;
                }

                /** @brief Returns @p position clamped into [0, @p extent], a negative one counted
                 *         from the end, as Slice does. */
                std::int64_t clamp_position(std::int64_t position, std::int64_t extent) noexcept
                {
                    if (position < 0)
                        position += extent;
                    return std::min(std::max<std::int64_t>(position, 0), extent);
                }
            } // namespace

            Status translate_onnx_slice(const Interchange::OnnxNode& node, OnnxProgress& progress)
            {
                SA_TRY(check_operand_count(node, 3, 5));
                SA_TRY(check_output_count(node, 1));
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>()));
                SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, 0));
                SA_FAIL_IF(input->dimensions.size() != K_IMAGE_RANK, StatusCode::Unsupported,
                           "{} reads a value of rank {}, and a slice reads [C, H, W]",
                           describe_node(node), input->dimensions.size());

                SA_TRY_ASSIGN(const std::vector<std::int64_t> starts,
                              read_integers(progress, node, 1));
                SA_TRY_ASSIGN(const std::vector<std::int64_t> ends,
                              read_integers(progress, node, 2));
                SA_TRY_ASSIGN(const std::vector<std::int64_t> axes,
                              read_integers(progress, node, 3));
                SA_TRY_ASSIGN(const std::vector<std::int64_t> steps,
                              read_integers(progress, node, 4));
                SA_FAIL_IF(starts.size() != 1 || ends.size() != 1 ||
                               (!axes.empty() && axes.size() != 1) ||
                               (!steps.empty() && steps.size() != 1),
                           StatusCode::Unsupported,
                           "{} slices {} axes, and only one axis is translated",
                           describe_node(node), starts.size());
                SA_FAIL_IF(!steps.empty() && steps[0] != 1, StatusCode::Unsupported,
                           "{} steps by {}, and only a step of 1 is translated",
                           describe_node(node), steps[0]);

                const std::int64_t axis = axes.empty() ? 0 : axes[0];
                SA_TRY_ASSIGN(const std::int64_t moved,
                              Interchange::channels_last_axis(axis, K_BATCH_RANK));
                SA_FAIL_IF(moved != K_BATCH_RANK - 1, StatusCode::Unsupported,
                           "{} slices axis {}, and only the channel axis is translated",
                           describe_node(node), axis);

                const std::int64_t channels = input->dimensions[0];
                const std::int64_t first = clamp_position(starts[0], channels);
                const std::int64_t last = clamp_position(ends[0], channels);
                SA_FAIL_IF(last <= first, StatusCode::InvalidDescription,
                           "{} keeps no channel: it slices {} to {} of {}", describe_node(node),
                           starts[0], ends[0], channels);

                const std::int64_t row = append_row(
                    progress, "slice_channels", {"offset", "count"},
                    {Config::Value::integer(first), Config::Value::integer(last - first)},
                    {input->row});
                define_value(progress, node.outputs[0], row,
                             {last - first, input->dimensions[1], input->dimensions[2]});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
