/**************************************************************************/
/* onnx_op_split.cpp                                                      */
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
 * @file onnx_op_split.cpp
 * @brief Defines the translation of an ONNX Split along the channel axis into slice_channels rows.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
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
                /** @brief Lists every attribute the translation of Split reads. */
                constexpr std::string_view K_SPLIT_ATTRIBUTES[] = {"axis", "num_outputs"};

                /** @brief Holds the rank of one sample of an image, [C, H, W]. */
                constexpr std::size_t K_IMAGE_RANK = 3;

                /** @brief Holds the rank of the batch of images the axes are counted in. */
                constexpr std::int64_t K_BATCH_RANK = 4;

                /**
                 * @brief Returns the channels of each output of the split.
                 * @return InvalidDescription for sizes that do not add up to @p channels, and
                 *         Unsupported for outputs that would not hold equal parts.
                 */
                Result<std::vector<std::int64_t>> read_parts(OnnxProgress& progress,
                                                             const Interchange::OnnxNode& node,
                                                             std::int64_t channels)
                {
                    const std::int64_t count = static_cast<std::int64_t>(node.outputs.size());
                    SA_TRY_ASSIGN(const std::int64_t stated,
                                  read_int_attribute(node, "num_outputs", count));
                    SA_FAIL_IF(stated != count, StatusCode::InvalidDescription,
                               "{} states num_outputs {} and has {} outputs", describe_node(node),
                               stated, count);
                    if (!has_operand(node, 1))
                    {
                        SA_FAIL_IF(channels % count != 0, StatusCode::Unsupported,
                                   "{} splits {} channels into {} outputs, and only equal parts "
                                   "are translated",
                                   describe_node(node), channels, count);
                        return std::vector<std::int64_t>(static_cast<std::size_t>(count),
                                                         channels / count);
                    }

                    SA_TRY_ASSIGN(const OnnxConstant* sizes,
                                  find_operand_constant(progress, node, 1));
                    std::int64_t total = 0;
                    for (const std::int64_t size : sizes->integers)
                        total += size;
                    SA_FAIL_IF(sizes->is_real || sizes->integers.size() != node.outputs.size() ||
                                   total != channels,
                               StatusCode::InvalidDescription,
                               "{}: the sizes do not list one part of each output that adds up to "
                               "the {} channels",
                               describe_node(node), channels);
                    for (const std::int64_t size : sizes->integers)
                    {
                        SA_FAIL_IF(size < 1, StatusCode::InvalidDescription,
                                   "{} states an empty part", describe_node(node));
                    }
                    return sizes->integers;
                }
            } // namespace

            Status translate_onnx_split(const Interchange::OnnxNode& node, OnnxProgress& progress)
            {
                SA_TRY(check_operand_count(node, 1, 2));
                SA_FAIL_IF(node.outputs.empty(), StatusCode::InvalidDescription,
                           "{} writes no output", describe_node(node));
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>(
                              K_SPLIT_ATTRIBUTES, std::size(K_SPLIT_ATTRIBUTES))));
                SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, 0));
                SA_FAIL_IF(input->dimensions.size() != K_IMAGE_RANK, StatusCode::Unsupported,
                           "{} reads a value of rank {}, and a split reads [C, H, W]",
                           describe_node(node), input->dimensions.size());

                SA_TRY_ASSIGN(const std::int64_t axis, read_int_attribute(node, "axis", 0));
                SA_TRY_ASSIGN(const std::int64_t moved,
                              Interchange::channels_last_axis(axis, K_BATCH_RANK));
                SA_FAIL_IF(moved != K_BATCH_RANK - 1, StatusCode::Unsupported,
                           "{} splits axis {}, and only the channel axis is translated",
                           describe_node(node), axis);

                SA_TRY_ASSIGN(const std::vector<std::int64_t> parts,
                              read_parts(progress, node, input->dimensions[0]));
                std::int64_t offset = 0;
                for (std::size_t at = 0; at < parts.size(); ++at)
                {
                    const std::int64_t row = append_row(
                        progress, "slice_channels", {"offset", "count"},
                        {Config::Value::integer(offset), Config::Value::integer(parts[at])},
                        {input->row});
                    define_value(progress, node.outputs[at], row,
                                 {parts[at], input->dimensions[1], input->dimensions[2]});
                    offset += parts[at];
                }
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
