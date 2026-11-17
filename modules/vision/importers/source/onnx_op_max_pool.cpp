/**************************************************************************/
/* onnx_op_max_pool.cpp                                                   */
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
 * @file onnx_op_max_pool.cpp
 * @brief Defines the translation of an ONNX MaxPool into a maxpool2d row.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every attribute the translation of MaxPool reads. */
                constexpr std::string_view K_MAX_POOL_ATTRIBUTES[] = {
                    "kernel_shape", "strides", "pads", "dilations", "auto_pad", "ceil_mode",
                    "storage_order"};

                /** @brief Holds the rank of one sample of an image, [C, H, W]. */
                constexpr std::size_t K_IMAGE_RANK = 3;
            } // namespace

            Status translate_onnx_max_pool(const Interchange::OnnxNode& node,
                                           OnnxProgress& progress)
            {
                SA_TRY(check_operand_count(node, 1, 1));
                SA_TRY(check_output_count(node, 1));
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>(
                              K_MAX_POOL_ATTRIBUTES, std::size(K_MAX_POOL_ATTRIBUTES))));
                SA_TRY_ASSIGN(const std::int64_t ceiling, read_int_attribute(node, "ceil_mode", 0));
                SA_TRY_ASSIGN(const std::int64_t order,
                              read_int_attribute(node, "storage_order", 0));
                SA_FAIL_IF(ceiling != 0 || order != 0, StatusCode::Unsupported,
                           "{} states ceil_mode {} and storage_order {}, and only 0 and 0 are "
                           "translated",
                           describe_node(node), ceiling, order);

                SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, 0));
                SA_FAIL_IF(input->dimensions.size() != K_IMAGE_RANK, StatusCode::Unsupported,
                           "{} reads a value of rank {}, and a pooling reads [C, H, W]",
                           describe_node(node), input->dimensions.size());
                SA_TRY_ASSIGN(const OnnxWindow window, read_window(node, {}));
                SA_FAIL_IF(window.dilation[0] != 1 || window.dilation[1] != 1,
                           StatusCode::Unsupported,
                           "{} states a dilation, and the maxpool2d row has none",
                           describe_node(node));
                SA_TRY_ASSIGN(const std::int64_t out_h,
                              window_extent(node, window, 0, input->dimensions[1]));
                SA_TRY_ASSIGN(const std::int64_t out_w,
                              window_extent(node, window, 1, input->dimensions[2]));

                std::vector<std::string> keys = {"kernel", "stride"};
                std::vector<Config::Value> values = {
                    pair_value(window.kernel[0], window.kernel[1]),
                    pair_value(window.stride[0], window.stride[1])};
                const bool symmetric = window.pad_begin[0] == window.pad_end[0] &&
                                       window.pad_begin[1] == window.pad_end[1];
                if (symmetric)
                {
                    keys.emplace_back("padding");
                    values.push_back(pair_value(window.pad_begin[0], window.pad_begin[1]));
                }
                else
                {
                    keys.emplace_back("leading_padding");
                    values.push_back(pair_value(window.pad_begin[0], window.pad_begin[1]));
                    keys.emplace_back("end_padding");
                    values.push_back(pair_value(window.pad_end[0], window.pad_end[1]));
                }
                const std::int64_t row = append_row(progress, "maxpool2d", std::move(keys),
                                                    std::move(values), {input->row});
                define_value(progress, node.outputs[0], row, {input->dimensions[0], out_h, out_w});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
