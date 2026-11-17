/**************************************************************************/
/* onnx_op_conv.cpp                                                       */
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
 * @file onnx_op_conv.cpp
 * @brief Defines the translation of an ONNX Conv, grouped or depthwise included, into a conv2d row.
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
#include <SushiAI/interchange/axis_translation.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every attribute the translation of Conv reads. */
                constexpr std::string_view K_CONV_ATTRIBUTES[] = {
                    "kernel_shape", "strides", "pads", "dilations", "group", "auto_pad"};

                /** @brief Holds the rank of a Conv weight, [out, in, kh, kw]. */
                constexpr std::size_t K_WEIGHT_RANK = 4;

                /** @brief Holds the rank of one sample of an image, [C, H, W]. */
                constexpr std::size_t K_IMAGE_RANK = 3;
            } // namespace

            Status translate_onnx_conv(const Interchange::OnnxNode& node, OnnxProgress& progress)
            {
                SA_TRY(check_operand_count(node, 2, 3));
                SA_TRY(check_output_count(node, 1));
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>(
                              K_CONV_ATTRIBUTES, std::size(K_CONV_ATTRIBUTES))));
                SA_TRY_ASSIGN(const std::int64_t group, read_int_attribute(node, "group", 1));
                SA_FAIL_IF(group < 1, StatusCode::InvalidDescription,
                           "{} states group {}, and a group is 1 or more", describe_node(node),
                           group);

                SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, 0));
                SA_FAIL_IF(input->dimensions.size() != K_IMAGE_RANK, StatusCode::Unsupported,
                           "{} reads a value of rank {}, and a convolution reads [C, H, W]",
                           describe_node(node), input->dimensions.size());
                SA_TRY_ASSIGN(const OnnxConstant* weight,
                              find_real_operand(progress, node, 1, K_WEIGHT_RANK));
                const std::vector<std::int64_t>& extents = weight->dimensions;
                SA_FAIL_IF(extents[1] * group != input->dimensions[0] ||
                               extents[0] % group != 0,
                           StatusCode::InvalidDescription,
                           "{}: the weight has {} filters that read {} channels each in {} groups, "
                           "and the value has {} channels",
                           describe_node(node), extents[0], extents[1], group,
                           input->dimensions[0]);
                const bool depthwise =
                    group > 1 && group == input->dimensions[0] && group == extents[0];

                SA_TRY_ASSIGN(const OnnxWindow window, read_window(node, {extents[2], extents[3]}));
                SA_TRY_ASSIGN(const std::int64_t out_h,
                              window_extent(node, window, 0, input->dimensions[1]));
                SA_TRY_ASSIGN(const std::int64_t out_w,
                              window_extent(node, window, 1, input->dimensions[2]));
                SA_FAIL_IF(window.pad_end[0] < window.pad_begin[0] ||
                               window.pad_end[1] < window.pad_begin[1],
                           StatusCode::Unsupported,
                           "{} pads fewer rows below or columns right than above or left, which "
                           "the conv2d row cannot state",
                           describe_node(node));
                SA_FAIL_IF(depthwise && (window.pad_end[0] != window.pad_begin[0] ||
                                         window.pad_end[1] != window.pad_begin[1]),
                           StatusCode::Unsupported,
                           "{} filters each channel on its own and pads its two sides of an axis "
                           "differently, which a depthwise conv2d row cannot state",
                           describe_node(node));

                const bool has_bias = has_operand(node, 2);
                const OnnxConstant* bias = nullptr;
                if (has_bias)
                {
                    SA_TRY_ASSIGN(const OnnxConstant* stated,
                                  find_real_operand(progress, node, 2, 1));
                    SA_FAIL_IF(stated->dimensions[0] != extents[0], StatusCode::InvalidDescription,
                               "{}: the bias holds {} values for {} filters", describe_node(node),
                               stated->dimensions[0], extents[0]);
                    bias = stated;
                }

                std::vector<std::string> keys = {"filters", "kernel", "stride", "padding"};
                std::vector<Config::Value> values = {
                    Config::Value::integer(extents[0]),
                    pair_value(window.kernel[0], window.kernel[1]),
                    pair_value(window.stride[0], window.stride[1]),
                    pair_value(window.pad_begin[0], window.pad_begin[1])};
                if (window.dilation[0] != 1 || window.dilation[1] != 1)
                {
                    keys.emplace_back("dilation");
                    values.push_back(pair_value(window.dilation[0], window.dilation[1]));
                }
                if (group != 1)
                {
                    keys.emplace_back("groups");
                    values.push_back(Config::Value::integer(group));
                }
                const std::int64_t trailing_h = window.pad_end[0] - window.pad_begin[0];
                const std::int64_t trailing_w = window.pad_end[1] - window.pad_begin[1];
                if (trailing_h != 0 || trailing_w != 0)
                {
                    keys.emplace_back("trailing_padding");
                    values.push_back(pair_value(trailing_h, trailing_w));
                }
                keys.emplace_back("use_bias");
                values.push_back(Config::Value::boolean(has_bias));

                SA_TRY_ASSIGN(const std::vector<float> laid_out,
                              Interchange::filter_to_record_layout(
                                  SushiRuntime::span<const float>(weight->reals.data(),
                                                                  weight->reals.size()),
                                  extents, depthwise));
                SA_TRY_ASSIGN(const std::vector<std::int64_t> record_shape,
                              Interchange::filter_record_dimensions(extents, depthwise));

                const std::int64_t row =
                    append_row(progress, "conv2d", std::move(keys), std::move(values),
                               {input->row});
                SA_TRY(store_weight(progress, row, "weight", record_shape, laid_out));
                if (bias != nullptr)
                    SA_TRY(store_weight(progress, row, "bias", {extents[0]}, bias->reals));
                define_value(progress, node.outputs[0], row, {extents[0], out_h, out_w});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
