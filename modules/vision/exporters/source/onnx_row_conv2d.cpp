/**************************************************************************/
/* onnx_row_conv2d.cpp                                                    */
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
 * @file onnx_row_conv2d.cpp
 * @brief Defines the writer of the "conv2d" row, one ONNX Conv node.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/axis_translation.hpp>
#include "onnx_export_progress.hpp"
#include "onnx_row_table.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every key of the row the writer reads. */
                constexpr std::string_view K_CONV2D_KEYS[] = {
                    "filters", "kernel", "stride", "padding", "dilation", "use_bias",
                    "trailing_padding", "groups"};

                /** @brief Holds the rank of a depthwise kernel record, [kh, kw, C]. */
                constexpr std::size_t K_DEPTHWISE_RANK = 3;

                /** @brief Holds the window of the row along both image axes. */
                struct ConvWindow
                {
                    std::vector<std::int64_t> kernel;   /**< Height and width of the window. */
                    std::vector<std::int64_t> stride;   /**< Step along height and width. */
                    std::vector<std::int64_t> padding;  /**< Rows above, columns left. */
                    std::vector<std::int64_t> trailing; /**< Extra rows below, columns right. */
                };

                /** @brief Returns the window the row states; Unsupported for a dilation. */
                Result<ConvWindow> read_conv_window(const OnnxRowCall& call)
                {
                    const RowKeys keys = row_keys(call);
                    ConvWindow window;
                    SA_TRY_ASSIGN(window.kernel, read_pair_key(keys, "kernel", 1));
                    SA_TRY_ASSIGN(window.stride, read_pair_key(keys, "stride", 1));
                    SA_TRY_ASSIGN(window.padding, read_pair_key(keys, "padding", 0));
                    SA_TRY_ASSIGN(window.trailing, read_pair_key(keys, "trailing_padding", 0));
                    SA_TRY_ASSIGN(const std::vector<std::int64_t> dilation,
                                  read_pair_key(keys, "dilation", 1));
                    SA_FAIL_IF(dilation[0] != 1 || dilation[1] != 1, StatusCode::Unsupported,
                               "it states a dilation of {} by {}, and the ONNX exporter writes "
                               "a dilation of 1 only",
                               dilation[0], dilation[1]);
                    SA_FAIL_IF(window.trailing[0] < 0 || window.trailing[1] < 0,
                               StatusCode::Unsupported,
                               "it states a negative trailing padding, which the ONNX importer "
                               "does not read back");
                    return window;
                }

                /** @brief Returns the extent the window leaves along @p axis of @p input. */
                Result<std::int64_t> output_extent(const OnnxRowCall& call,
                                                   const ConvWindow& window, std::size_t axis,
                                                   std::int64_t input)
                {
                    const std::int64_t span = input + 2 * window.padding[axis] +
                                              window.trailing[axis] - window.kernel[axis];
                    SA_FAIL_IF(span < 0 || window.stride[axis] < 1,
                               StatusCode::InvalidDescription,
                               "row {} 'conv2d' does not fit its window in the input",
                               call.index);
                    return span / window.stride[axis] + 1;
                }
            } // namespace

            Status write_onnx_conv2d(const OnnxRowCall& call, OnnxExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>(
                                                     K_CONV2D_KEYS, std::size(K_CONV2D_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const OnnxRowValue& input = *call.sources[0];
                SA_TRY(check_image(input));
                SA_TRY_ASSIGN(const ConvWindow window, read_conv_window(call));
                SA_TRY_ASSIGN(const std::int64_t filters, read_int_key(keys, "filters", 0));
                SA_TRY_ASSIGN(const std::int64_t groups, read_int_key(keys, "groups", 1));
                std::vector<std::int64_t> record;
                SA_TRY_ASSIGN(const std::vector<float> stored,
                              read_weight(*progress.model, call.index, "weight",
                                          K_ONNX_FLOAT32_HOLDER, record));
                const bool depthwise = record.size() == K_DEPTHWISE_RANK;
                const std::int64_t channels = input.shape[2];
                SA_FAIL_IF(depthwise && (groups != channels || filters != channels),
                           StatusCode::InvalidDescription,
                           "row {} 'conv2d' holds a depthwise kernel and states {} groups of {} "
                           "filters over {} channels",
                           call.index, groups, filters, channels);
                SA_FAIL_IF(depthwise && (window.trailing[0] != 0 || window.trailing[1] != 0),
                           StatusCode::Unsupported,
                           "it filters each channel on its own with a trailing padding, which "
                           "the ONNX importer does not read back");
                SA_TRY_ASSIGN(const std::vector<float> laid_out,
                              Interchange::filter_to_source_layout(
                                  SushiRuntime::span<const float>(stored.data(), stored.size()),
                                  record, depthwise));
                const std::vector<std::int64_t> extents =
                    depthwise ? std::vector<std::int64_t>{record[2], 1, record[0], record[1]}
                              : std::vector<std::int64_t>{record[0], record[3], record[1],
                                                          record[2]};
                SA_FAIL_IF(extents[0] != filters || extents[1] * groups != channels,
                           StatusCode::InvalidDescription,
                           "row {} 'conv2d' holds a weight of {} filters reading {} channels in "
                           "{} groups, and the row reads {} channels into {} filters",
                           call.index, extents[0], extents[1], groups, channels, filters);
                SA_TRY_ASSIGN(const std::int64_t height,
                              output_extent(call, window, 0, input.shape[0]));
                SA_TRY_ASSIGN(const std::int64_t width,
                              output_extent(call, window, 1, input.shape[1]));
                const std::string weight_name = row_tensor_name(call.index, "weight");
                add_float_initializer(progress, weight_name, extents, laid_out);
                std::vector<std::string> inputs = {input.name, weight_name};
                SA_TRY_ASSIGN(const bool use_bias, read_bool_key(keys, "use_bias", true));
                const bool has_bias = find_weight(*progress.model, call.index, "bias") != nullptr;
                SA_FAIL_IF(use_bias != has_bias, StatusCode::InvalidDescription,
                           "row {} 'conv2d' states use_bias {}, and the weights disagree",
                           call.index, use_bias);
                if (use_bias)
                {
                    std::vector<std::int64_t> bias_extents;
                    SA_TRY_ASSIGN(const std::vector<float> bias,
                                  read_weight(*progress.model, call.index, "bias",
                                              K_ONNX_FLOAT32_HOLDER, bias_extents));
                    inputs.push_back(row_tensor_name(call.index, "bias"));
                    add_float_initializer(progress, inputs.back(), bias_extents, bias);
                }
                std::vector<Interchange::OnnxAttribute> attributes = {
                    make_ints_attribute("kernel_shape", window.kernel),
                    make_ints_attribute("strides", window.stride),
                    make_ints_attribute("pads",
                                        {window.padding[0], window.padding[1],
                                         window.padding[0] + window.trailing[0],
                                         window.padding[1] + window.trailing[1]})};
                if (groups != 1)
                    attributes.push_back(make_int_attribute("group", groups));
                add_node(progress, call, "Conv", "node", std::move(inputs),
                         {row_output(progress, call)}, std::move(attributes));
                define_row_value(progress, call, {height, width, filters});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
