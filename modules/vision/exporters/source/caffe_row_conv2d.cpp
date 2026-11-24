/**************************************************************************/
/* caffe_row_conv2d.cpp                                                   */
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
 * @file caffe_row_conv2d.cpp
 * @brief Defines the Caffe writer of a "conv2d" row: one Convolution layer.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/axis_translation.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/io/protobuf_text.hpp>
#include "caffe_export_progress.hpp"
#include "caffe_row_table.hpp"

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

                /** @brief Holds what the row states, defaults filled in. */
                struct ConvRow
                {
                    std::int64_t filters = 0;            /**< Output channels. */
                    std::int64_t groups = 1;             /**< Channel groups. */
                    bool use_bias = true;                /**< Whether the row adds a bias. */
                    std::vector<std::int64_t> kernel;   /**< Height and width of the window. */
                    std::vector<std::int64_t> stride;   /**< Step along height and width. */
                    std::vector<std::int64_t> padding;  /**< Pad on both sides of each axis. */
                    std::vector<std::int64_t> dilation; /**< Gap between the taps. */
                };

                /** @brief Returns what the row states; Unsupported for a trailing padding. */
                Result<ConvRow> read_conv_row(const CaffeRowCall& call)
                {
                    const RowKeys keys = row_keys(call);
                    ConvRow row;
                    SA_TRY_ASSIGN(row.filters, read_int_key(keys, "filters", 0));
                    SA_TRY_ASSIGN(row.groups, read_int_key(keys, "groups", 1));
                    SA_TRY_ASSIGN(row.use_bias, read_bool_key(keys, "use_bias", true));
                    SA_TRY_ASSIGN(row.kernel, read_pair_key(keys, "kernel", 1));
                    SA_TRY_ASSIGN(row.stride, read_pair_key(keys, "stride", 1));
                    SA_TRY_ASSIGN(row.padding, read_pair_key(keys, "padding", 0));
                    SA_TRY_ASSIGN(row.dilation, read_pair_key(keys, "dilation", 1));
                    SA_TRY_ASSIGN(const std::vector<std::int64_t> trailing,
                                  read_pair_key(keys, "trailing_padding", 0));
                    SA_FAIL_IF(trailing[0] != 0 || trailing[1] != 0, StatusCode::Unsupported,
                               "it states a trailing padding of {} by {}, and Caffe pads both "
                               "sides alike",
                               trailing[0], trailing[1]);
                    return row;
                }

                /** @brief Returns the extent the window leaves along @p axis of @p input. */
                Result<std::int64_t> output_extent(const CaffeRowCall& call, const ConvRow& row,
                                                   std::size_t axis, std::int64_t input)
                {
                    const std::int64_t span = input + 2 * row.padding[axis] -
                                              row.dilation[axis] * (row.kernel[axis] - 1) - 1;
                    SA_FAIL_IF(span < 0 || row.stride[axis] < 1, StatusCode::InvalidDescription,
                               "row {} 'conv2d' does not fit its window in the input",
                               call.index);
                    return span / row.stride[axis] + 1;
                }

                /** @brief Returns the convolution_param of @p row. */
                IO::TextField describe_row(const ConvRow& row)
                {
                    std::vector<IO::TextField> fields = {
                        make_int_field("num_output", row.filters),
                        make_word_field("bias_term", row.use_bias ? "true" : "false"),
                        make_int_field("pad_h", row.padding[0]),
                        make_int_field("pad_w", row.padding[1]),
                        make_int_field("kernel_h", row.kernel[0]),
                        make_int_field("kernel_w", row.kernel[1]),
                        make_int_field("stride_h", row.stride[0]),
                        make_int_field("stride_w", row.stride[1])};
                    if (row.groups != 1)
                        fields.push_back(make_int_field("group", row.groups));
                    if (row.dilation[0] != 1 || row.dilation[1] != 1)
                    {
                        fields.push_back(make_int_field("dilation_h", row.dilation[0]));
                        fields.push_back(make_int_field("dilation_w", row.dilation[1]));
                    }
                    return make_message("convolution_param", std::move(fields));
                }

                /**
                 * @brief Returns the weight blob [out, in / g, kh, kw] and the bias blob.
                 * @return Unsupported for a kernel the importer reads back in another layout.
                 */
                Result<std::vector<Interchange::CaffeBlob>> read_blobs(
                    const CaffeRowCall& call, const CaffeExportProgress& progress,
                    const ConvRow& row, std::int64_t channels)
                {
                    std::vector<Interchange::CaffeBlob> blobs;
                    if (!progress.has_weights)
                        return blobs;
                    std::vector<std::int64_t> record;
                    SA_TRY_ASSIGN(const std::vector<float> stored,
                                  read_weight(*progress.model, call.index, "weight",
                                              K_CAFFE_FLOAT32_HOLDER, record));
                    const bool depthwise = record.size() == K_DEPTHWISE_RANK;
                    const bool read_as_depthwise =
                        row.groups > 1 && row.groups == channels && row.groups == row.filters;
                    SA_FAIL_IF(depthwise != read_as_depthwise, StatusCode::Unsupported,
                               "it holds a kernel of rank {} for {} groups of {} filters over {} "
                               "channels, and the Caffe importer reads it back in the other "
                               "layout",
                               record.size(), row.groups, row.filters, channels);
                    SA_TRY_ASSIGN(const std::vector<float> laid_out,
                                  Interchange::filter_to_source_layout(
                                      SushiRuntime::span<const float>(stored.data(),
                                                                      stored.size()),
                                      record, depthwise));
                    std::vector<std::int64_t> extents =
                        depthwise ? std::vector<std::int64_t>{record[2], 1, record[0], record[1]}
                                  : std::vector<std::int64_t>{record[0], record[3], record[1],
                                                              record[2]};
                    SA_FAIL_IF(extents[0] != row.filters || extents[1] * row.groups != channels,
                               StatusCode::InvalidDescription,
                               "row {} 'conv2d' holds a weight of {} filters reading {} channels "
                               "in {} groups, and the row reads {} channels into {} filters",
                               call.index, extents[0], extents[1], row.groups, channels,
                               row.filters);
                    blobs.push_back(make_blob(std::move(extents), laid_out));
                    if (row.use_bias)
                    {
                        std::vector<std::int64_t> bias_extents;
                        SA_TRY_ASSIGN(const std::vector<float> bias,
                                      read_weight(*progress.model, call.index, "bias",
                                                  K_CAFFE_FLOAT32_HOLDER, bias_extents));
                        blobs.push_back(make_blob(std::move(bias_extents), bias));
                    }
                    return blobs;
                }
            } // namespace

            Status write_caffe_conv2d(const CaffeRowCall& call, CaffeExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>(
                                                     K_CONV2D_KEYS, std::size(K_CONV2D_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const CaffeRowValue& input = *call.sources[0];
                SA_TRY(check_image(input));
                SA_TRY_ASSIGN(const ConvRow row, read_conv_row(call));
                SA_TRY(check_bias(call, progress, row.use_bias));
                SA_TRY_ASSIGN(const std::int64_t height,
                              output_extent(call, row, 0, input.shape[0]));
                SA_TRY_ASSIGN(const std::int64_t width,
                              output_extent(call, row, 1, input.shape[1]));
                SA_TRY_ASSIGN(std::vector<Interchange::CaffeBlob> blobs,
                              read_blobs(call, progress, row, input.shape[2]));
                add_layer(progress, call, "Convolution", {input.blob}, describe_row(row),
                          std::move(blobs));
                define_row_value(progress, call, {height, width, row.filters});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
