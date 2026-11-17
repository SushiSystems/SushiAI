/**************************************************************************/
/* caffe_op_convolution.cpp                                               */
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
 * @file caffe_op_convolution.cpp
 * @brief Implements the translator of a Caffe Convolution layer.
 * @author Mustafa Garip
 */

#include "caffe_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/axis_translation.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/shape/shape.hpp>

#include "caffe_parameters.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail::Caffe
        {
            namespace
            {
                /** @brief Holds what a convolution_param states, defaults filled in. */
                struct Convolution
                {
                    std::int64_t filters = 0;      /**< Output channels. */
                    bool use_bias = true;          /**< Whether the layer adds a bias. */
                    WindowPair kernel;             /**< Kernel extents. */
                    WindowPair stride{1, 1};       /**< Step of the window. */
                    WindowPair padding{0, 0};      /**< Pad on each side of an axis. */
                    WindowPair dilation{1, 1};     /**< Gap between the taps of the kernel. */
                    std::int64_t groups = 1;       /**< Channel groups; 1 convolves them all. */

                    /** @brief Tells whether every channel has its own filter and its own taps. */
                    [[nodiscard]] bool is_depthwise(std::int64_t channels) const noexcept
                    {
                        return groups > 1 && groups == channels && groups == filters;
                    }
                };

                /**
                 * @brief Reads the convolution_param of @p layer over its defaults.
                 * @return Unsupported for another channel axis; InvalidDescription for
                 *         a value that is malformed or out of range.
                 */
                [[nodiscard]] Result<Convolution> read_convolution(
                    const Interchange::CaffeLayer& layer)
                {
                    const ParameterReader parameters(layer, "convolution_param");
                    Convolution read;
                    SA_TRY_ASSIGN(read.filters, parameters.read_integer("num_output", 0));
                    SA_TRY_ASSIGN(read.use_bias, parameters.read_bool("bias_term", true));
                    SA_TRY_ASSIGN(read.kernel, parameters.read_window("kernel_size", "kernel_h",
                                                                      "kernel_w", WindowPair{}));
                    SA_TRY_ASSIGN(read.stride, parameters.read_window("stride", "stride_h",
                                                                      "stride_w", read.stride));
                    SA_TRY_ASSIGN(read.padding, parameters.read_window("pad", "pad_h", "pad_w",
                                                                       read.padding));
                    SA_TRY_ASSIGN(read.dilation, parameters.read_window("dilation", "dilation_h",
                                                                        "dilation_w",
                                                                        read.dilation));
                    SA_TRY_ASSIGN(read.groups, parameters.read_integer("group", 1));
                    SA_TRY_ASSIGN(const std::int64_t axis, parameters.read_integer("axis", 1));

                    SA_FAIL_IF(axis != 1, StatusCode::Unsupported,
                               "{} states axis {}; only the channel axis 1 is translated",
                               describe_layer(layer), axis);
                    SA_FAIL_IF(read.filters < 1 || read.kernel.height < 1 || read.kernel.width < 1 ||
                                   read.stride.height < 1 || read.stride.width < 1 ||
                                   read.padding.height < 0 || read.padding.width < 0 ||
                                   read.dilation.height < 1 || read.dilation.width < 1 ||
                                   read.groups < 1,
                               StatusCode::InvalidDescription,
                               "{} needs a num_output, a kernel, a stride and a group of 1 or "
                               "more, a pad of 0 or more and a dilation of 1 or more",
                               describe_layer(layer));
                    return read;
                }

                /** @brief Returns the extent one axis of a convolution yields, or 0 for none. */
                [[nodiscard]] std::int64_t output_extent(std::int64_t input, std::int64_t kernel,
                                                         std::int64_t stride, std::int64_t pad,
                                                         std::int64_t dilation) noexcept
                {
                    const std::int64_t span = input + 2 * pad - dilation * (kernel - 1) - 1;
                    return span < 0 ? 0 : span / stride + 1;
                }

                /** @brief Returns the attributes of the conv2d row of @p read. */
                [[nodiscard]] Config::Value describe_row(const Convolution& read)
                {
                    std::vector<std::string> keys = {"filters", "kernel", "stride", "padding"};
                    std::vector<Config::Value> values = {
                        Config::Value::integer(read.filters),
                        pair_value(read.kernel.height, read.kernel.width),
                        pair_value(read.stride.height, read.stride.width),
                        pair_value(read.padding.height, read.padding.width)};
                    if (read.dilation.height != 1 || read.dilation.width != 1)
                    {
                        keys.emplace_back("dilation");
                        values.push_back(pair_value(read.dilation.height, read.dilation.width));
                    }
                    if (read.groups != 1)
                    {
                        keys.emplace_back("groups");
                        values.push_back(Config::Value::integer(read.groups));
                    }
                    keys.emplace_back("use_bias");
                    values.push_back(Config::Value::boolean(read.use_bias));
                    return make_attributes(std::move(keys), std::move(values));
                }

                /** @brief Appends the weight and bias tensors of the row @p row. */
                [[nodiscard]] Status store_weights(const Interchange::CaffeLayer& layer,
                                                   const Convolution& read, std::int64_t channels,
                                                   std::int64_t row, TranslationProgress& progress)
                {
                    SA_TRY(expect_blobs(progress, layer, read.use_bias ? 2 : 1));
                    if (!progress.has_weights)
                        return Status();

                    // A caffe weight is [out, in / group, kh, kw]; the blob holds that many floats.
                    const std::int64_t group_channels = channels / read.groups;
                    const bool depthwise = read.is_depthwise(channels);
                    const std::vector<std::int64_t> dimensions = {
                        read.filters, group_channels, read.kernel.height, read.kernel.width};
                    const std::size_t count = static_cast<std::size_t>(
                        read.filters * group_channels * read.kernel.height * read.kernel.width);
                    SA_TRY_ASSIGN(const std::vector<float> weights, read_blob(layer, 0, count));
                    SA_TRY_ASSIGN(const std::vector<float> record,
                                  Interchange::filter_to_record_layout(
                                      SushiRuntime::span<const float>(weights.data(),
                                                                      weights.size()),
                                      dimensions, depthwise));
                    SA_TRY_ASSIGN(const std::vector<std::int64_t> record_shape,
                                  Interchange::filter_record_dimensions(dimensions, depthwise));
                    SA_TRY_ASSIGN(const Shape stored,
                                  Shape::make(SushiRuntime::span<const std::int64_t>(
                                      record_shape.data(), record_shape.size())));
                    append_tensor(progress, entry_name(row, "weight"), stored, record);
                    if (read.use_bias)
                    {
                        SA_TRY_ASSIGN(const std::vector<float> bias,
                                      read_blob(layer, 1, static_cast<std::size_t>(read.filters)));
                        append_tensor(progress, entry_name(row, "bias"), Shape{read.filters}, bias);
                    }
                    return Status();
                }
            } // namespace

            Status translate_convolution(const Interchange::CaffeLayer& layer, std::int64_t index,
                                         TranslationProgress& progress)
            {
                SA_TRY(expect_shape(layer, 1));
                SA_TRY_ASSIGN(const BlobValue source, read_bottom(progress, layer, 0));
                SA_TRY(expect_image(layer, source));
                SA_TRY_ASSIGN(const Convolution read, read_convolution(layer));
                SA_FAIL_IF(source.channels % read.groups != 0 || read.filters % read.groups != 0,
                           StatusCode::InvalidDescription,
                           "{} states group {}, which divides neither the {} channels it reads "
                           "nor its {} filters",
                           describe_layer(layer), read.groups, source.channels, read.filters);

                const std::int64_t height = output_extent(source.height, read.kernel.height,
                                                          read.stride.height, read.padding.height,
                                                          read.dilation.height);
                const std::int64_t width = output_extent(source.width, read.kernel.width,
                                                         read.stride.width, read.padding.width,
                                                         read.dilation.width);
                SA_FAIL_IF(height < 1 || width < 1, StatusCode::InvalidDescription,
                           "{}: a kernel of {} by {} with pad {} by {} does not fit the {} by {} "
                           "blob it reads",
                           describe_layer(layer), read.kernel.height, read.kernel.width,
                           read.padding.height, read.padding.width, source.height, source.width);

                const std::int64_t sources[] = {source.row};
                const std::int64_t row =
                    append_row(progress, "conv2d", describe_row(read), sources);
                SA_TRY(store_weights(layer, read, source.channels, row, progress));

                BlobValue result;
                result.row = row;
                result.is_image = true;
                result.height = height;
                result.width = width;
                result.channels = read.filters;
                write_blob(progress, layer.tops.front(), result, index);
                return Status();
            }
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
