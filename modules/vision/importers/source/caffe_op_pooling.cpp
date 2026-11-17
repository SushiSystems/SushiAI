/**************************************************************************/
/* caffe_op_pooling.cpp                                                   */
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
 * @file caffe_op_pooling.cpp
 * @brief Implements the translator of a Caffe Pooling layer.
 * @author Mustafa Garip
 */

#include "caffe_operator_table.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/caffe_net.hpp>

#include "caffe_parameters.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail::Caffe
        {
            namespace
            {
                /** @brief Names how a Pooling layer combines a window. */
                enum class PoolMethod : std::uint8_t
                {
                    Max,     /**< The largest value of the window. */
                    Average, /**< The mean of the window, over its full area. */
                };

                /** @brief Holds what a pooling_param states, defaults filled in. */
                struct Pool
                {
                    PoolMethod method = PoolMethod::Max; /**< How a window is combined. */
                    bool global = false;                 /**< Whether the window is the image. */
                    WindowPair kernel;                   /**< Window extents. */
                    WindowPair stride{1, 1};             /**< Step of the window. */
                    WindowPair padding{0, 0};            /**< Pad on each side of an axis. */
                };

                /** @brief Holds one axis of a pool: the windows it yields and the pad it needs. */
                struct PoolAxis
                {
                    std::int64_t extent = 0;   /**< Windows along the axis. */
                    std::int64_t trailing = 0; /**< Pad after the input beyond the leading. */
                };

                /**
                 * @brief Reads the pooling_param of @p layer over its defaults.
                 * @return Unsupported for stochastic pooling; InvalidDescription for a value that
                 *         is malformed or out of range.
                 */
                [[nodiscard]] Result<Pool> read_pool(const Interchange::CaffeLayer& layer)
                {
                    const ParameterReader parameters(layer, "pooling_param");
                    Pool read;
                    SA_TRY_ASSIGN(const std::string method, parameters.read_word("pool", "MAX"));
                    SA_FAIL_IF(method != "MAX" && method != "AVE", StatusCode::Unsupported,
                               "{} states the pool method {}; MAX and AVE are translated",
                               describe_layer(layer), method);
                    read.method = method == "MAX" ? PoolMethod::Max : PoolMethod::Average;
                    SA_TRY_ASSIGN(read.global, parameters.read_bool("global_pooling", false));
                    if (read.global)
                        return read;

                    SA_TRY_ASSIGN(read.kernel, parameters.read_window("kernel_size", "kernel_h",
                                                                      "kernel_w", WindowPair{}));
                    SA_TRY_ASSIGN(read.stride, parameters.read_window("stride", "stride_h",
                                                                      "stride_w", read.stride));
                    SA_TRY_ASSIGN(read.padding, parameters.read_window("pad", "pad_h", "pad_w",
                                                                       read.padding));
                    SA_FAIL_IF(read.kernel.height < 1 || read.kernel.width < 1 ||
                                   read.stride.height < 1 || read.stride.width < 1 ||
                                   read.padding.height < 0 || read.padding.width < 0 ||
                                   read.padding.height >= read.kernel.height ||
                                   read.padding.width >= read.kernel.width,
                               StatusCode::InvalidDescription,
                               "{} needs a kernel and a stride of 1 or more and a pad below the "
                               "kernel",
                               describe_layer(layer));
                    return read;
                }

                /**
                 * @brief Returns the windows Caffe yields along one axis and the trailing pad a
                 *        pooling row needs to yield as many.
                 * @return InvalidDescription for a window wider than the padded input;
                 *         Unsupported when no trailing pad gives Caffe's count.
                 */
                [[nodiscard]] Result<PoolAxis> fit_axis(const Interchange::CaffeLayer& layer,
                                                        std::int64_t kernel, std::int64_t stride,
                                                        std::int64_t pad, std::int64_t input)
                {
                    SA_FAIL_IF(input + 2 * pad < kernel, StatusCode::InvalidDescription,
                               "{}: a window of {} does not fit the extent {} with pad {}",
                               describe_layer(layer), kernel, input, pad);
                    PoolAxis axis;
                    axis.extent = (input + 2 * pad - kernel + stride - 1) / stride + 1;
                    if (pad > 0 && (axis.extent - 1) * stride >= input + pad)
                        --axis.extent;

                    const std::int64_t covered = (axis.extent - 1) * stride + kernel;
                    axis.trailing = std::max<std::int64_t>(0, covered - input - 2 * pad);
                    const std::int64_t yielded =
                        (input + 2 * pad + axis.trailing - kernel) / stride + 1;
                    SA_FAIL_IF(yielded != axis.extent, StatusCode::Unsupported,
                               "{}: no pooling row gives the {} windows Caffe yields over the "
                               "extent {} with kernel {}, stride {} and pad {}",
                               describe_layer(layer), axis.extent, input, kernel, stride, pad);
                    return axis;
                }

                /** @brief Appends the one row of a pool that windows the image. */
                [[nodiscard]] Status append_window(const Interchange::CaffeLayer& layer,
                                                   const Pool& read, const BlobValue& source,
                                                   std::int64_t index, TranslationProgress& progress)
                {
                    SA_TRY_ASSIGN(const PoolAxis rows,
                                  fit_axis(layer, read.kernel.height, read.stride.height,
                                           read.padding.height, source.height));
                    SA_TRY_ASSIGN(const PoolAxis columns,
                                  fit_axis(layer, read.kernel.width, read.stride.width,
                                           read.padding.width, source.width));
                    const bool is_trailing = rows.trailing != 0 || columns.trailing != 0;
                    SA_FAIL_IF(read.method == PoolMethod::Average && is_trailing,
                               StatusCode::Unsupported,
                               "{}: an average window reaches past the padded edge, where Caffe "
                               "divides by a smaller area than avgpool2d does",
                               describe_layer(layer));

                    std::vector<std::string> keys = {"kernel", "stride", "padding"};
                    std::vector<Config::Value> values = {
                        pair_value(read.kernel.height, read.kernel.width),
                        pair_value(read.stride.height, read.stride.width),
                        pair_value(read.padding.height, read.padding.width)};
                    if (is_trailing)
                    {
                        keys.emplace_back("trailing_padding");
                        values.push_back(Config::Value::array(
                            {Config::Value::integer(rows.trailing),
                             Config::Value::integer(columns.trailing)}));
                    }

                    const std::int64_t sources[] = {source.row};
                    BlobValue result = source;
                    result.row = append_row(
                        progress, read.method == PoolMethod::Max ? "maxpool2d" : "avgpool2d",
                        make_attributes(std::move(keys), std::move(values)), sources);
                    result.height = rows.extent;
                    result.width = columns.extent;
                    write_blob(progress, layer.tops.front(), result, index);
                    return Status();
                }

                /** @brief Appends the global_avgpool row of an average over the whole image. */
                [[nodiscard]] Status append_global(const Interchange::CaffeLayer& layer,
                                                   const Pool& read, const BlobValue& source,
                                                   std::int64_t index, TranslationProgress& progress)
                {
                    SA_FAIL_IF(read.method != PoolMethod::Average, StatusCode::Unsupported,
                               "{} takes the maximum of the whole image; only the average has a "
                               "global row",
                               describe_layer(layer));
                    const std::int64_t sources[] = {source.row};
                    BlobValue result;
                    result.row = append_row(progress, "global_avgpool", make_attributes({}, {}),
                                            sources);
                    result.channels = source.channels;
                    write_blob(progress, layer.tops.front(), result, index);
                    return Status();
                }
            } // namespace

            Status translate_pooling(const Interchange::CaffeLayer& layer, std::int64_t index,
                                     TranslationProgress& progress)
            {
                SA_TRY(expect_shape(layer, 1));
                SA_TRY_ASSIGN(const BlobValue source, read_bottom(progress, layer, 0));
                SA_TRY(expect_image(layer, source));
                SA_TRY_ASSIGN(const Pool read, read_pool(layer));
                if (read.global)
                    return append_global(layer, read, source, index, progress);
                return append_window(layer, read, source, index, progress);
            }
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
