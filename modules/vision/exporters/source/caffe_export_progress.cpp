/**************************************************************************/
/* caffe_export_progress.cpp                                              */
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
 * @file caffe_export_progress.cpp
 * @brief Defines the helpers the Caffe row writers share.
 * @author Mustafa Garip
 */

#include "caffe_export_progress.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/interchange/tensor_bytes.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/protobuf_text.hpp>
#include "export_walk.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Holds the rank of one image sample, [H, W, C]. */
                constexpr std::size_t K_IMAGE_RANK = 3;

                /** @brief Holds room for the shortest text of any double. */
                constexpr std::size_t K_REAL_TEXT_SIZE = 32;

                /** @brief Returns a leaf field named @p name of @p kind holding @p text. */
                IO::TextField make_leaf(std::string_view name, IO::TextKind kind,
                                        std::string text)
                {
                    IO::TextField field;
                    field.name = std::string(name);
                    field.kind = kind;
                    field.text = std::move(text);
                    return field;
                }
            } // namespace

            RowKeys row_keys(const CaffeRowCall& call)
            {
                return RowKeys{call.spec, call.index, K_CAFFE_FORMAT};
            }

            Result<PoolWindow> read_pool_window(const CaffeRowCall& call)
            {
                const RowKeys keys = row_keys(call);
                PoolWindow window;
                SA_TRY_ASSIGN(window.kernel, read_pair_key(keys, "kernel", 1));
                window.stride = window.kernel;
                if (call.spec->attributes.has("stride"))
                {
                    SA_TRY_ASSIGN(window.stride, read_pair_key(keys, "stride", 1));
                }
                if (!call.spec->attributes.has("leading_padding"))
                {
                    SA_TRY_ASSIGN(window.padding, read_pair_key(keys, "padding", 0));
                    SA_TRY_ASSIGN(window.trailing, read_pair_key(keys, "trailing_padding", 0));
                    return window;
                }
                SA_TRY_ASSIGN(window.padding, read_pair_key(keys, "leading_padding", 0));
                SA_TRY_ASSIGN(const std::vector<std::int64_t> end,
                              read_pair_key(keys, "end_padding", 0));
                window.trailing = {end[0] - window.padding[0], end[1] - window.padding[1]};
                SA_FAIL_IF(window.trailing[0] < 0 || window.trailing[1] < 0,
                           StatusCode::Unsupported,
                           "it pads the far edge less than the near one, and Caffe pads both "
                           "edges alike before it rounds up");
                return window;
            }

            Result<CaffePoolAxis> fit_caffe_pool_axis(std::int64_t kernel, std::int64_t stride,
                                                      std::int64_t pad, std::int64_t input)
            {
                SA_FAIL_IF(kernel < 1 || stride < 1 || pad < 0 || pad >= kernel,
                           StatusCode::Unsupported,
                           "it states a kernel of {}, a stride of {} and a pad of {}, and Caffe "
                           "reads a pad below the kernel",
                           kernel, stride, pad);
                SA_FAIL_IF(input + 2 * pad < kernel, StatusCode::InvalidDescription,
                           "a window of {} does not fit the extent {} with pad {}", kernel, input,
                           pad);
                CaffePoolAxis axis;
                axis.extent = (input + 2 * pad - kernel + stride - 1) / stride + 1;
                if (pad > 0 && (axis.extent - 1) * stride >= input + pad)
                    --axis.extent;
                const std::int64_t covered = (axis.extent - 1) * stride + kernel;
                axis.trailing = std::max<std::int64_t>(0, covered - input - 2 * pad);
                const std::int64_t yielded =
                    (input + 2 * pad + axis.trailing - kernel) / stride + 1;
                SA_FAIL_IF(yielded != axis.extent, StatusCode::Unsupported,
                           "no pooling row gives the {} windows Caffe yields over the extent {}",
                           axis.extent, input);
                return axis;
            }

            Status check_source_count(const CaffeRowCall& call, std::size_t count)
            {
                return check_read_count(call.index, *call.spec, call.sources.size(), count);
            }

            Status check_image(const CaffeRowValue& value)
            {
                SA_FAIL_IF(value.shape.size() != K_IMAGE_RANK, StatusCode::Unsupported,
                           "it reads '{}' of rank {}, and its Caffe layer reads an image "
                           "[H, W, C]",
                           value.blob, value.shape.size());
                return Status();
            }

            Status check_bias(const CaffeRowCall& call, const CaffeExportProgress& progress,
                              bool use_bias)
            {
                if (!progress.has_weights)
                    return Status();
                const bool has_bias = find_weight(*progress.model, call.index, "bias") != nullptr;
                SA_FAIL_IF(use_bias != has_bias, StatusCode::InvalidDescription,
                           "row {} '{}' states use_bias {}, and the weights disagree", call.index,
                           call.spec->type, use_bias);
                return Status();
            }

            Interchange::CaffeBlob make_blob(std::vector<std::int64_t> extents,
                                             const std::vector<float>& values)
            {
                std::vector<std::byte> bytes(values.size() * sizeof(float));
                if (!bytes.empty())
                    std::memcpy(bytes.data(), values.data(), bytes.size());
                Interchange::CaffeBlob blob;
                blob.dimensions = std::move(extents);
                blob.data = Interchange::TensorBytes::owning(std::move(bytes));
                return blob;
            }

            IO::TextField make_int_field(std::string_view name, std::int64_t value)
            {
                return make_leaf(name, IO::TextKind::Number, std::to_string(value));
            }

            IO::TextField make_real_field(std::string_view name, double value)
            {
                char text[K_REAL_TEXT_SIZE];
                const std::to_chars_result written =
                    std::to_chars(text, text + K_REAL_TEXT_SIZE, value);
                const std::size_t length =
                    written.ec == std::errc() ? static_cast<std::size_t>(written.ptr - text) : 0;
                return make_leaf(name, IO::TextKind::Number, std::string(text, length));
            }

            IO::TextField make_word_field(std::string_view name, std::string_view word)
            {
                return make_leaf(name, IO::TextKind::Identifier, std::string(word));
            }

            IO::TextField make_message(std::string_view name, std::vector<IO::TextField> fields)
            {
                IO::TextField message;
                message.name = std::string(name);
                message.kind = IO::TextKind::Message;
                message.fields = std::move(fields);
                return message;
            }

            void add_layer(CaffeExportProgress& progress, const CaffeRowCall& call,
                           std::string_view type, std::vector<std::string> bottoms,
                           IO::TextField parameters, std::vector<Interchange::CaffeBlob> blobs)
            {
                Interchange::CaffeLayer layer;
                layer.name = "row_" + std::to_string(call.index);
                layer.type = std::string(type);
                layer.bottoms = std::move(bottoms);
                layer.tops = {progress.blob_names[call.index]};
                if (!parameters.name.empty())
                    layer.parameters.fields.push_back(std::move(parameters));
                if (progress.has_weights)
                    layer.blobs = std::move(blobs);
                progress.net.layers.push_back(std::move(layer));
            }

            void define_row_value(CaffeExportProgress& progress, const CaffeRowCall& call,
                                  std::vector<std::int64_t> shape)
            {
                progress.values[call.index] =
                    CaffeRowValue{progress.blob_names[call.index], std::move(shape), {}};
            }

            Status write_eltwise_row(const CaffeRowCall& call, CaffeExportProgress& progress,
                                     std::string_view operation)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>()));
                SA_TRY(check_source_count(call, 2));
                const CaffeRowValue& first = *call.sources[0];
                const CaffeRowValue& second = *call.sources[1];
                SA_TRY(check_image(first));
                SA_TRY(check_image(second));
                SA_FAIL_IF(first.shape != second.shape, StatusCode::Unsupported,
                           "its operands '{}' and '{}' differ in shape, and Eltwise combines "
                           "blobs of equal extents",
                           first.blob, second.blob);
                add_layer(progress, call, "Eltwise", {first.blob, second.blob},
                          make_message("eltwise_param",
                                       {make_word_field("operation", operation)}));
                define_row_value(progress, call, first.shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
