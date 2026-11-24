/**************************************************************************/
/* caffe_row_linear.cpp                                                   */
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
 * @file caffe_row_linear.cpp
 * @brief Defines the Caffe writer of a "linear" row: one InnerProduct layer.
 * @author Mustafa Garip
 */

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
                constexpr std::string_view K_LINEAR_KEYS[] = {"units", "use_bias"};

                /** @brief Returns the weight blob, columns in Caffe's order, and the bias blob. */
                Result<std::vector<Interchange::CaffeBlob>> read_blobs(
                    const CaffeRowCall& call, const CaffeExportProgress& progress,
                    const CaffeRowValue& input, std::int64_t units, bool use_bias)
                {
                    std::vector<Interchange::CaffeBlob> blobs;
                    if (!progress.has_weights)
                        return blobs;
                    std::vector<std::int64_t> extents;
                    SA_TRY_ASSIGN(std::vector<float> weight,
                                  read_weight(*progress.model, call.index, "weight",
                                              K_CAFFE_FLOAT32_HOLDER, extents));
                    SA_FAIL_IF(extents.size() != 2 || extents[0] != units ||
                                   extents[1] != input.shape[0],
                               StatusCode::InvalidDescription,
                               "row {} 'linear' holds a weight that does not map {} features to "
                               "{} units",
                               call.index, input.shape[0], units);
                    if (!input.flattened_from.empty())
                    {
                        const std::vector<std::int64_t>& image = input.flattened_from;
                        SA_TRY_ASSIGN(weight, Interchange::flatten_weights_to_channels_first(
                                                  SushiRuntime::span<const float>(weight.data(),
                                                                                  weight.size()),
                                                  units, image[2], image[0], image[1]));
                    }
                    blobs.push_back(make_blob(std::move(extents), weight));
                    if (use_bias)
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

            Status write_caffe_linear(const CaffeRowCall& call, CaffeExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>(
                                                     K_LINEAR_KEYS, std::size(K_LINEAR_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const CaffeRowValue& input = *call.sources[0];
                SA_FAIL_IF(input.shape.size() != 1, StatusCode::Unsupported,
                           "it reads '{}' of rank {}, and InnerProduct reads a vector or a "
                           "flatten",
                           input.blob, input.shape.size());
                SA_TRY_ASSIGN(const std::int64_t units, read_int_key(keys, "units", 0));
                SA_TRY_ASSIGN(const bool use_bias, read_bool_key(keys, "use_bias", true));
                SA_TRY(check_bias(call, progress, use_bias));
                SA_TRY_ASSIGN(std::vector<Interchange::CaffeBlob> blobs,
                              read_blobs(call, progress, input, units, use_bias));
                add_layer(progress, call, "InnerProduct", {input.blob},
                          make_message("inner_product_param",
                                       {make_int_field("num_output", units),
                                        make_word_field("bias_term",
                                                        use_bias ? "true" : "false")}),
                          std::move(blobs));
                define_row_value(progress, call, {units});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
