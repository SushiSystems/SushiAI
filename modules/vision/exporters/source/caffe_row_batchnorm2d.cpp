/**************************************************************************/
/* caffe_row_batchnorm2d.cpp                                              */
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
 * @file caffe_row_batchnorm2d.cpp
 * @brief Defines the Caffe writer of a "batchnorm2d" row: a BatchNorm and a Scale in place.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
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
                constexpr std::string_view K_BATCHNORM2D_KEYS[] = {"bn_epsilon"};

                /** @brief Holds the epsilon of a row that states none, as BatchNorm2d has it. */
                constexpr double K_DEFAULT_EPSILON = 1e-5;

                /** @brief Returns the per-channel weight `<row>.<field>` as a blob [C]. */
                Result<Interchange::CaffeBlob> read_vector_blob(
                    const CaffeRowCall& call, const CaffeExportProgress& progress,
                    std::string_view field, std::int64_t channels)
                {
                    std::vector<std::int64_t> extents;
                    SA_TRY_ASSIGN(const std::vector<float> values,
                                  read_weight(*progress.model, call.index, field,
                                              K_CAFFE_FLOAT32_HOLDER, extents));
                    SA_FAIL_IF(extents.size() != 1 || extents[0] != channels,
                               StatusCode::InvalidDescription,
                               "row {} 'batchnorm2d' holds '{}' that is not one value for each of "
                               "{} channels",
                               call.index, row_tensor_name(call.index, field), channels);
                    return make_blob(std::move(extents), values);
                }
            } // namespace

            Status write_caffe_batchnorm2d(const CaffeRowCall& call, CaffeExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(
                    keys, SushiRuntime::span<const std::string_view>(
                              K_BATCHNORM2D_KEYS, std::size(K_BATCHNORM2D_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const CaffeRowValue& input = *call.sources[0];
                SA_TRY(check_image(input));
                SA_TRY_ASSIGN(const double epsilon,
                              read_real_key(keys, "bn_epsilon", K_DEFAULT_EPSILON));
                const std::int64_t channels = input.shape[2];
                std::vector<Interchange::CaffeBlob> norm_blobs;
                std::vector<Interchange::CaffeBlob> scale_blobs;
                if (progress.has_weights)
                {
                    SA_TRY_ASSIGN(Interchange::CaffeBlob mean,
                                  read_vector_blob(call, progress, "running_mean", channels));
                    SA_TRY_ASSIGN(Interchange::CaffeBlob variance,
                                  read_vector_blob(call, progress, "running_var", channels));
                    SA_TRY_ASSIGN(Interchange::CaffeBlob gamma,
                                  read_vector_blob(call, progress, "weight", channels));
                    SA_TRY_ASSIGN(Interchange::CaffeBlob beta,
                                  read_vector_blob(call, progress, "bias", channels));
                    norm_blobs.push_back(std::move(mean));
                    norm_blobs.push_back(std::move(variance));
                    norm_blobs.push_back(make_blob({1}, {1.0f}));
                    scale_blobs.push_back(std::move(gamma));
                    scale_blobs.push_back(std::move(beta));
                }
                add_layer(progress, call, "BatchNorm", {input.blob},
                          make_message("batch_norm_param", {make_real_field("eps", epsilon)}),
                          std::move(norm_blobs));
                const std::string& blob = progress.blob_names[call.index];
                add_layer(progress, call, "Scale", {blob},
                          make_message("scale_param", {make_word_field("bias_term", "true")}),
                          std::move(scale_blobs));
                progress.net.layers.back().name += "_scale";
                define_row_value(progress, call, input.shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
