/**************************************************************************/
/* onnx_row_linear.cpp                                                    */
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
 * @file onnx_row_linear.cpp
 * @brief Defines the writer of the "linear" row, one ONNX Gemm node with transB 1.
 * @author Mustafa Garip
 */

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
                constexpr std::string_view K_LINEAR_KEYS[] = {"units", "use_bias"};
            } // namespace

            Status write_onnx_linear(const OnnxRowCall& call, OnnxExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>(
                                                     K_LINEAR_KEYS, std::size(K_LINEAR_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const OnnxRowValue& input = *call.sources[0];
                SA_FAIL_IF(input.shape.size() != 1, StatusCode::Unsupported,
                           "it reads '{}' of rank {}, and Gemm reads a vector", input.name,
                           input.shape.size());
                std::vector<std::int64_t> extents;
                SA_TRY_ASSIGN(std::vector<float> weight,
                              read_weight(*progress.model, call.index, "weight",
                                          K_ONNX_FLOAT32_HOLDER, extents));
                SA_FAIL_IF(extents.size() != 2 || extents[1] != input.shape[0],
                           StatusCode::InvalidDescription,
                           "row {} 'linear' holds a weight that does not read {} features",
                           call.index, input.shape[0]);
                if (!input.flattened_from.empty())
                {
                    const std::vector<std::int64_t>& image = input.flattened_from;
                    SA_TRY_ASSIGN(weight, Interchange::flatten_weights_to_channels_first(
                                              SushiRuntime::span<const float>(weight.data(),
                                                                              weight.size()),
                                              extents[0], image[2], image[0], image[1]));
                }
                const std::string weight_name = row_tensor_name(call.index, "weight");
                add_float_initializer(progress, weight_name, extents, weight);
                std::vector<std::string> inputs = {input.name, weight_name};
                SA_TRY_ASSIGN(const bool use_bias, read_bool_key(keys, "use_bias", true));
                const bool has_bias = find_weight(*progress.model, call.index, "bias") != nullptr;
                SA_FAIL_IF(use_bias != has_bias, StatusCode::InvalidDescription,
                           "row {} 'linear' states use_bias {}, and the weights disagree",
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
                add_node(progress, call, "Gemm", "node", std::move(inputs),
                         {row_output(progress, call)}, {make_int_attribute("transB", 1)});
                define_row_value(progress, call, {extents[0]});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
