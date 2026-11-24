/**************************************************************************/
/* onnx_row_batchnorm2d.cpp                                               */
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
 * @file onnx_row_batchnorm2d.cpp
 * @brief Defines the writer of the "batchnorm2d" row, one ONNX BatchNormalization node.
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
                constexpr std::string_view K_BATCHNORM2D_KEYS[] = {"bn_epsilon"};

                /** @brief Holds the epsilon of a row that states none. */
                constexpr double K_DEFAULT_EPSILON = 1e-5;

                /** @brief Names the four vectors, in the operand order of the node. */
                constexpr std::string_view K_FIELDS[] = {"weight", "bias", "running_mean",
                                                         "running_var"};
            } // namespace

            Status write_onnx_batchnorm2d(const OnnxRowCall& call, OnnxExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(
                    keys, SushiRuntime::span<const std::string_view>(
                              K_BATCHNORM2D_KEYS, std::size(K_BATCHNORM2D_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const OnnxRowValue& input = *call.sources[0];
                SA_TRY(check_image(input));
                SA_TRY_ASSIGN(const double epsilon,
                              read_real_key(keys, "bn_epsilon", K_DEFAULT_EPSILON));
                std::vector<std::string> inputs = {input.name};
                for (const std::string_view field : K_FIELDS)
                {
                    std::vector<std::int64_t> extents;
                    SA_TRY_ASSIGN(const std::vector<float> values,
                                  read_weight(*progress.model, call.index, field,
                                              K_ONNX_FLOAT32_HOLDER, extents));
                    SA_FAIL_IF(extents.size() != 1 || extents[0] != input.shape[2],
                               StatusCode::InvalidDescription,
                               "row {} 'batchnorm2d' holds a '{}' that is not one value per "
                               "channel",
                               call.index, field);
                    inputs.push_back(row_tensor_name(call.index, field));
                    add_float_initializer(progress, inputs.back(), extents, values);
                }
                add_node(progress, call, "BatchNormalization", "node", std::move(inputs),
                         {row_output(progress, call)},
                         {make_real_attribute("epsilon", static_cast<float>(epsilon))});
                define_row_value(progress, call, input.shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
