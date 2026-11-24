/**************************************************************************/
/* onnx_row_upsample.cpp                                                  */
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
 * @file onnx_row_upsample.cpp
 * @brief Defines the writer of the "upsample" row, one nearest ONNX Resize node.
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
                constexpr std::string_view K_UPSAMPLE_KEYS[] = {"scale", "mode"};

                /** @brief Holds the factor of a row that states none. */
                constexpr std::int64_t K_DEFAULT_SCALE = 2;

                /** @brief Names the one mode the ONNX importer reads back. */
                constexpr std::string_view K_NEAREST = "nearest";
            } // namespace

            Status write_onnx_upsample(const OnnxRowCall& call, OnnxExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>(
                                                     K_UPSAMPLE_KEYS, std::size(K_UPSAMPLE_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const OnnxRowValue& input = *call.sources[0];
                SA_TRY(check_image(input));
                SA_TRY_ASSIGN(const std::string mode, read_text_key(keys, "mode", K_NEAREST));
                SA_FAIL_IF(mode != K_NEAREST, StatusCode::Unsupported,
                           "it states mode '{}', and the ONNX importer reads a nearest Resize "
                           "only",
                           mode);
                SA_TRY_ASSIGN(const std::int64_t scale,
                              read_int_key(keys, "scale", K_DEFAULT_SCALE));
                SA_FAIL_IF(scale < 1, StatusCode::InvalidDescription,
                           "row {} 'upsample' states scale {}", call.index, scale);
                const float factor = static_cast<float>(scale);
                const std::string scales = row_tensor_name(call.index, "scales");
                add_float_initializer(progress, scales, {4}, {1.0f, 1.0f, factor, factor});
                add_node(progress, call, "Resize", "node", {input.name, "", scales},
                         {row_output(progress, call)},
                         {make_text_attribute("mode", std::string(K_NEAREST)),
                          make_text_attribute("coordinate_transformation_mode", "asymmetric"),
                          make_text_attribute("nearest_mode", "floor")});
                define_row_value(progress, call,
                                 {input.shape[0] * scale, input.shape[1] * scale, input.shape[2]});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
