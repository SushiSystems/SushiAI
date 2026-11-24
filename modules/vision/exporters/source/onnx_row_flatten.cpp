/**************************************************************************/
/* onnx_row_flatten.cpp                                                   */
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
 * @file onnx_row_flatten.cpp
 * @brief Defines the writer of the "flatten" row, one ONNX Flatten node on axis 1.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include "onnx_export_progress.hpp"
#include "onnx_row_table.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            Status write_onnx_flatten(const OnnxRowCall& call, OnnxExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>()));
                SA_TRY(check_source_count(call, 1));
                const OnnxRowValue& input = *call.sources[0];
                SA_TRY(check_image(input));
                SA_FAIL_IF(progress.published[call.index], StatusCode::Unsupported,
                           "the model publishes it, and ONNX orders its features (C, H, W) "
                           "where the row orders them (H, W, C)");
                for (const std::size_t reader : progress.readers[call.index])
                {
                    SA_FAIL_IF(progress.model->model.layers[reader].type != "linear",
                               StatusCode::Unsupported,
                               "row {} '{}' reads it, and ONNX orders its features (C, H, W) "
                               "where the row orders them (H, W, C); only a linear row reads it",
                               reader, progress.model->model.layers[reader].type);
                }
                add_node(progress, call, "Flatten", "node", {input.name},
                         {row_output(progress, call)}, {make_int_attribute("axis", 1)});
                define_row_value(progress, call,
                                 {input.shape[0] * input.shape[1] * input.shape[2]});
                progress.values[call.index]->flattened_from = input.shape;
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
