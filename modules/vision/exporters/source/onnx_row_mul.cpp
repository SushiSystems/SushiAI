/**************************************************************************/
/* onnx_row_mul.cpp                                                       */
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
 * @file onnx_row_mul.cpp
 * @brief Defines the writer of the "mul" row, one ONNX Mul node.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <SushiAI/config/model_config.hpp>
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
                /**
                 * @brief Returns whether operand @p gate of the row is a sigmoid row that only
                 *        the row reads and whose input is operand @p other.
                 */
                bool is_silu_pair(const OnnxRowCall& call, const OnnxExportProgress& progress,
                                  std::size_t gate, std::size_t other)
                {
                    const Config::SourceSpec& gate_source = call.spec->from[gate];
                    if (gate_source.node < 0)
                        return false;
                    const std::size_t row = static_cast<std::size_t>(gate_source.node);
                    const Config::LayerSpec& gate_row = progress.model->model.layers[row];
                    if (gate_row.type != "sigmoid" || gate_row.from.size() != 1 ||
                        progress.readers[row].size() != 1 || progress.published[row])
                        return false;
                    return gate_row.from[0].node == call.spec->from[other].node;
                }
            } // namespace

            Status write_onnx_mul(const OnnxRowCall& call, OnnxExportProgress& progress)
            {
                SA_TRY(check_source_count(call, 2));
                SA_FAIL_IF(is_silu_pair(call, progress, 0, 1) || is_silu_pair(call, progress, 1, 0),
                           StatusCode::Unsupported,
                           "it multiplies a sigmoid row, read by it alone, by that row's input, "
                           "and the ONNX importer reads the pair back as one silu row");
                return write_binary_row(call, progress, "Mul");
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
