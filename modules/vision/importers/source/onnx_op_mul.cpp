/**************************************************************************/
/* onnx_op_mul.cpp                                                        */
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
 * @file onnx_op_mul.cpp
 * @brief Defines the translation of an ONNX Mul: a SiLU into a silu row, any other product into a mul row.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            Status translate_onnx_mul(const Interchange::OnnxNode& node, OnnxProgress& progress)
            {
                if (progress.role != OnnxNodeRole::Silu)
                    return translate_binary_row(node, progress, "mul");

                SA_TRY(check_operand_count(node, 2, 2));
                SA_TRY(check_output_count(node, 1));
                SA_TRY_ASSIGN(const OnnxValue* source,
                              find_value(progress, node, progress.gate_source));
                const std::int64_t row = append_pointwise_row(progress, "silu", *source);
                define_value(progress, node.outputs[0], row, source->dimensions,
                             source->flattened_from);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
