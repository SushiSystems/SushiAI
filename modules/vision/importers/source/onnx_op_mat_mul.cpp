/**************************************************************************/
/* onnx_op_mat_mul.cpp                                                    */
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
 * @file onnx_op_mat_mul.cpp
 * @brief Defines the translation of an ONNX MatMul with a constant weight into a linear row.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Holds the rank of the weight, [in, out]. */
                constexpr std::size_t K_MATRIX_RANK = 2;
            } // namespace

            Status translate_onnx_mat_mul(const Interchange::OnnxNode& node, OnnxProgress& progress)
            {
                SA_TRY(check_operand_count(node, 2, 2));
                SA_TRY(check_output_count(node, 1));
                SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, 0));
                SA_TRY_ASSIGN(const OnnxConstant* weight,
                              find_real_operand(progress, node, 1, K_MATRIX_RANK));

                const std::size_t inputs = static_cast<std::size_t>(weight->dimensions[0]);
                const std::size_t outputs = static_cast<std::size_t>(weight->dimensions[1]);
                std::vector<float> transposed(weight->reals.size());
                for (std::size_t feature = 0; feature < inputs; ++feature)
                {
                    for (std::size_t unit = 0; unit < outputs; ++unit)
                        transposed[unit * inputs + feature] =
                            weight->reals[feature * outputs + unit];
                }
                return append_linear(progress, node, *input,
                                     {weight->dimensions[1], weight->dimensions[0]}, transposed,
                                     nullptr);
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
