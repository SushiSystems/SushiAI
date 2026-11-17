/**************************************************************************/
/* onnx_operator_table.cpp                                                */
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
 * @file onnx_operator_table.cpp
 * @brief Defines the table of ONNX operator translators.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <string_view>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every operator the translation reads, one translator each. */
                constexpr OnnxOperatorRow K_OPERATOR_TABLE[] = {
                    {"Conv", &translate_onnx_conv},
                    {"BatchNormalization", &translate_onnx_batch_normalization},
                    {"Relu", &translate_onnx_relu},
                    {"Sigmoid", &translate_onnx_sigmoid},
                    {"Mul", &translate_onnx_mul},
                    {"Add", &translate_onnx_add},
                    {"Sub", &translate_onnx_sub},
                    {"Div", &translate_onnx_div},
                    {"Softmax", &translate_onnx_softmax},
                    {"Transpose", &translate_onnx_transpose},
                    {"Concat", &translate_onnx_concat},
                    {"MaxPool", &translate_onnx_max_pool},
                    {"Resize", &translate_onnx_resize},
                    {"Reshape", &translate_onnx_reshape},
                    {"Slice", &translate_onnx_slice},
                    {"Split", &translate_onnx_split},
                    {"Flatten", &translate_onnx_flatten},
                    {"Gemm", &translate_onnx_gemm},
                    {"MatMul", &translate_onnx_mat_mul},
                    {"Identity", &translate_onnx_identity},
                    {"Constant", &translate_onnx_constant},
                };
            } // namespace

            const OnnxOperatorRow* find_onnx_operator(std::string_view op_type) noexcept
            {
                for (const OnnxOperatorRow& row : K_OPERATOR_TABLE)
                {
                    if (row.op_type == op_type)
                        return &row;
                }
                return nullptr;
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
