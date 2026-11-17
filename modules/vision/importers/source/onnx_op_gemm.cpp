/**************************************************************************/
/* onnx_op_gemm.cpp                                                       */
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
 * @file onnx_op_gemm.cpp
 * @brief Defines the translation of an ONNX Gemm with a constant weight into a linear row.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
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
                /** @brief Lists every attribute the translation of Gemm reads. */
                constexpr std::string_view K_GEMM_ATTRIBUTES[] = {"alpha", "beta", "transA",
                                                                  "transB"};

                /** @brief Holds the rank of the weight and of a matrix bias. */
                constexpr std::size_t K_MATRIX_RANK = 2;

                /** @brief Returns the weight [out, in] of the B operand of a Gemm. */
                std::vector<float> weight_out_in(const OnnxConstant& operand, bool transposed)
                {
                    if (transposed)
                        return operand.reals;
                    const std::size_t inputs = static_cast<std::size_t>(operand.dimensions[0]);
                    const std::size_t outputs = static_cast<std::size_t>(operand.dimensions[1]);
                    std::vector<float> weight(operand.reals.size());
                    for (std::size_t input = 0; input < inputs; ++input)
                    {
                        for (std::size_t output = 0; output < outputs; ++output)
                            weight[output * inputs + input] =
                                operand.reals[input * outputs + output];
                    }
                    return weight;
                }
            } // namespace

            Status translate_onnx_gemm(const Interchange::OnnxNode& node, OnnxProgress& progress)
            {
                SA_TRY(check_operand_count(node, 2, 3));
                SA_TRY(check_output_count(node, 1));
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>(
                              K_GEMM_ATTRIBUTES, std::size(K_GEMM_ATTRIBUTES))));
                SA_TRY_ASSIGN(const float alpha, read_real_attribute(node, "alpha", 1.0f));
                SA_TRY_ASSIGN(const float beta, read_real_attribute(node, "beta", 1.0f));
                SA_TRY_ASSIGN(const std::int64_t transposed_a,
                              read_int_attribute(node, "transA", 0));
                SA_TRY_ASSIGN(const std::int64_t transposed_b,
                              read_int_attribute(node, "transB", 0));
                SA_FAIL_IF(alpha != 1.0f || beta != 1.0f || transposed_a != 0,
                           StatusCode::Unsupported,
                           "{} states alpha {}, beta {} and transA {}, and only 1, 1 and 0 are "
                           "translated",
                           describe_node(node), alpha, beta, transposed_a);

                SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, 0));
                SA_TRY_ASSIGN(const OnnxConstant* weight,
                              find_real_operand(progress, node, 1, K_MATRIX_RANK));
                const std::vector<std::int64_t> extents =
                    transposed_b != 0 ? weight->dimensions
                                      : std::vector<std::int64_t>{weight->dimensions[1],
                                                                  weight->dimensions[0]};

                const std::vector<float>* bias = nullptr;
                if (has_operand(node, 2))
                {
                    SA_TRY_ASSIGN(const OnnxConstant* stated,
                                  find_operand_constant(progress, node, 2));
                    SA_FAIL_IF(!stated->is_real || stated->dimensions.empty() ||
                                   stated->dimensions.size() > K_MATRIX_RANK,
                               StatusCode::Unsupported,
                               "{} adds a bias that is not a float vector", describe_node(node));
                    bias = &stated->reals;
                }
                return append_linear(progress, node, *input, extents,
                                     weight_out_in(*weight, transposed_b != 0), bias);
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
