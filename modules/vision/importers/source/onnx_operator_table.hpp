/**************************************************************************/
/* onnx_operator_table.hpp                                                */
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
 * @file onnx_operator_table.hpp
 * @brief Declares the table of ONNX operator translators and one translator per operator.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>
#include "onnx_translation.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            /**
             * @brief Translates one node: appends its rows and weights and defines its outputs.
             * @return Success once every output of the node is defined or a constant; Unsupported
             *         for a form of the operator the rows cannot say, which names the node.
             */
            using OnnxTranslator = Status (*)(const Interchange::OnnxNode& node,
                                              OnnxProgress& progress);

            /** @brief Holds one row of the table: an operator and its translator. */
            struct OnnxOperatorRow
            {
                std::string_view op_type; /**< The operator name the file spells. */
                OnnxTranslator translate; /**< The translator of that operator. */
            };

            /** @brief Returns the row of @p op_type, or null when the table holds none. */
            [[nodiscard]] const OnnxOperatorRow* find_onnx_operator(
                std::string_view op_type) noexcept;

            /** @brief Translates Conv, grouped or depthwise included, into a conv2d row and its weights. */
            [[nodiscard]] Status translate_onnx_conv(const Interchange::OnnxNode& node,
                                                     OnnxProgress& progress);

            /** @brief Translates BatchNormalization into a batchnorm2d row and its four vectors. */
            [[nodiscard]] Status translate_onnx_batch_normalization(
                const Interchange::OnnxNode& node, OnnxProgress& progress);

            /** @brief Translates Relu into a relu row. */
            [[nodiscard]] Status translate_onnx_relu(const Interchange::OnnxNode& node,
                                                     OnnxProgress& progress);

            /** @brief Translates Sigmoid into a sigmoid row, or into nothing as a silu's gate. */
            [[nodiscard]] Status translate_onnx_sigmoid(const Interchange::OnnxNode& node,
                                                        OnnxProgress& progress);

            /** @brief Translates the Mul of a SiLU into a silu row. */
            [[nodiscard]] Status translate_onnx_mul(const Interchange::OnnxNode& node,
                                                    OnnxProgress& progress);

            /** @brief Translates Add of two computed tensors into an add row. */
            [[nodiscard]] Status translate_onnx_add(const Interchange::OnnxNode& node,
                                                    OnnxProgress& progress);

            /** @brief Translates Sub of two computed tensors into a sub row. */
            [[nodiscard]] Status translate_onnx_sub(const Interchange::OnnxNode& node,
                                                    OnnxProgress& progress);

            /** @brief Translates Div of two computed tensors into a div row. */
            [[nodiscard]] Status translate_onnx_div(const Interchange::OnnxNode& node,
                                                    OnnxProgress& progress);

            /** @brief Translates Softmax into a softmax row. */
            [[nodiscard]] Status translate_onnx_softmax(const Interchange::OnnxNode& node,
                                                        OnnxProgress& progress);

            /** @brief Translates Transpose of an image into a transpose row. */
            [[nodiscard]] Status translate_onnx_transpose(const Interchange::OnnxNode& node,
                                                          OnnxProgress& progress);

            /** @brief Translates Concat along the channel axis into a concat row. */
            [[nodiscard]] Status translate_onnx_concat(const Interchange::OnnxNode& node,
                                                       OnnxProgress& progress);

            /** @brief Translates MaxPool into a maxpool2d row, with unequal pads as leading and end. */
            [[nodiscard]] Status translate_onnx_max_pool(const Interchange::OnnxNode& node,
                                                         OnnxProgress& progress);

            /** @brief Translates a nearest Resize by a whole factor into an upsample row. */
            [[nodiscard]] Status translate_onnx_resize(const Interchange::OnnxNode& node,
                                                       OnnxProgress& progress);

            /** @brief Translates Reshape into a flatten row or no row. */
            [[nodiscard]] Status translate_onnx_reshape(const Interchange::OnnxNode& node,
                                                        OnnxProgress& progress);

            /** @brief Translates Slice along the channel axis into a slice_channels row. */
            [[nodiscard]] Status translate_onnx_slice(const Interchange::OnnxNode& node,
                                                      OnnxProgress& progress);

            /** @brief Translates Split along the channel axis into one slice_channels row each. */
            [[nodiscard]] Status translate_onnx_split(const Interchange::OnnxNode& node,
                                                      OnnxProgress& progress);

            /** @brief Translates Flatten of an image into a flatten row. */
            [[nodiscard]] Status translate_onnx_flatten(const Interchange::OnnxNode& node,
                                                        OnnxProgress& progress);

            /** @brief Translates Gemm with a constant weight into a linear row. */
            [[nodiscard]] Status translate_onnx_gemm(const Interchange::OnnxNode& node,
                                                     OnnxProgress& progress);

            /** @brief Translates MatMul with a constant weight into a linear row. */
            [[nodiscard]] Status translate_onnx_mat_mul(const Interchange::OnnxNode& node,
                                                        OnnxProgress& progress);

            /** @brief Translates Identity into an alias of its operand. */
            [[nodiscard]] Status translate_onnx_identity(const Interchange::OnnxNode& node,
                                                         OnnxProgress& progress);

            /** @brief Translates Constant into a constant its readers take. */
            [[nodiscard]] Status translate_onnx_constant(const Interchange::OnnxNode& node,
                                                         OnnxProgress& progress);
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
