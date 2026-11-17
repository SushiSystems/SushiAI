/**************************************************************************/
/* onnx_op_constant.cpp                                                   */
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
 * @file onnx_op_constant.cpp
 * @brief Defines the translation of an ONNX Constant into a constant its readers take.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <utility>
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
                /** @brief Lists every attribute the translation of Constant reads. */
                constexpr std::string_view K_CONSTANT_ATTRIBUTES[] = {
                    "value", "value_float", "value_floats", "value_int", "value_ints"};
            } // namespace

            Status translate_onnx_constant(const Interchange::OnnxNode& node,
                                                OnnxProgress& progress)
            {
                SA_FAIL_IF(!node.inputs.empty(), StatusCode::InvalidDescription,
                           "{} reads operands, and a Constant reads none", describe_node(node));
                SA_TRY(check_output_count(node, 1));
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>(
                              K_CONSTANT_ATTRIBUTES, std::size(K_CONSTANT_ATTRIBUTES))));
                SA_FAIL_IF(node.attributes.size() != 1, StatusCode::InvalidDescription,
                           "{} states {} values, and a Constant states one", describe_node(node),
                           node.attributes.size());

                const Interchange::OnnxAttribute& stated = node.attributes.front();
                OnnxConstant constant;
                switch (stated.type)
                {
                    case Interchange::OnnxAttributeType::Tensor:
                    {
                        SA_TRY_ASSIGN(OnnxConstant read, read_constant_tensor(stated.tensor));
                        constant = std::move(read);
                        break;
                    }
                    case Interchange::OnnxAttributeType::Float:
                        constant.is_real = true;
                        constant.reals = {stated.real};
                        break;
                    case Interchange::OnnxAttributeType::Floats:
                        constant.is_real = true;
                        constant.reals = stated.reals;
                        constant.dimensions = {static_cast<std::int64_t>(stated.reals.size())};
                        break;
                    case Interchange::OnnxAttributeType::Int:
                        constant.integers = {stated.integer};
                        break;
                    case Interchange::OnnxAttributeType::Ints:
                        constant.integers = stated.integers;
                        constant.dimensions = {static_cast<std::int64_t>(stated.integers.size())};
                        break;
                    default:
                        return Status::from_message(StatusCode::InvalidDescription, [&]()
                        {
                            return describe_node(node) + " states '" + stated.name +
                                   "' with a value of a kind a Constant does not hold";
                        });
                }
                define_constant(progress, node.outputs[0], std::move(constant));
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
