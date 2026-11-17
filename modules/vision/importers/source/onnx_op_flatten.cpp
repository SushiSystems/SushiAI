/**************************************************************************/
/* onnx_op_flatten.cpp                                                    */
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
 * @file onnx_op_flatten.cpp
 * @brief Defines the translation of an ONNX Flatten of an image into a flatten row.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/axis_translation.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every attribute the translation of Flatten reads. */
                constexpr std::string_view K_FLATTEN_ATTRIBUTES[] = {"axis"};

                /** @brief Holds the rank of one sample of an image, [C, H, W]. */
                constexpr std::size_t K_IMAGE_RANK = 3;

                /** @brief Holds the rank of the batch of images the axis is counted in. */
                constexpr std::int64_t K_BATCH_RANK = 4;

                /** @brief Holds the axis that leaves the batch alone on one side. */
                constexpr std::int64_t K_BATCH_SPLIT = 1;
            } // namespace

            Status translate_onnx_flatten(const Interchange::OnnxNode& node, OnnxProgress& progress)
            {
                SA_TRY(check_operand_count(node, 1, 1));
                SA_TRY(check_output_count(node, 1));
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>(
                              K_FLATTEN_ATTRIBUTES, std::size(K_FLATTEN_ATTRIBUTES))));
                SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, 0));
                SA_FAIL_IF(input->dimensions.size() != K_IMAGE_RANK, StatusCode::Unsupported,
                           "{} reads a value of rank {}, and a flatten reads [C, H, W]",
                           describe_node(node), input->dimensions.size());

                SA_TRY_ASSIGN(const std::int64_t axis, read_int_attribute(node, "axis", 1));
                SA_TRY_ASSIGN(const std::int64_t moved,
                              Interchange::channels_last_flatten_axis(axis, K_BATCH_RANK));
                SA_FAIL_IF(moved != K_BATCH_SPLIT, StatusCode::Unsupported,
                           "{} flattens at axis {}, and only the axis after the batch is "
                           "translated",
                           describe_node(node), axis);

                append_flatten(progress, node.outputs[0], *input);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
