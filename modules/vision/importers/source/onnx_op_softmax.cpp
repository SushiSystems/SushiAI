/**************************************************************************/
/* onnx_op_softmax.cpp                                                    */
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
 * @file onnx_op_softmax.cpp
 * @brief Defines the translation of an ONNX Softmax into a softmax row.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
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
                /** @brief Lists every attribute the translation of Softmax reads. */
                constexpr std::string_view K_SOFTMAX_ATTRIBUTES[] = {"axis"};

                /** @brief Holds the axis a Softmax of opset 13 or later normalises when it states none. */
                constexpr std::int64_t K_DEFAULT_AXIS = -1;
            } // namespace

            Status translate_onnx_softmax(const Interchange::OnnxNode& node,
                                          OnnxProgress& progress)
            {
                SA_TRY(check_operand_count(node, 1, 1));
                SA_TRY(check_output_count(node, 1));
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>(
                              K_SOFTMAX_ATTRIBUTES, std::size(K_SOFTMAX_ATTRIBUTES))));
                SA_TRY_ASSIGN(const std::int64_t axis,
                              read_int_attribute(node, "axis", K_DEFAULT_AXIS));

                SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, 0));
                // The tensor is [N, ...dimensions]; a Softmax of opset 13 or later reads one axis.
                const std::int64_t rank = static_cast<std::int64_t>(input->dimensions.size()) + 1;
                const Result<std::int64_t> placed = Interchange::channels_last_axis(axis, rank);
                SA_TRY(SA_ANNOTATE(placed.status(), "{}", describe_node(node)));
                const std::int64_t moved = placed.value();
                SA_FAIL_IF(moved == 0, StatusCode::Unsupported,
                           "{} normalises over the batch axis, which a row cannot state",
                           describe_node(node));

                // The last axis of one sample is the row's default, so only another axis is stated.
                const std::int64_t sample_axis = moved - 1;
                std::vector<std::string> keys;
                std::vector<Config::Value> values;
                if (sample_axis != rank - 2)
                {
                    keys.emplace_back("axis");
                    values.push_back(Config::Value::integer(sample_axis));
                }

                const std::int64_t row =
                    append_row(progress, "softmax", std::move(keys), std::move(values),
                               {input->row});
                define_value(progress, node.outputs[0], row, input->dimensions,
                             input->flattened_from);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
