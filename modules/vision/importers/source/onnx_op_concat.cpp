/**************************************************************************/
/* onnx_op_concat.cpp                                                     */
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
 * @file onnx_op_concat.cpp
 * @brief Defines the translation of an ONNX Concat along the channel axis into a concat row.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <utility>
#include <vector>
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
                /** @brief Lists every attribute the translation of Concat reads. */
                constexpr std::string_view K_CONCAT_ATTRIBUTES[] = {"axis"};

                /** @brief Holds the number of axes the batch adds to the rank of one sample. */
                constexpr std::int64_t K_BATCH_AXES = 1;

                /** @brief Returns whether @p left and @p right differ only at the channel axis. */
                bool agree_but_for_channels(const std::vector<std::int64_t>& left,
                                            const std::vector<std::int64_t>& right) noexcept
                {
                    if (left.size() != right.size())
                        return false;
                    for (std::size_t at = 1; at < left.size(); ++at)
                    {
                        if (left[at] != right[at])
                            return false;
                    }
                    return true;
                }
            } // namespace

            Status translate_onnx_concat(const Interchange::OnnxNode& node, OnnxProgress& progress)
            {
                SA_TRY(check_operand_count(node, 1, node.inputs.size()));
                SA_TRY(check_output_count(node, 1));
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>(
                              K_CONCAT_ATTRIBUTES, std::size(K_CONCAT_ATTRIBUTES))));
                const Interchange::OnnxAttribute* stated = node.find_attribute("axis");
                SA_FAIL_IF(stated == nullptr, StatusCode::InvalidDescription,
                           "{} states no axis", describe_node(node));
                SA_TRY_ASSIGN(const std::int64_t axis, read_int_attribute(node, "axis", 0));

                std::vector<const OnnxValue*> inputs;
                std::vector<std::int64_t> sources;
                std::int64_t channels = 0;
                for (std::size_t at = 0; at < node.inputs.size(); ++at)
                {
                    SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, at));
                    SA_FAIL_IF(!inputs.empty() &&
                                   !agree_but_for_channels(inputs.front()->dimensions,
                                                           input->dimensions),
                               StatusCode::InvalidDescription,
                               "{}: operand {} differs from operand 0 on an axis other than the "
                               "channels",
                               describe_node(node), at);
                    inputs.push_back(input);
                    sources.push_back(input->row);
                    channels += input->dimensions[0];
                }

                // A batch axis plus the sample's axes; the channels sit last once translated.
                const std::int64_t rank =
                    static_cast<std::int64_t>(inputs.front()->dimensions.size()) + K_BATCH_AXES;
                SA_TRY_ASSIGN(const std::int64_t moved,
                              Interchange::channels_last_axis(axis, rank));
                SA_FAIL_IF(moved != rank - 1, StatusCode::Unsupported,
                           "{} joins along axis {}, and only the channel axis is translated",
                           describe_node(node), axis);

                std::vector<std::int64_t> dimensions = inputs.front()->dimensions;
                dimensions[0] = channels;
                const std::int64_t row = append_row(progress, "concat", {}, {}, sources);
                define_value(progress, node.outputs[0], row, std::move(dimensions));
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
