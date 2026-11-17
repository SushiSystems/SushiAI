/**************************************************************************/
/* onnx_op_reshape.cpp                                                    */
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
 * @file onnx_op_reshape.cpp
 * @brief Defines the translation of an ONNX Reshape into a flatten row, or into no row.
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
                /** @brief Lists every attribute the translation of Reshape reads. */
                constexpr std::string_view K_RESHAPE_ATTRIBUTES[] = {"allowzero"};

                /** @brief Holds the rank of one sample of an image, [C, H, W]. */
                constexpr std::size_t K_IMAGE_RANK = 3;

                /** @brief Returns the product of the extents of @p dimensions. */
                std::int64_t count_elements(const std::vector<std::int64_t>& dimensions) noexcept
                {
                    std::int64_t count = 1;
                    for (const std::int64_t extent : dimensions)
                        count *= extent;
                    return count;
                }

                /**
                 * @brief Returns the extents of one sample after the reshape, 0 and -1 resolved.
                 * @param target The shape operand, batch extent first.
                 * @return InvalidDescription for a shape that does not hold the sample's elements.
                 */
                Result<std::vector<std::int64_t>> resolve_sample(
                    const Interchange::OnnxNode& node, const OnnxValue& input,
                    const std::vector<std::int64_t>& target)
                {
                    SA_FAIL_IF(target.size() < 2, StatusCode::Unsupported,
                               "{} reshapes to rank {}, and a sample keeps at least one axis",
                               describe_node(node), target.size());
                    SA_FAIL_IF(target[0] != 0 && target[0] != 1 && target[0] != -1,
                               StatusCode::Unsupported,
                               "{} reshapes the batch axis to {}, and the batch is kept",
                               describe_node(node), target[0]);

                    std::vector<std::int64_t> sample(target.begin() + 1, target.end());
                    std::size_t inferred = sample.size();
                    std::int64_t known = 1;
                    for (std::size_t at = 0; at < sample.size(); ++at)
                    {
                        if (sample[at] == 0)
                        {
                            SA_FAIL_IF(at >= input.dimensions.size(),
                                       StatusCode::InvalidDescription,
                                       "{}: a 0 at axis {} copies an axis the value lacks",
                                       describe_node(node), at + 1);
                            sample[at] = input.dimensions[at];
                        }
                        if (sample[at] == -1)
                        {
                            SA_FAIL_IF(inferred != sample.size(), StatusCode::InvalidDescription,
                                       "{} states -1 twice", describe_node(node));
                            inferred = at;
                            continue;
                        }
                        SA_FAIL_IF(sample[at] < 1, StatusCode::InvalidDescription,
                                   "{} states the extent {}", describe_node(node), sample[at]);
                        known *= sample[at];
                    }

                    const std::int64_t total = count_elements(input.dimensions);
                    if (inferred != sample.size())
                        sample[inferred] = total / known;
                    SA_FAIL_IF(count_elements(sample) != total, StatusCode::InvalidDescription,
                               "{} reshapes {} elements of a sample into {}", describe_node(node),
                               total, count_elements(sample));
                    return sample;
                }
            } // namespace

            Status translate_onnx_reshape(const Interchange::OnnxNode& node, OnnxProgress& progress)
            {
                SA_TRY(check_operand_count(node, 2, 2));
                SA_TRY(check_output_count(node, 1));
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>(
                              K_RESHAPE_ATTRIBUTES, std::size(K_RESHAPE_ATTRIBUTES))));
                SA_TRY_ASSIGN(const std::int64_t allow_zero,
                              read_int_attribute(node, "allowzero", 0));
                SA_FAIL_IF(allow_zero != 0, StatusCode::Unsupported, "{} states allowzero {}",
                           describe_node(node), allow_zero);

                SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, 0));
                SA_TRY_ASSIGN(const OnnxConstant* shape, find_operand_constant(progress, node, 1));
                SA_FAIL_IF(shape->is_real || shape->dimensions.size() != 1,
                           StatusCode::InvalidDescription,
                           "{}: the shape operand is not a list of integers", describe_node(node));
                SA_TRY_ASSIGN(const std::vector<std::int64_t> sample,
                              resolve_sample(node, *input, shape->integers));

                if (sample == input->dimensions)
                {
                    define_value(progress, node.outputs[0], input->row, input->dimensions,
                                 input->flattened_from);
                    return Status();
                }
                SA_FAIL_IF(sample.size() != 1 || input->dimensions.size() != K_IMAGE_RANK,
                           StatusCode::Unsupported,
                           "{} reshapes a sample of {} axes into {} axes, and only an image into "
                           "one vector is translated",
                           describe_node(node), input->dimensions.size(), sample.size());
                append_flatten(progress, node.outputs[0], *input);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
