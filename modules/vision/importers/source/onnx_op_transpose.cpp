/**************************************************************************/
/* onnx_op_transpose.cpp                                                  */
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
 * @file onnx_op_transpose.cpp
 * @brief Defines the translation of an ONNX Transpose of an image into a transpose row.
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
                /** @brief Lists every attribute the translation of Transpose reads. */
                constexpr std::string_view K_TRANSPOSE_ATTRIBUTES[] = {"perm"};

                /** @brief Holds the rank of a tensor of one image, [N, C, H, W]. */
                constexpr std::size_t K_IMAGE_TENSOR_RANK = 4;

                /** @brief Returns the permutation of an image tensor that moves no axis. */
                std::vector<std::int64_t> unmoved()
                {
                    return {0, 1, 2, 3};
                }

                /** @brief Returns the permutation that reverses @p rank axes, a Transpose's default. */
                std::vector<std::int64_t> reversed(std::size_t rank)
                {
                    std::vector<std::int64_t> permutation(rank);
                    for (std::size_t axis = 0; axis < rank; ++axis)
                        permutation[axis] = static_cast<std::int64_t>(rank - 1 - axis);

                    return permutation;
                }
            } // namespace

            Status translate_onnx_transpose(const Interchange::OnnxNode& node,
                                            OnnxProgress& progress)
            {
                SA_TRY(check_operand_count(node, 1, 1));
                SA_TRY(check_output_count(node, 1));
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>(
                              K_TRANSPOSE_ATTRIBUTES, std::size(K_TRANSPOSE_ATTRIBUTES))));

                SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, 0));
                SA_FAIL_IF(input->dimensions.size() + 1 != K_IMAGE_TENSOR_RANK,
                           StatusCode::Unsupported,
                           "{} reads a value of rank {}, and a transpose reads [C, H, W]",
                           describe_node(node), input->dimensions.size());

                SA_TRY_ASSIGN(std::vector<std::int64_t> permutation,
                              read_ints_attribute(node, "perm"));
                if (permutation.empty())
                    permutation = reversed(K_IMAGE_TENSOR_RANK);
                SA_FAIL_IF(permutation.size() != K_IMAGE_TENSOR_RANK, StatusCode::InvalidDescription,
                           "{} lists {} axes in 'perm' for a tensor of rank {}", describe_node(node),
                           permutation.size(), K_IMAGE_TENSOR_RANK);
                SA_FAIL_IF(permutation.front() != 0, StatusCode::Unsupported,
                           "{} moves the batch axis, which a row cannot state", describe_node(node));

                // The permutation of the channel-first tensor, then of the channel-last one.
                const Result<std::vector<std::int64_t>> placed =
                    Interchange::channels_last_permutation(permutation);
                SA_TRY(SA_ANNOTATE(placed.status(), "{}", describe_node(node)));
                const std::vector<std::int64_t>& stored = placed.value();

                std::vector<std::int64_t> dimensions;
                std::vector<Config::Value> axes;
                for (std::size_t axis = 1; axis < K_IMAGE_TENSOR_RANK; ++axis)
                {
                    dimensions.push_back(
                        input->dimensions[static_cast<std::size_t>(permutation[axis] - 1)]);
                    axes.push_back(Config::Value::integer(stored[axis] - 1));
                }

                // A permutation that moves no axis is an alias, as an Identity is.
                if (permutation == unmoved())
                {
                    define_value(progress, node.outputs[0], input->row, dimensions);
                    return Status();
                }

                const std::int64_t row =
                    append_row(progress, "transpose", {"perm"},
                               {Config::Value::array(std::move(axes))}, {input->row});
                define_value(progress, node.outputs[0], row, std::move(dimensions));
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
