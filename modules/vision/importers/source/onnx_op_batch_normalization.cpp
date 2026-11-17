/**************************************************************************/
/* onnx_op_batch_normalization.cpp                                        */
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
 * @file onnx_op_batch_normalization.cpp
 * @brief Defines the translation of an ONNX BatchNormalization into a batchnorm2d row.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
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
                /** @brief Lists every attribute the translation of BatchNormalization reads. */
                constexpr std::string_view K_BATCH_NORMALIZATION_ATTRIBUTES[] = {
                    "epsilon", "momentum", "training_mode"};

                /** @brief Holds the epsilon ONNX gives a node that states none. */
                constexpr float K_DEFAULT_EPSILON = 1e-5f;

                /** @brief Holds the rank of one sample of an image, [C, H, W]. */
                constexpr std::size_t K_IMAGE_RANK = 3;

                /** @brief Names the four vectors of the row, in the operand order of the node. */
                constexpr std::string_view K_FIELDS[] = {"weight", "bias", "running_mean",
                                                         "running_var"};
            } // namespace

            Status translate_onnx_batch_normalization(const Interchange::OnnxNode& node,
                                                      OnnxProgress& progress)
            {
                SA_TRY(check_operand_count(node, 5, 5));
                SA_TRY(check_output_count(node, 1));
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>(
                              K_BATCH_NORMALIZATION_ATTRIBUTES,
                              std::size(K_BATCH_NORMALIZATION_ATTRIBUTES))));
                SA_TRY_ASSIGN(const std::int64_t training,
                              read_int_attribute(node, "training_mode", 0));
                SA_FAIL_IF(training != 0, StatusCode::Unsupported,
                           "{} states training_mode {}, and only inference is translated",
                           describe_node(node), training);
                SA_TRY_ASSIGN(const float epsilon,
                              read_real_attribute(node, "epsilon", K_DEFAULT_EPSILON));

                SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, 0));
                SA_FAIL_IF(input->dimensions.size() != K_IMAGE_RANK, StatusCode::Unsupported,
                           "{} reads a value of rank {}, and the batchnorm2d row reads [C, H, W]",
                           describe_node(node), input->dimensions.size());
                const std::int64_t channels = input->dimensions[0];

                const OnnxConstant* vectors[4] = {};
                for (std::size_t at = 0; at < 4; ++at)
                {
                    SA_TRY_ASSIGN(const OnnxConstant* stated,
                                  find_real_operand(progress, node, at + 1, 1));
                    SA_FAIL_IF(stated->dimensions[0] != channels, StatusCode::InvalidDescription,
                               "{}: operand {} holds {} values for {} channels",
                               describe_node(node),
                               at + 1, stated->dimensions[0], channels);
                    vectors[at] = stated;
                }

                const std::int64_t row = append_row(
                    progress, "batchnorm2d", {"bn_epsilon"},
                    {Config::Value::real(static_cast<double>(epsilon))}, {input->row});
                for (std::size_t at = 0; at < 4; ++at)
                    SA_TRY(store_weight(progress, row, K_FIELDS[at], {channels},
                             vectors[at]->reals));
                define_value(progress, node.outputs[0], row, input->dimensions);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
