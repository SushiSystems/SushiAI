/**************************************************************************/
/* onnx_op_resize.cpp                                                     */
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
 * @file onnx_op_resize.cpp
 * @brief Defines the translation of an ONNX nearest Resize by a whole factor into an upsample row.
 * @author Mustafa Garip
 */

#include "onnx_operator_table.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
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
                /** @brief Lists every attribute the translation of Resize reads. */
                constexpr std::string_view K_RESIZE_ATTRIBUTES[] = {
                    "mode", "coordinate_transformation_mode", "nearest_mode"};

                /** @brief Holds the first opset whose Resize takes roi, scales and sizes. */
                constexpr std::int64_t K_FIRST_OPSET = 11;

                /** @brief Holds the rank of one sample of an image, [C, H, W]. */
                constexpr std::size_t K_IMAGE_RANK = 3;

                /** @brief Holds the rank of the batch of images a scale or size lists. */
                constexpr std::size_t K_BATCH_RANK = 4;

                /** @brief Returns whether @p text is one of the @p count entries of @p allowed. */
                bool is_one_of(const std::string& text, const std::string_view* allowed,
                               std::size_t count) noexcept
                {
                    for (std::size_t at = 0; at < count; ++at)
                    {
                        if (allowed[at] == text)
                            return true;
                    }
                    return false;
                }

                /**
                 * @brief Returns the whole factor the operands of @p node scale the image by.
                 * @return Unsupported when the factor is not one whole number on both axes, or the
                 *         batch and the channels change.
                 */
                Result<std::int64_t> read_factor(OnnxProgress& progress,
                                                 const Interchange::OnnxNode& node,
                                                 const OnnxValue& input)
                {
                    SA_FAIL_IF(has_operand(node, 2) && has_operand(node, 3),
                               StatusCode::InvalidDescription,
                               "{} states both scales and sizes", describe_node(node));
                    if (has_operand(node, 2))
                    {
                        SA_TRY_ASSIGN(const OnnxConstant* scales,
                                      find_real_operand(progress, node, 2, 1));
                        SA_FAIL_IF(scales->reals.size() != K_BATCH_RANK ||
                                       scales->reals[0] != 1.0f || scales->reals[1] != 1.0f,
                                   StatusCode::Unsupported,
                                   "{} scales the batch or the channels, or lists other than four "
                                   "scales",
                                   describe_node(node));
                        const float factor = scales->reals[2];
                        SA_FAIL_IF(factor != scales->reals[3] || factor < 1.0f ||
                                       factor != std::floor(factor),
                                   StatusCode::Unsupported,
                                   "{} scales by {} and {}, and only one whole factor of 1 or more "
                                   "is translated",
                                   describe_node(node), scales->reals[2], scales->reals[3]);
                        return static_cast<std::int64_t>(factor);
                    }

                    SA_FAIL_IF(!has_operand(node, 3), StatusCode::Unsupported,
                               "{} states neither scales nor sizes", describe_node(node));
                    SA_TRY_ASSIGN(const OnnxConstant* sizes,
                                  find_operand_constant(progress, node, 3));
                    SA_FAIL_IF(sizes->is_real || sizes->integers.size() != K_BATCH_RANK,
                               StatusCode::Unsupported,
                               "{} states sizes that are not four integers",
                               describe_node(node));
                    const std::int64_t height = input.dimensions[1];
                    const std::int64_t width = input.dimensions[2];
                    const std::int64_t factor = sizes->integers[2] / height;
                    SA_FAIL_IF(sizes->integers[1] != input.dimensions[0] || factor < 1 ||
                                   sizes->integers[2] != factor * height ||
                                   sizes->integers[3] != factor * width,
                               StatusCode::Unsupported,
                               "{} resizes to {} by {} from {} by {}, and only a whole factor is "
                               "translated",
                               describe_node(node), sizes->integers[2], sizes->integers[3], height,
                               width);
                    return factor;
                }
            } // namespace

            Status translate_onnx_resize(const Interchange::OnnxNode& node, OnnxProgress& progress)
            {
                // The second operand, roi, is empty in every export and is not read.
                SA_TRY(check_operand_count(node, 1, 4));
                SA_TRY(check_output_count(node, 1));
                SA_FAIL_IF(default_opset(progress) < K_FIRST_OPSET, StatusCode::Unsupported,
                           "{} is of opset {}, and the translation reads Resize from opset {}",
                           describe_node(node), default_opset(progress), K_FIRST_OPSET);
                SA_TRY(reject_unknown_attributes(
                    node, SushiRuntime::span<const std::string_view>(
                              K_RESIZE_ATTRIBUTES, std::size(K_RESIZE_ATTRIBUTES))));

                // Every pairing below replicates each source cell over a factor by factor block.
                constexpr std::string_view K_COORDINATES[] = {"asymmetric", "half_pixel",
                                                              "pytorch_half_pixel"};
                constexpr std::string_view K_NEAREST_MODES[] = {"floor", "round_prefer_floor"};
                SA_TRY_ASSIGN(const std::string mode, read_text_attribute(node, "mode", "nearest"));
                SA_TRY_ASSIGN(const std::string coordinates,
                              read_text_attribute(node, "coordinate_transformation_mode",
                                                  "half_pixel"));
                SA_TRY_ASSIGN(const std::string rounding,
                              read_text_attribute(node, "nearest_mode", "round_prefer_floor"));
                SA_FAIL_IF(mode != "nearest" ||
                               !is_one_of(coordinates, K_COORDINATES, std::size(K_COORDINATES)) ||
                               !is_one_of(rounding, K_NEAREST_MODES, std::size(K_NEAREST_MODES)),
                           StatusCode::Unsupported,
                           "{} states mode '{}', coordinates '{}' and nearest_mode '{}', and only "
                           "a nearest copy of each cell is translated",
                           describe_node(node), mode, coordinates, rounding);

                SA_TRY_ASSIGN(const OnnxValue* input, find_operand_value(progress, node, 0));
                SA_FAIL_IF(input->dimensions.size() != K_IMAGE_RANK, StatusCode::Unsupported,
                           "{} reads a value of rank {}, and an upsample reads [C, H, W]",
                           describe_node(node), input->dimensions.size());
                SA_TRY_ASSIGN(const std::int64_t factor, read_factor(progress, node, *input));

                const std::int64_t row =
                    append_row(progress, "upsample", {"scale"}, {Config::Value::integer(factor)},
                               {input->row});
                define_value(progress, node.outputs[0], row,
                             {input->dimensions[0], input->dimensions[1] * factor,
                              input->dimensions[2] * factor});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
