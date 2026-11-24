/**************************************************************************/
/* onnx_row_transpose.cpp                                                 */
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
 * @file onnx_row_transpose.cpp
 * @brief Defines the writer of the "transpose" row, one ONNX Transpose node.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/axis_translation.hpp>
#include "onnx_export_progress.hpp"
#include "onnx_row_table.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every key of the row the writer reads. */
                constexpr std::string_view K_TRANSPOSE_KEYS[] = {"perm"};

                /** @brief Returns the row's "perm" as integers. */
                Result<std::vector<std::int64_t>> read_permutation(const OnnxRowCall& call)
                {
                    SA_TRY_ASSIGN(const Config::Value& stated,
                                  call.spec->attributes.at("perm", "row 'transpose'"));
                    SA_TRY_ASSIGN(const std::vector<Config::Value>& entries,
                                  stated.as_array("row 'transpose' 'perm'"));
                    std::vector<std::int64_t> permutation;
                    for (const Config::Value& entry : entries)
                    {
                        SA_TRY_ASSIGN(const std::int64_t axis,
                                      entry.as_int("row 'transpose' 'perm'"));
                        permutation.push_back(axis);
                    }
                    return permutation;
                }
            } // namespace

            Status write_onnx_transpose(const OnnxRowCall& call, OnnxExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(
                    keys, SushiRuntime::span<const std::string_view>(
                              K_TRANSPOSE_KEYS, std::size(K_TRANSPOSE_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const OnnxRowValue& input = *call.sources[0];
                SA_TRY(check_image(input));
                SA_TRY_ASSIGN(const std::vector<std::int64_t> permutation,
                              read_permutation(call));
                SA_FAIL_IF(permutation.size() != input.shape.size(),
                           StatusCode::InvalidDescription,
                           "row {} 'transpose' lists {} axes for a value of rank {}", call.index,
                           permutation.size(), input.shape.size());
                // The batch axis leads the tensor and stays in place.
                std::vector<std::int64_t> whole = {0};
                std::vector<std::int64_t> shape;
                bool moves = false;
                for (std::size_t axis = 0; axis < permutation.size(); ++axis)
                {
                    SA_FAIL_IF(permutation[axis] < 0 ||
                                   permutation[axis] >= static_cast<std::int64_t>(
                                                            input.shape.size()),
                               StatusCode::InvalidDescription,
                               "row {} 'transpose' lists the axis {}", call.index,
                               permutation[axis]);
                    whole.push_back(permutation[axis] + 1);
                    shape.push_back(input.shape[static_cast<std::size_t>(permutation[axis])]);
                    moves = moves || permutation[axis] != static_cast<std::int64_t>(axis);
                }
                SA_FAIL_IF(!moves, StatusCode::Unsupported,
                           "its permutation moves no axis, and the ONNX importer reads such a "
                           "Transpose back as no row");
                SA_TRY_ASSIGN(const std::vector<std::int64_t> placed,
                              Interchange::channels_first_permutation(whole));
                add_node(progress, call, "Transpose", "node", {input.name},
                         {row_output(progress, call)}, {make_ints_attribute("perm", placed)});
                define_row_value(progress, call, shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
