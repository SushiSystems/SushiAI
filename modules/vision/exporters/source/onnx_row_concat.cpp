/**************************************************************************/
/* onnx_row_concat.cpp                                                    */
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
 * @file onnx_row_concat.cpp
 * @brief Defines the writer of the "concat" row, one ONNX Concat node on the channel axis.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
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
                /** @brief Holds the channel axis of an ONNX tensor [N, C, ...]. */
                constexpr std::int64_t K_CHANNEL_AXIS = 1;

                /** @brief Returns whether @p first and @p second agree on all but the last axis. */
                bool agree_but_for_channels(const std::vector<std::int64_t>& first,
                                            const std::vector<std::int64_t>& second) noexcept
                {
                    if (first.size() != second.size() || first.empty())
                        return false;
                    for (std::size_t axis = 0; axis + 1 < first.size(); ++axis)
                    {
                        if (first[axis] != second[axis])
                            return false;
                    }
                    return true;
                }
            } // namespace

            Status write_onnx_concat(const OnnxRowCall& call, OnnxExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>()));
                SA_FAIL_IF(call.sources.empty(), StatusCode::InvalidDescription,
                           "row {} 'concat' reads no value", call.index);
                SA_FAIL_IF(call.sources.front()->shape.empty(), StatusCode::InvalidDescription,
                           "row {} 'concat' reads a value of no axis", call.index);
                std::vector<std::string> inputs;
                std::vector<std::int64_t> shape = call.sources.front()->shape;
                shape.back() = 0;
                for (const OnnxRowValue* source : call.sources)
                {
                    SA_FAIL_IF(!agree_but_for_channels(call.sources.front()->shape, source->shape),
                               StatusCode::InvalidDescription,
                               "row {} 'concat' joins '{}', which differs from '{}' on an axis "
                               "other than the channels",
                               call.index, source->name, call.sources.front()->name);
                    inputs.push_back(source->name);
                    shape.back() += source->shape.back();
                }
                add_node(progress, call, "Concat", "node", std::move(inputs),
                         {row_output(progress, call)},
                         {make_int_attribute("axis", K_CHANNEL_AXIS)});
                define_row_value(progress, call, shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
