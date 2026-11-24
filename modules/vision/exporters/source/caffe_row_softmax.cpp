/**************************************************************************/
/* caffe_row_softmax.cpp                                                  */
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
 * @file caffe_row_softmax.cpp
 * @brief Defines the Caffe writer of a "softmax" row: one Softmax layer, axis channel-first.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <iterator>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/axis_translation.hpp>
#include "caffe_export_progress.hpp"
#include "caffe_row_table.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every key of the row the writer reads. */
                constexpr std::string_view K_SOFTMAX_KEYS[] = {"axis"};

                /** @brief Holds the axis of a row that states none: the last. */
                constexpr std::int64_t K_LAST_AXIS = -1;
            } // namespace

            Status write_caffe_softmax(const CaffeRowCall& call, CaffeExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>(
                                                     K_SOFTMAX_KEYS, std::size(K_SOFTMAX_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const CaffeRowValue& input = *call.sources[0];
                const std::int64_t rank = static_cast<std::int64_t>(input.shape.size());
                SA_TRY_ASSIGN(std::int64_t axis, read_int_key(keys, "axis", K_LAST_AXIS));
                SA_FAIL_IF(axis < -rank || axis >= rank, StatusCode::InvalidDescription,
                           "row {} 'softmax' states axis {} for a value of rank {}", call.index,
                           axis, rank);
                if (axis < 0)
                    axis += rank;
                // The Caffe blob is [N, ...]; the sample axis moves one place for the batch.
                SA_TRY_ASSIGN(const std::int64_t placed,
                              Interchange::channels_first_axis(axis + 1, rank + 1));
                add_layer(progress, call, "Softmax", {input.blob},
                          make_message("softmax_param", {make_int_field("axis", placed)}));
                define_row_value(progress, call, input.shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
