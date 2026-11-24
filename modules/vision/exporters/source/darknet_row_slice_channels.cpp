/**************************************************************************/
/* darknet_row_slice_channels.cpp                                         */
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
 * @file darknet_row_slice_channels.cpp
 * @brief Defines the writer of the "slice_channels" row, a [route] of one layer and a group.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <iterator>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include "darknet_export_progress.hpp"
#include "darknet_row_table.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every key of the row the writer reads. */
                constexpr std::string_view K_SLICE_KEYS[] = {"offset", "count"};
            } // namespace

            Status write_darknet_slice_channels(const DarknetRowCall& call,
                                                DarknetExportProgress& progress)
            {
                const Config::LayerSpec& spec = *call.spec;
                const RowKeys keys = row_keys(spec);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>(
                                                     K_SLICE_KEYS, std::size(K_SLICE_KEYS))));
                SA_TRY(check_source_count(call, 1));
                SA_TRY_ASSIGN(const std::int64_t offset, read_int_key(keys, "offset", 0));
                SA_TRY_ASSIGN(const std::int64_t count, read_int_key(keys, "count", 0));
                const std::vector<std::int64_t>& input = call.values[0]->shape;
                SA_FAIL_IF(count < 1 || offset < 0 || offset + count > input[2],
                           StatusCode::InvalidDescription,
                           "row {} 'slice_channels' takes {} channels from {} of {}", call.index,
                           count, offset, input[2]);
                SA_FAIL_IF(input[2] % count != 0 || offset % count != 0 || count == input[2],
                           StatusCode::Unsupported,
                           "it takes channels {} to {} of {}, which is not one group of two or "
                           "more equal parts",
                           offset, offset + count, input[2]);
                SA_FAIL_IF(call.sources[0] < 0, StatusCode::Unsupported,
                           "it slices the input, and a Darknet [route] names layers");
                add_section(progress, "route",
                            {DarknetOptionText{"layers",
                                               spell_integer(layer_of(progress, call.sources[0]))},
                             DarknetOptionText{"groups", spell_integer(input[2] / count)},
                             DarknetOptionText{"group_id", spell_integer(offset / count)}});
                define_row_value(progress, call.index, {input[0], input[1], count});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
