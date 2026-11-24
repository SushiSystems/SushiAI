/**************************************************************************/
/* darknet_row_concat.cpp                                                 */
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
 * @file darknet_row_concat.cpp
 * @brief Defines the writer of the "concat" row, one [route] over the layers it joins.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
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
            Status write_darknet_concat(const DarknetRowCall& call,
                                        DarknetExportProgress& progress)
            {
                const RowKeys keys = row_keys(*call.spec);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>()));
                SA_FAIL_IF(call.values.size() < 2, StatusCode::Unsupported,
                           "it joins {} values, and a [route] of one layer gives back no row",
                           call.values.size());
                std::vector<std::int64_t> layers;
                std::vector<std::int64_t> shape = call.values[0]->shape;
                shape[2] = 0;
                for (std::size_t at = 0; at < call.values.size(); ++at)
                {
                    const std::vector<std::int64_t>& part = call.values[at]->shape;
                    SA_FAIL_IF(call.sources[at] < 0, StatusCode::Unsupported,
                               "it joins the input, and a Darknet [route] names layers");
                    SA_FAIL_IF(part[0] != shape[0] || part[1] != shape[1],
                               StatusCode::InvalidDescription,
                               "row {} 'concat' joins values of different heights or widths",
                               call.index);
                    shape[2] += part[2];
                    layers.push_back(layer_of(progress, call.sources[at]));
                }
                add_section(progress, "route",
                            {DarknetOptionText{"layers", spell_integer_list(layers)}});
                define_row_value(progress, call.index, std::move(shape));
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
