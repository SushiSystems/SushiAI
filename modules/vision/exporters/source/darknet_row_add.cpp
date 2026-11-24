/**************************************************************************/
/* darknet_row_add.cpp                                                    */
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
 * @file darknet_row_add.cpp
 * @brief Defines the writer of the "add" row, one linear [shortcut] section.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string_view>
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
            Status write_darknet_add(const DarknetRowCall& call, DarknetExportProgress& progress)
            {
                const RowKeys keys = row_keys(*call.spec);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>()));
                SA_TRY(check_source_count(call, 2));
                SA_FAIL_IF(call.values[0]->shape != call.values[1]->shape,
                           StatusCode::Unsupported,
                           "its operands differ in shape, and a [shortcut] adds two values of "
                           "one shape");
                SA_FAIL_IF(call.sources[1] < 0, StatusCode::Unsupported,
                           "it adds the input, and a Darknet [shortcut] names a layer");
                const std::int64_t named = layer_of(progress, call.sources[1]);
                SA_TRY(follow_row(progress, call.sources[0]));
                add_section(progress, "shortcut",
                            {DarknetOptionText{"from", spell_integer(named)},
                             DarknetOptionText{"activation", "linear"}});
                define_row_value(progress, call.index, call.values[0]->shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
