/**************************************************************************/
/* darknet_row_upsample.cpp                                               */
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
 * @file darknet_row_upsample.cpp
 * @brief Defines the writer of the "upsample" row, one [upsample] section.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <iterator>
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
            namespace
            {
                /** @brief Lists every key of the row the writer reads. */
                constexpr std::string_view K_UPSAMPLE_KEYS[] = {"scale", "mode"};

                /** @brief Holds the one mode a Darknet [upsample] states. */
                constexpr std::string_view K_NEAREST = "nearest";

                /** @brief Holds the scale of a row that states none. */
                constexpr std::int64_t K_DEFAULT_SCALE = 2;
            } // namespace

            Status write_darknet_upsample(const DarknetRowCall& call,
                                          DarknetExportProgress& progress)
            {
                const Config::LayerSpec& spec = *call.spec;
                const RowKeys keys = row_keys(spec);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>(
                                                     K_UPSAMPLE_KEYS, std::size(K_UPSAMPLE_KEYS))));
                SA_TRY(check_source_count(call, 1));
                SA_TRY_ASSIGN(const std::string mode, read_text_key(keys, "mode", K_NEAREST));
                SA_FAIL_IF(mode != K_NEAREST, StatusCode::Unsupported,
                           "it states the mode '{}', and a Darknet [upsample] repeats the nearest "
                           "value",
                           mode);
                SA_TRY_ASSIGN(const std::int64_t scale,
                              read_int_key(keys, "scale", K_DEFAULT_SCALE));
                SA_FAIL_IF(scale < 1, StatusCode::InvalidDescription,
                           "row {} 'upsample' states a scale of {}", call.index, scale);
                const std::vector<std::int64_t>& input = call.values[0]->shape;
                SA_TRY(follow_row(progress, call.sources[0]));
                add_section(progress, "upsample",
                            {DarknetOptionText{"stride", spell_integer(scale)}});
                define_row_value(progress, call.index,
                                 {input[0] * scale, input[1] * scale, input[2]});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
