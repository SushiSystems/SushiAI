/**************************************************************************/
/* caffe_row_concat.cpp                                                   */
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
 * @file caffe_row_concat.cpp
 * @brief Defines the Caffe writer of a "concat" row: one Concat layer on the channel axis.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
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
                /** @brief Holds the channel axis of a Caffe blob [N, C, H, W]. */
                constexpr std::int64_t K_CHANNEL_AXIS = 1;
            } // namespace

            Status write_caffe_concat(const CaffeRowCall& call, CaffeExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>()));
                SA_FAIL_IF(call.sources.size() < 2, StatusCode::Unsupported,
                           "it joins {} values, and the Caffe importer reads a Concat of one "
                           "bottom as no row",
                           call.sources.size());
                std::vector<std::string> bottoms;
                std::vector<std::int64_t> shape = call.sources.front()->shape;
                shape.back() = 0;
                for (const CaffeRowValue* source : call.sources)
                {
                    SA_TRY(check_image(*source));
                    SA_FAIL_IF(source->shape[0] != shape[0] || source->shape[1] != shape[1],
                               StatusCode::InvalidDescription,
                               "row {} 'concat' joins '{}', which differs from '{}' in height or "
                               "width",
                               call.index, source->blob, call.sources.front()->blob);
                    bottoms.push_back(source->blob);
                    shape.back() += source->shape.back();
                }
                add_layer(progress, call, "Concat", std::move(bottoms),
                          make_message("concat_param", {make_int_field("axis", K_CHANNEL_AXIS)}));
                define_row_value(progress, call, shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
