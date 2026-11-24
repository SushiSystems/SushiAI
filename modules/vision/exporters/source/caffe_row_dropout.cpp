/**************************************************************************/
/* caffe_row_dropout.cpp                                                  */
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
 * @file caffe_row_dropout.cpp
 * @brief Defines the Caffe writer of a "dropout" row: one Dropout layer.
 * @author Mustafa Garip
 */

#include <iterator>
#include <string_view>
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
                /** @brief Lists every key of the row the writer reads. */
                constexpr std::string_view K_DROPOUT_KEYS[] = {"rate"};
            } // namespace

            Status write_caffe_dropout(const CaffeRowCall& call, CaffeExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>(
                                                     K_DROPOUT_KEYS, std::size(K_DROPOUT_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const CaffeRowValue& input = *call.sources[0];
                SA_TRY_ASSIGN(const double rate, read_real_key(keys, "rate", 0.0));
                SA_FAIL_IF(rate < 0.0 || rate >= 1.0, StatusCode::InvalidDescription,
                           "row {} 'dropout' drops the rate {}; Caffe reads a rate of 0 or more "
                           "and below 1",
                           call.index, rate);
                add_layer(progress, call, "Dropout", {input.blob},
                          make_message("dropout_param", {make_real_field("dropout_ratio", rate)}));
                define_row_value(progress, call, input.shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
