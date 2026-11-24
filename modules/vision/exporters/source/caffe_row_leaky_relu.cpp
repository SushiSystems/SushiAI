/**************************************************************************/
/* caffe_row_leaky_relu.cpp                                               */
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
 * @file caffe_row_leaky_relu.cpp
 * @brief Defines the Caffe writer of a "leaky_relu" row: one ReLU layer with a negative_slope.
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
                constexpr std::string_view K_LEAKY_RELU_KEYS[] = {"slope"};

                /** @brief Holds the slope of a row that states none, as nn's LeakyReLU has it. */
                constexpr double K_DEFAULT_SLOPE = 0.01;
            } // namespace

            Status write_caffe_leaky_relu(const CaffeRowCall& call, CaffeExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(
                    keys, SushiRuntime::span<const std::string_view>(
                              K_LEAKY_RELU_KEYS, std::size(K_LEAKY_RELU_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const CaffeRowValue& input = *call.sources[0];
                SA_TRY_ASSIGN(const double slope, read_real_key(keys, "slope", K_DEFAULT_SLOPE));
                SA_FAIL_IF(slope == 0.0, StatusCode::Unsupported,
                           "it states a slope of 0, which the Caffe importer reads back as relu");
                add_layer(progress, call, "ReLU", {input.blob},
                          make_message("relu_param", {make_real_field("negative_slope", slope)}));
                define_row_value(progress, call, input.shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
