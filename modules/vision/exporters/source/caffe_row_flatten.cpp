/**************************************************************************/
/* caffe_row_flatten.cpp                                                  */
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
 * @file caffe_row_flatten.cpp
 * @brief Defines the Caffe writer of a "flatten" row, which the InnerProduct after it absorbs.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
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
            Status write_caffe_flatten(const CaffeRowCall& call, CaffeExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>()));
                SA_TRY(check_source_count(call, 1));
                const CaffeRowValue& input = *call.sources[0];
                SA_TRY(check_image(input));
                const std::vector<std::size_t>& readers = progress.readers[call.index];
                const std::vector<Config::LayerSpec>& layers = progress.model->model.layers;
                const bool read_by_next_linear = readers.size() == 1 &&
                                                 readers[0] == call.index + 1 &&
                                                 layers[readers[0]].type == "linear";
                SA_FAIL_IF(progress.published[call.index] || !read_by_next_linear,
                           StatusCode::Unsupported,
                           "Caffe states a flatten only inside the InnerProduct that reads it, "
                           "and this one is published or read by another row than the linear "
                           "right after it");
                CaffeRowValue value;
                value.blob = input.blob;
                value.shape = {input.shape[0] * input.shape[1] * input.shape[2]};
                value.flattened_from = input.shape;
                progress.values[call.index] = std::move(value);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
