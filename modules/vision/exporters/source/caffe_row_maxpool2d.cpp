/**************************************************************************/
/* caffe_row_maxpool2d.cpp                                                */
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
 * @file caffe_row_maxpool2d.cpp
 * @brief Defines the Caffe writer of a "maxpool2d" row: one Pooling layer of method MAX.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>
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
                /** @brief Lists every key of the row the writer reads. */
                constexpr std::string_view K_MAXPOOL2D_KEYS[] = {
                    "kernel", "stride", "padding", "trailing_padding", "leading_padding",
                    "end_padding"};
            } // namespace

            Status write_caffe_maxpool2d(const CaffeRowCall& call, CaffeExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(
                    keys, SushiRuntime::span<const std::string_view>(
                              K_MAXPOOL2D_KEYS, std::size(K_MAXPOOL2D_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const CaffeRowValue& input = *call.sources[0];
                SA_TRY(check_image(input));
                SA_TRY_ASSIGN(const PoolWindow window, read_pool_window(call));
                std::vector<std::int64_t> shape = input.shape;
                for (std::size_t axis = 0; axis < 2; ++axis)
                {
                    SA_TRY_ASSIGN(const CaffePoolAxis fitted,
                                  fit_caffe_pool_axis(window.kernel[axis], window.stride[axis],
                                                      window.padding[axis], input.shape[axis]));
                    SA_FAIL_IF(window.trailing[axis] != fitted.trailing, StatusCode::Unsupported,
                               "it pads the far edge of axis {} by {} beyond the near one, and "
                               "Caffe's rounding up adds {}",
                               axis, window.trailing[axis], fitted.trailing);
                    shape[axis] = fitted.extent;
                }
                add_layer(progress, call, "Pooling", {input.blob},
                          make_message("pooling_param",
                                       {make_word_field("pool", "MAX"),
                                        make_int_field("kernel_h", window.kernel[0]),
                                        make_int_field("kernel_w", window.kernel[1]),
                                        make_int_field("stride_h", window.stride[0]),
                                        make_int_field("stride_w", window.stride[1]),
                                        make_int_field("pad_h", window.padding[0]),
                                        make_int_field("pad_w", window.padding[1])}));
                define_row_value(progress, call, shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
