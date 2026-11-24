/**************************************************************************/
/* onnx_row_maxpool2d.cpp                                                 */
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
 * @file onnx_row_maxpool2d.cpp
 * @brief Defines the writer of the "maxpool2d" row, one ONNX MaxPool node.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <iterator>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include "onnx_export_progress.hpp"
#include "onnx_row_table.hpp"

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

                /** @brief Holds the pads of a window: above, left, below and right. */
                struct WindowPads
                {
                    std::vector<std::int64_t> begin; /**< The rows above and columns left. */
                    std::vector<std::int64_t> end;   /**< The rows below and columns right. */
                };

                /** @brief Returns the pads the row states in either of its two forms. */
                Result<WindowPads> read_pads(const OnnxRowCall& call)
                {
                    const RowKeys keys = row_keys(call);
                    const bool has_leading = call.spec->attributes.has("leading_padding");
                    if (has_leading)
                    {
                        SA_TRY_ASSIGN(std::vector<std::int64_t> begin,
                                      read_pair_key(keys, "leading_padding", 0));
                        SA_TRY_ASSIGN(std::vector<std::int64_t> end,
                                      read_pair_key(keys, "end_padding", 0));
                        return WindowPads{begin, end};
                    }
                    SA_TRY_ASSIGN(const std::vector<std::int64_t> padding,
                                  read_pair_key(keys, "padding", 0));
                    SA_TRY_ASSIGN(const std::vector<std::int64_t> trailing,
                                  read_pair_key(keys, "trailing_padding", 0));
                    return WindowPads{padding,
                                      {padding[0] + trailing[0], padding[1] + trailing[1]}};
                }
            } // namespace

            Status write_onnx_maxpool2d(const OnnxRowCall& call, OnnxExportProgress& progress)
            {
                const RowKeys keys = row_keys(call);
                SA_TRY(reject_unknown_keys(
                    keys, SushiRuntime::span<const std::string_view>(
                              K_MAXPOOL2D_KEYS, std::size(K_MAXPOOL2D_KEYS))));
                SA_TRY(check_source_count(call, 1));
                const OnnxRowValue& input = *call.sources[0];
                SA_TRY(check_image(input));
                SA_TRY_ASSIGN(const std::vector<std::int64_t> kernel,
                              read_pair_key(keys, "kernel", 1));
                std::vector<std::int64_t> stride = kernel;
                if (call.spec->attributes.has("stride"))
                {
                    SA_TRY_ASSIGN(stride, read_pair_key(keys, "stride", 1));
                }
                SA_TRY_ASSIGN(const WindowPads pads, read_pads(call));
                std::vector<std::int64_t> shape = input.shape;
                for (std::size_t axis = 0; axis < 2; ++axis)
                {
                    const std::int64_t span =
                        input.shape[axis] + pads.begin[axis] + pads.end[axis] - kernel[axis];
                    SA_FAIL_IF(span < 0 || stride[axis] < 1, StatusCode::InvalidDescription,
                               "row {} 'maxpool2d' does not fit its window in the input",
                               call.index);
                    shape[axis] = span / stride[axis] + 1;
                }
                add_node(progress, call, "MaxPool", "node", {input.name},
                         {row_output(progress, call)},
                         {make_ints_attribute("kernel_shape", kernel),
                          make_ints_attribute("strides", stride),
                          make_ints_attribute("pads", {pads.begin[0], pads.begin[1],
                                                       pads.end[0], pads.end[1]}),
                          make_int_attribute("ceil_mode", 0)});
                define_row_value(progress, call, shape);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
