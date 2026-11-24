/**************************************************************************/
/* onnx_row_table.cpp                                                     */
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
 * @file onnx_row_table.cpp
 * @brief Defines the table of the rows the ONNX exporter writes.
 * @author Mustafa Garip
 */

#include "onnx_row_table.hpp"

#include <iterator>
#include <string_view>

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every row type the ONNX importer reads back, with its writer. */
                constexpr OnnxRowEntry K_ONNX_ROWS[] = {
                    {"conv2d", &write_onnx_conv2d},
                    {"batchnorm2d", &write_onnx_batchnorm2d},
                    {"relu", &write_onnx_relu},
                    {"sigmoid", &write_onnx_sigmoid},
                    {"silu", &write_onnx_silu},
                    {"add", &write_onnx_add},
                    {"sub", &write_onnx_sub},
                    {"mul", &write_onnx_mul},
                    {"div", &write_onnx_div},
                    {"softmax", &write_onnx_softmax},
                    {"transpose", &write_onnx_transpose},
                    {"concat", &write_onnx_concat},
                    {"maxpool2d", &write_onnx_maxpool2d},
                    {"upsample", &write_onnx_upsample},
                    {"flatten", &write_onnx_flatten},
                    {"slice_channels", &write_onnx_slice_channels},
                    {"linear", &write_onnx_linear},
                };
            } // namespace

            const OnnxRowEntry* find_onnx_row(std::string_view type) noexcept
            {
                for (const OnnxRowEntry& entry : K_ONNX_ROWS)
                {
                    if (entry.type == type)
                        return &entry;
                }
                return nullptr;
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
