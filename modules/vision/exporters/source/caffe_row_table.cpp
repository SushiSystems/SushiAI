/**************************************************************************/
/* caffe_row_table.cpp                                                    */
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
 * @file caffe_row_table.cpp
 * @brief Defines the table of the rows the Caffe exporter writes.
 * @author Mustafa Garip
 */

#include "caffe_row_table.hpp"

#include <string_view>

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every row type the Caffe importer reads back, with its writer. */
                constexpr CaffeRowEntry K_CAFFE_ROWS[] = {
                    {"conv2d", &write_caffe_conv2d},
                    {"maxpool2d", &write_caffe_maxpool2d},
                    {"avgpool2d", &write_caffe_avgpool2d},
                    {"global_avgpool", &write_caffe_global_avgpool},
                    {"relu", &write_caffe_relu},
                    {"leaky_relu", &write_caffe_leaky_relu},
                    {"flatten", &write_caffe_flatten},
                    {"linear", &write_caffe_linear},
                    {"concat", &write_caffe_concat},
                    {"add", &write_caffe_add},
                    {"mul", &write_caffe_mul},
                    {"maximum", &write_caffe_maximum},
                    {"softmax", &write_caffe_softmax},
                    {"dropout", &write_caffe_dropout},
                    {"batchnorm2d", &write_caffe_batchnorm2d},
                };
            } // namespace

            const CaffeRowEntry* find_caffe_row(std::string_view type) noexcept
            {
                for (const CaffeRowEntry& entry : K_CAFFE_ROWS)
                {
                    if (entry.type == type)
                        return &entry;
                }
                return nullptr;
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
