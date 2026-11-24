/**************************************************************************/
/* caffe_row_table.hpp                                                    */
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
 * @file caffe_row_table.hpp
 * @brief Declares the table of the rows the Caffe exporter writes, one writer per row type.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/failure/status.hpp>
#include "caffe_export_progress.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            /**
             * @brief Writes one row as Caffe layers and records its value.
             * @return Unsupported, with a reason that does not name the row, when the row
             *         states what the Caffe importer cannot read back; any other code stops
             *         the export.
             */
            using CaffeRowWriter = Status (*)(const CaffeRowCall& call,
                                              CaffeExportProgress& progress);

            /** @brief Holds one row of the table: a row type and its writer. */
            struct CaffeRowEntry
            {
                std::string_view type; /**< The row type, such as "conv2d". */
                CaffeRowWriter write;  /**< What writes a row of the type. */
            };

            /** @brief Returns the entry of row type @p type, or null when Caffe refuses it. */
            [[nodiscard]] const CaffeRowEntry* find_caffe_row(std::string_view type) noexcept;

            /** @brief Writes a "conv2d" row as Convolution. */
            [[nodiscard]] Status write_caffe_conv2d(const CaffeRowCall& call,
                                                    CaffeExportProgress& progress);

            /** @brief Writes a "maxpool2d" row as Pooling MAX. */
            [[nodiscard]] Status write_caffe_maxpool2d(const CaffeRowCall& call,
                                                       CaffeExportProgress& progress);

            /** @brief Writes an "avgpool2d" row as Pooling AVE. */
            [[nodiscard]] Status write_caffe_avgpool2d(const CaffeRowCall& call,
                                                       CaffeExportProgress& progress);

            /** @brief Writes a "global_avgpool" row as a global Pooling AVE. */
            [[nodiscard]] Status write_caffe_global_avgpool(const CaffeRowCall& call,
                                                            CaffeExportProgress& progress);

            /** @brief Writes a "relu" row as ReLU. */
            [[nodiscard]] Status write_caffe_relu(const CaffeRowCall& call,
                                                  CaffeExportProgress& progress);

            /** @brief Writes a "leaky_relu" row as ReLU with a negative_slope. */
            [[nodiscard]] Status write_caffe_leaky_relu(const CaffeRowCall& call,
                                                        CaffeExportProgress& progress);

            /** @brief Records a "flatten" row for the linear after it; it writes no layer. */
            [[nodiscard]] Status write_caffe_flatten(const CaffeRowCall& call,
                                                     CaffeExportProgress& progress);

            /** @brief Writes a "linear" row as InnerProduct. */
            [[nodiscard]] Status write_caffe_linear(const CaffeRowCall& call,
                                                    CaffeExportProgress& progress);

            /** @brief Writes a "concat" row as Concat on axis 1. */
            [[nodiscard]] Status write_caffe_concat(const CaffeRowCall& call,
                                                    CaffeExportProgress& progress);

            /** @brief Writes an "add" row as Eltwise SUM. */
            [[nodiscard]] Status write_caffe_add(const CaffeRowCall& call,
                                                 CaffeExportProgress& progress);

            /** @brief Writes a "mul" row as Eltwise PROD. */
            [[nodiscard]] Status write_caffe_mul(const CaffeRowCall& call,
                                                 CaffeExportProgress& progress);

            /** @brief Writes a "maximum" row as Eltwise MAX. */
            [[nodiscard]] Status write_caffe_maximum(const CaffeRowCall& call,
                                                     CaffeExportProgress& progress);

            /** @brief Writes a "softmax" row as Softmax over the channel-first axis. */
            [[nodiscard]] Status write_caffe_softmax(const CaffeRowCall& call,
                                                     CaffeExportProgress& progress);

            /** @brief Writes a "dropout" row as Dropout. */
            [[nodiscard]] Status write_caffe_dropout(const CaffeRowCall& call,
                                                     CaffeExportProgress& progress);

            /** @brief Writes a "batchnorm2d" row as BatchNorm and a Scale on its top. */
            [[nodiscard]] Status write_caffe_batchnorm2d(const CaffeRowCall& call,
                                                         CaffeExportProgress& progress);
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
