/**************************************************************************/
/* onnx_row_table.hpp                                                     */
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
 * @file onnx_row_table.hpp
 * @brief Declares the table of the rows the ONNX exporter writes, one writer per row type.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/failure/status.hpp>
#include "onnx_export_progress.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            /**
             * @brief Writes one row as ONNX nodes and initializers and records its value.
             * @return Unsupported, with a reason that does not name the row, when the row
             *         states what the ONNX importer cannot read back; any other code stops
             *         the export.
             */
            using OnnxRowWriter = Status (*)(const OnnxRowCall& call,
                                             OnnxExportProgress& progress);

            /** @brief Holds one row of the table: a row type and its writer. */
            struct OnnxRowEntry
            {
                std::string_view type; /**< The row type, such as "conv2d". */
                OnnxRowWriter write;   /**< What writes a row of the type. */
            };

            /** @brief Returns the entry of row type @p type, or null when ONNX refuses it. */
            [[nodiscard]] const OnnxRowEntry* find_onnx_row(std::string_view type) noexcept;

            /** @brief Writes a "conv2d" row as Conv. */
            [[nodiscard]] Status write_onnx_conv2d(const OnnxRowCall& call,
                                                   OnnxExportProgress& progress);

            /** @brief Writes a "batchnorm2d" row as BatchNormalization. */
            [[nodiscard]] Status write_onnx_batchnorm2d(const OnnxRowCall& call,
                                                        OnnxExportProgress& progress);

            /** @brief Writes a "relu" row as Relu. */
            [[nodiscard]] Status write_onnx_relu(const OnnxRowCall& call,
                                                 OnnxExportProgress& progress);

            /** @brief Writes a "sigmoid" row as Sigmoid. */
            [[nodiscard]] Status write_onnx_sigmoid(const OnnxRowCall& call,
                                                    OnnxExportProgress& progress);

            /** @brief Writes a "silu" row as a Sigmoid and the Mul of it with its input. */
            [[nodiscard]] Status write_onnx_silu(const OnnxRowCall& call,
                                                 OnnxExportProgress& progress);

            /** @brief Writes an "add" row as Add. */
            [[nodiscard]] Status write_onnx_add(const OnnxRowCall& call,
                                                OnnxExportProgress& progress);

            /** @brief Writes a "sub" row as Sub. */
            [[nodiscard]] Status write_onnx_sub(const OnnxRowCall& call,
                                                OnnxExportProgress& progress);

            /** @brief Writes a "mul" row as Mul. */
            [[nodiscard]] Status write_onnx_mul(const OnnxRowCall& call,
                                                OnnxExportProgress& progress);

            /** @brief Writes a "div" row as Div. */
            [[nodiscard]] Status write_onnx_div(const OnnxRowCall& call,
                                                OnnxExportProgress& progress);

            /** @brief Writes a "softmax" row as Softmax over the channel-first axis. */
            [[nodiscard]] Status write_onnx_softmax(const OnnxRowCall& call,
                                                    OnnxExportProgress& progress);

            /** @brief Writes a "transpose" row as Transpose with a channel-first perm. */
            [[nodiscard]] Status write_onnx_transpose(const OnnxRowCall& call,
                                                      OnnxExportProgress& progress);

            /** @brief Writes a "concat" row as Concat on axis 1. */
            [[nodiscard]] Status write_onnx_concat(const OnnxRowCall& call,
                                                   OnnxExportProgress& progress);

            /** @brief Writes a "maxpool2d" row as MaxPool. */
            [[nodiscard]] Status write_onnx_maxpool2d(const OnnxRowCall& call,
                                                      OnnxExportProgress& progress);

            /** @brief Writes an "upsample" row as a nearest Resize. */
            [[nodiscard]] Status write_onnx_upsample(const OnnxRowCall& call,
                                                     OnnxExportProgress& progress);

            /** @brief Writes a "flatten" row as Flatten on axis 1. */
            [[nodiscard]] Status write_onnx_flatten(const OnnxRowCall& call,
                                                    OnnxExportProgress& progress);

            /** @brief Writes a "slice_channels" row as Slice on axis 1. */
            [[nodiscard]] Status write_onnx_slice_channels(const OnnxRowCall& call,
                                                           OnnxExportProgress& progress);

            /** @brief Writes a "linear" row as Gemm with transB 1. */
            [[nodiscard]] Status write_onnx_linear(const OnnxRowCall& call,
                                                   OnnxExportProgress& progress);
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
