/**************************************************************************/
/* onnx_export.hpp                                                        */
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
 * @file onnx_export.hpp
 * @brief Declares the ONNX exporter, which writes a model as one `.onnx` file of opset 13.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/EXPORT.md
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace Vision::Exporters
    {
        /** @brief Holds the format name `sushiai export` gives the ONNX exporter. */
        inline constexpr std::string_view K_ONNX_EXPORT_FORMAT = "onnx";

        /**
         * @brief Returns the graph of @p model: input `input` [batch, C, H, W], one node or two
         *        per row, the weights as initializers `<row>.<field>`.
         * @return Unsupported naming each refused row as `row <i> '<type>': <reason>`, or the
         *         model's inputs, input type, input rank or missing weights.
         */
        [[nodiscard]] Result<Interchange::OnnxGraph> export_onnx_graph(
            const IO::ImportedModel& model) noexcept;

        /** @brief Returns one note for each value of @p model the ONNX file does not hold. */
        [[nodiscard]] std::vector<std::string> describe_onnx_losses(
            const IO::ImportedModel& model) noexcept;

        /**
         * @brief Writes @p model as the file `<stem>.onnx`, the exporter row of `onnx`.
         * @return What export_onnx_graph and write_onnx_graph return, and the notes.
         */
        [[nodiscard]] Result<IO::ExportedModel> export_onnx_files(const IO::ImportedModel& model,
                                                                  std::string_view stem) noexcept;
    } // namespace Vision::Exporters
} // namespace SushiAI
