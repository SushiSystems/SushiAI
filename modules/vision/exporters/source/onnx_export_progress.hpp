/**************************************************************************/
/* onnx_export_progress.hpp                                               */
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
 * @file onnx_export_progress.hpp
 * @brief Declares the state of an ONNX export under way and the helpers its row writers share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>
#include <SushiAI/io/imported_model.hpp>
#include "row_keys.hpp"
#include "row_weights.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        /** @brief Holds the units of the ONNX export that no caller names. */
        namespace Detail
        {
            /** @brief Names the format in a key refusal: "the ONNX importer". */
            inline constexpr std::string_view K_ONNX_FORMAT = "ONNX";

            /** @brief Holds the words a weight refusal puts before "float32". */
            inline constexpr std::string_view K_ONNX_FLOAT32_HOLDER = "the export writes";

            /** @brief Holds a value of the graph: its tensor name and its per-sample shape. */
            struct OnnxRowValue
            {
                /** @brief Holds the name of the tensor that carries the value. */
                std::string name;

                /** @brief Holds the extents of one sample in SushiAI's order, [H, W, C] or [F]. */
                std::vector<std::int64_t> shape;

                /** @brief Holds the [H, W, C] a flatten row read; empty for any other value. */
                std::vector<std::int64_t> flattened_from;
            };

            /** @brief Holds an export under way: the model, its row values and the graph. */
            struct OnnxExportProgress
            {
                /** @brief Holds the model being written; it outlives the export. */
                const IO::ImportedModel* model = nullptr;

                /** @brief Holds the graph input, `input`, of the model's input shape. */
                OnnxRowValue input;

                /** @brief Holds the value of each row; empty for a row refused or not reached. */
                std::vector<std::optional<OnnxRowValue>> values;

                /** @brief Holds the tensor name each row's value takes, set before any row. */
                std::vector<std::string> value_names;

                /** @brief Holds, for each row, the indices of the rows that read it. */
                std::vector<std::vector<std::size_t>> readers;

                /** @brief Holds, for each row, whether the model publishes it. */
                std::vector<bool> published;

                /** @brief Holds the graph the rows write into. */
                Interchange::OnnxGraph graph;
            };

            /** @brief Holds one row as its writer sees it: its index, its spec and its operands. */
            struct OnnxRowCall
            {
                /** @brief Holds the index of the row in the model. */
                std::size_t index = 0;

                /** @brief Holds the row as the model file states it. */
                const Config::LayerSpec* spec = nullptr;

                /** @brief Holds the values the row reads, in the order of its "from". */
                std::vector<const OnnxRowValue*> sources;
            };

            /** @brief Returns the keys of the row, read with its index in diagnostics. */
            [[nodiscard]] RowKeys row_keys(const OnnxRowCall& call);

            /** @brief Returns InvalidDescription unless the row reads exactly @p count values. */
            [[nodiscard]] Status check_source_count(const OnnxRowCall& call, std::size_t count);

            /** @brief Returns Unsupported unless @p value is an image [H, W, C]. */
            [[nodiscard]] Status check_image(const OnnxRowValue& value);

            /** @brief Appends a float32 initializer @p name of @p extents holding @p values. */
            void add_float_initializer(OnnxExportProgress& progress, std::string name,
                                       std::vector<std::int64_t> extents,
                                       const std::vector<float>& values);

            /** @brief Appends an int64 initializer @p name of @p extents holding @p values. */
            void add_int_initializer(OnnxExportProgress& progress, std::string name,
                                     std::vector<std::int64_t> extents,
                                     const std::vector<std::int64_t>& values);

            /** @brief Returns an attribute of type Int. */
            [[nodiscard]] Interchange::OnnxAttribute make_int_attribute(std::string name,
                                                                        std::int64_t value);

            /** @brief Returns an attribute of type Ints. */
            [[nodiscard]] Interchange::OnnxAttribute make_ints_attribute(
                std::string name, std::vector<std::int64_t> values);

            /** @brief Returns an attribute of type Float. */
            [[nodiscard]] Interchange::OnnxAttribute make_real_attribute(std::string name,
                                                                         float value);

            /** @brief Returns an attribute of type String. */
            [[nodiscard]] Interchange::OnnxAttribute make_text_attribute(std::string name,
                                                                         std::string value);

            /** @brief Appends a node of @p op_type, named after the row and @p suffix. */
            void add_node(OnnxExportProgress& progress, const OnnxRowCall& call,
                          std::string_view op_type, std::string_view suffix,
                          std::vector<std::string> inputs, std::vector<std::string> outputs,
                          std::vector<Interchange::OnnxAttribute> attributes = {});

            /** @brief Records the row's value as its tensor name with the extents @p shape. */
            void define_row_value(OnnxExportProgress& progress, const OnnxRowCall& call,
                                  std::vector<std::int64_t> shape);

            /** @brief Returns the tensor name the row's value takes. */
            [[nodiscard]] const std::string& row_output(const OnnxExportProgress& progress,
                                                        const OnnxRowCall& call);

            /** @brief Writes a row of no key and one operand as one node of @p op_type. */
            [[nodiscard]] Status write_pointwise_row(const OnnxRowCall& call,
                                                     OnnxExportProgress& progress,
                                                     std::string_view op_type);

            /**
             * @brief Writes a row of two operands of one shape as one node of @p op_type.
             * @return Unsupported when the two shapes differ, since the importer reads equal
             *         extents only.
             */
            [[nodiscard]] Status write_binary_row(const OnnxRowCall& call,
                                                  OnnxExportProgress& progress,
                                                  std::string_view op_type);
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
