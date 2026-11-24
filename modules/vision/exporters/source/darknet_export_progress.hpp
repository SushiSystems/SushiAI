/**************************************************************************/
/* darknet_export_progress.hpp                                            */
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
 * @file darknet_export_progress.hpp
 * @brief Declares the state of a Darknet export under way and the helpers its row writers share.
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
#include <SushiAI/io/imported_model.hpp>
#include "row_keys.hpp"
#include "row_weights.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            /** @brief Names the format in a key refusal: "the Darknet importer". */
            inline constexpr std::string_view K_DARKNET_FORMAT = "Darknet";

            /** @brief Holds the words a weight refusal puts before "float32". */
            inline constexpr std::string_view K_DARKNET_FLOAT32_HOLDER = "a .weights file holds";

            /** @brief Holds one `key=value` line of a .cfg section. */
            struct DarknetOptionText
            {
                std::string key;   /**< The key, as the importer spells it. */
                std::string value; /**< The value, as the importer parses it. */
            };

            /** @brief Holds one bracketed section of the .cfg and its lines, in file order. */
            struct DarknetSectionText
            {
                std::string type;                       /**< The name between the brackets. */
                std::vector<DarknetOptionText> options; /**< The lines under it. */
            };

            /** @brief Holds the floats of one [convolutional] section, in .weights order. */
            struct DarknetConvolutionFloats
            {
                std::vector<float> biases;    /**< The bias, or the batch-norm shift. */
                std::vector<float> scales;    /**< The batch-norm scale; empty if unnormalised. */
                std::vector<float> means;     /**< The running means; empty if unnormalised. */
                std::vector<float> variances; /**< The running variances; empty if unnormalised. */
                std::vector<float> weights;   /**< The kernel as [out, in, kh, kw]. */
            };

            /** @brief Holds a value of the network: its sample shape and the layer holding it. */
            struct DarknetRowValue
            {
                /** @brief Holds the extents of one sample, [H, W, C]. */
                std::vector<std::int64_t> shape;

                /** @brief Holds the Darknet layer whose output the value is; -1 for the input. */
                std::int64_t layer = -1;
            };

            /** @brief Holds an export under way: the model, its row values and the sections. */
            struct DarknetExportProgress
            {
                /** @brief Holds the model being written; it outlives the export. */
                const IO::ImportedModel* model = nullptr;

                /** @brief Holds the network input, of the model's input shape. */
                DarknetRowValue input;

                /** @brief Holds the value of each row; empty for a row refused or not reached. */
                std::vector<std::optional<DarknetRowValue>> values;

                /** @brief Holds, for each row, the indices of the rows that read it. */
                std::vector<std::vector<std::size_t>> readers;

                /** @brief Holds, for each row, whether the model publishes it. */
                std::vector<bool> published;

                /** @brief Holds, for each row, whether an earlier row's section wrote it. */
                std::vector<bool> absorbed;

                /** @brief Holds the row the last section outputs; K_MODEL_INPUT before any. */
                std::int64_t current_row = Config::K_MODEL_INPUT;

                /** @brief Holds the sections after [net]; index i is Darknet's layer i. */
                std::vector<DarknetSectionText> sections;

                /** @brief Holds the floats of each [convolutional] section, in file order. */
                std::vector<DarknetConvolutionFloats> convolutions;
            };

            /** @brief Holds one row as its writer sees it: its index, its spec and its operands. */
            struct DarknetRowCall
            {
                /** @brief Holds the index of the row in the model. */
                std::size_t index = 0;

                /** @brief Holds the row as the model file states it. */
                const Config::LayerSpec* spec = nullptr;

                /** @brief Holds the row each operand reads, K_MODEL_INPUT for the input. */
                std::vector<std::int64_t> sources;

                /** @brief Holds the values the row reads, in the order of its "from". */
                std::vector<const DarknetRowValue*> values;
            };

            /** @brief Returns the keys of @p spec, read without a row index in diagnostics. */
            [[nodiscard]] RowKeys row_keys(const Config::LayerSpec& spec);

            /**
             * @brief Returns the key @p key, an integer or an [h, w] array, as one side.
             * @return @p fallback when absent; Unsupported when h and w differ, since a
             *         Darknet window is square.
             */
            [[nodiscard]] Result<std::int64_t> read_square_key(const RowKeys& keys,
                                                               std::string_view key,
                                                               std::int64_t fallback);

            /** @brief Returns InvalidDescription unless the row reads exactly @p count values. */
            [[nodiscard]] Status check_source_count(const DarknetRowCall& call,
                                                    std::size_t count);

            /**
             * @brief Makes @p row the value the next section reads, adding a one-layer [route]
             *        when the last section outputs another row.
             * @return Unsupported when @p row is the input and a section already follows it.
             */
            [[nodiscard]] Status follow_row(DarknetExportProgress& progress, std::int64_t row);

            /** @brief Returns the layer that holds the value of @p row, or of the input. */
            [[nodiscard]] std::int64_t layer_of(const DarknetExportProgress& progress,
                                                std::int64_t row) noexcept;

            /** @brief Appends a section of @p type and @p options; returns its layer index. */
            std::int64_t add_section(DarknetExportProgress& progress, std::string type,
                                     std::vector<DarknetOptionText> options);

            /** @brief Records @p row as the output of the last section, of extents @p shape. */
            void define_row_value(DarknetExportProgress& progress, std::size_t row,
                                  std::vector<std::int64_t> shape);

            /** @brief Returns the decimal of @p value, the integer of a .cfg line. */
            [[nodiscard]] std::string spell_integer(std::int64_t value);

            /** @brief Returns the shortest decimal that parses back to the double @p value. */
            [[nodiscard]] std::string spell_real(double value);

            /** @brief Returns the shortest decimal whose double narrows to the float @p value. */
            [[nodiscard]] std::string spell_float(float value);

            /**
             * @brief Returns the shortest pixel side that the importer divides by @p side into
             *        the float @p fraction, bit for bit.
             */
            [[nodiscard]] std::string spell_anchor_side(float fraction, std::int64_t side);

            /** @brief Returns the entries of @p values joined by commas, a .cfg list. */
            [[nodiscard]] std::string spell_integer_list(const std::vector<std::int64_t>& values);
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
