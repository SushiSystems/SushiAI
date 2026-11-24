/**************************************************************************/
/* caffe_export_progress.hpp                                              */
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
 * @file caffe_export_progress.hpp
 * @brief Declares the state of a Caffe export under way and the helpers its row writers share.
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
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/protobuf_text.hpp>
#include "row_keys.hpp"
#include "row_weights.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        /** @brief Holds the units of the Caffe export that no caller names. */
        namespace Detail
        {
            /** @brief Names the format in a key refusal: "the Caffe importer". */
            inline constexpr std::string_view K_CAFFE_FORMAT = "Caffe";

            /** @brief Holds the words a weight refusal puts before "float32". */
            inline constexpr std::string_view K_CAFFE_FLOAT32_HOLDER = "a caffemodel holds";

            /** @brief Holds a value of the net: its blob name and its per-sample shape. */
            struct CaffeRowValue
            {
                /** @brief Holds the name of the blob that carries the value. */
                std::string blob;

                /** @brief Holds the extents of one sample in SushiAI's order, [H, W, C] or [F]. */
                std::vector<std::int64_t> shape;

                /** @brief Holds the [H, W, C] a flatten row read; empty for any other value. */
                std::vector<std::int64_t> flattened_from;
            };

            /** @brief Holds an export under way: the model, its row values and the net. */
            struct CaffeExportProgress
            {
                /** @brief Holds the model being written; it outlives the export. */
                const IO::ImportedModel* model = nullptr;

                /** @brief Holds whether the model comes with weights, which layers then hold. */
                bool has_weights = false;

                /** @brief Holds the blob of the Input layer, of the model's input shape. */
                CaffeRowValue input;

                /** @brief Holds the value of each row; empty for a row refused or not reached. */
                std::vector<std::optional<CaffeRowValue>> values;

                /** @brief Holds the blob name each row writes, set before any row. */
                std::vector<std::string> blob_names;

                /** @brief Holds, for each row, the indices of the rows that read it. */
                std::vector<std::vector<std::size_t>> readers;

                /** @brief Holds, for each row, whether the model publishes it. */
                std::vector<bool> published;

                /** @brief Holds the net the rows write into. */
                Interchange::CaffeNet net;
            };

            /** @brief Holds one row as its writer sees it: its index, its spec and its operands. */
            struct CaffeRowCall
            {
                /** @brief Holds the index of the row in the model. */
                std::size_t index = 0;

                /** @brief Holds the row as the model file states it. */
                const Config::LayerSpec* spec = nullptr;

                /** @brief Holds the values the row reads, in the order of its "from". */
                std::vector<const CaffeRowValue*> sources;
            };

            /** @brief Holds one axis of a Caffe pooling: the windows it yields and the end pad. */
            struct CaffePoolAxis
            {
                std::int64_t extent = 0;   /**< Windows along the axis. */
                std::int64_t trailing = 0; /**< Pad past the far edge beyond the leading pad. */
            };

            /** @brief Holds the window of a pooling row along both image axes. */
            struct PoolWindow
            {
                std::vector<std::int64_t> kernel;   /**< Height and width of the window. */
                std::vector<std::int64_t> stride;   /**< Step along height and width. */
                std::vector<std::int64_t> padding;  /**< Rows above, columns left. */
                std::vector<std::int64_t> trailing; /**< Extra rows below, columns right. */
            };

            /**
             * @brief Returns the window of a pooling row; the stride defaults to the kernel.
             *
             * A row of `leading_padding` and `end_padding` gives the leading pad as `padding`
             * and the rest of the end pad as `trailing`.
             * @return Unsupported when an end pad is below the leading pad, which Caffe cannot
             *         state.
             */
            [[nodiscard]] Result<PoolWindow> read_pool_window(const CaffeRowCall& call);

            /**
             * @brief Returns the windows Caffe's Pooling yields along one axis and the end pad
             *        its rounding up adds, as the Caffe importer counts them.
             * @return Unsupported when Caffe refuses the window or no end pad gives its count.
             */
            [[nodiscard]] Result<CaffePoolAxis> fit_caffe_pool_axis(std::int64_t kernel,
                                                                    std::int64_t stride,
                                                                    std::int64_t pad,
                                                                    std::int64_t input);

            /** @brief Returns the keys of the row, read with its index in diagnostics. */
            [[nodiscard]] RowKeys row_keys(const CaffeRowCall& call);

            /** @brief Returns InvalidDescription unless the row reads exactly @p count values. */
            [[nodiscard]] Status check_source_count(const CaffeRowCall& call, std::size_t count);

            /** @brief Returns Unsupported unless @p value is an image [H, W, C]. */
            [[nodiscard]] Status check_image(const CaffeRowValue& value);

            /**
             * @brief Returns InvalidDescription unless the weights hold `<row>.bias` exactly
             *        when @p use_bias is true; success for a model without weights.
             */
            [[nodiscard]] Status check_bias(const CaffeRowCall& call,
                                            const CaffeExportProgress& progress, bool use_bias);

            /** @brief Returns a blob of @p extents holding @p values. */
            [[nodiscard]] Interchange::CaffeBlob make_blob(std::vector<std::int64_t> extents,
                                                           const std::vector<float>& values);

            /** @brief Returns a number field @p name holding @p value. */
            [[nodiscard]] IO::TextField make_int_field(std::string_view name, std::int64_t value);

            /** @brief Returns a number field @p name holding the shortest text of @p value. */
            [[nodiscard]] IO::TextField make_real_field(std::string_view name, double value);

            /** @brief Returns a bare word field @p name, such as an enumerator or true. */
            [[nodiscard]] IO::TextField make_word_field(std::string_view name,
                                                        std::string_view word);

            /** @brief Returns a message field @p name holding @p fields. */
            [[nodiscard]] IO::TextField make_message(std::string_view name,
                                                     std::vector<IO::TextField> fields);

            /**
             * @brief Appends the layer `row_<i>` of @p type that reads @p bottoms and writes the
             *        row's blob, with the parameter message @p parameters when it has a name.
             */
            void add_layer(CaffeExportProgress& progress, const CaffeRowCall& call,
                           std::string_view type, std::vector<std::string> bottoms,
                           IO::TextField parameters,
                           std::vector<Interchange::CaffeBlob> blobs = {});

            /** @brief Records the row's value as its blob with the extents @p shape. */
            void define_row_value(CaffeExportProgress& progress, const CaffeRowCall& call,
                                  std::vector<std::int64_t> shape);

            /** @brief Writes a row of two images of one shape as an Eltwise of @p operation. */
            [[nodiscard]] Status write_eltwise_row(const CaffeRowCall& call,
                                                   CaffeExportProgress& progress,
                                                   std::string_view operation);
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
