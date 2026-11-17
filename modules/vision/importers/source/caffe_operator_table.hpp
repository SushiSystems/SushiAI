/**************************************************************************/
/* caffe_operator_table.hpp                                               */
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
 * @file caffe_operator_table.hpp
 * @brief Declares the table of Caffe layer translators and what every translator works with.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds the units of the Caffe translation that no caller names. */
        namespace Detail::Caffe
        {
            /** @brief Holds the row that carries a blob's value and the extents of one sample. */
            struct BlobValue
            {
                std::int64_t row = Config::K_MODEL_INPUT; /**< Row index, or the model input. */
                bool is_image = false;     /**< Whether a sample is [H, W, C]; else [channels]. */
                std::int64_t height = 1;   /**< Rows of an image sample. */
                std::int64_t width = 1;    /**< Columns of an image sample. */
                std::int64_t channels = 0; /**< Channels of an image, or the features of a vector. */
            };

            /** @brief Holds a blob's name and its current value. */
            struct NamedBlob
            {
                std::string name;        /**< The blob's name in the net. */
                BlobValue value;         /**< What the last layer that wrote it left. */
                std::int64_t writer = -1; /**< That layer's index; -1 for a net-level input. */
            };

            /** @brief Holds a translation under way: the rows so far and each blob's value. */
            struct TranslationProgress
            {
                const Interchange::CaffeNet* net = nullptr; /**< The net being translated. */
                bool has_weights = false;                   /**< Whether any layer holds blobs. */
                bool has_input = false;                     /**< Whether the model input is set. */
                std::vector<NamedBlob> blobs;               /**< The blobs written so far. */
                Config::ModelSpec spec;                     /**< The model the translators fill. */
                std::vector<IO::ImportedTensor> tensors;    /**< The weights, by state name. */
                std::vector<bool> absorbed;                 /**< Per layer: another one wrote it. */
            };

            /**
             * @brief Translates one layer: reads its bottoms, appends its rows and tensors, and
             *        writes its tops.
             * @param index The layer's position in the net.
             * @return Success, or a failure naming the layer.
             */
            using OperatorTranslator = Status (*)(const Interchange::CaffeLayer& layer,
                                                  std::int64_t index,
                                                  TranslationProgress& progress);

            /** @brief Holds one row of the table: a Caffe layer type and its translator. */
            struct OperatorTableRow
            {
                std::string_view type;        /**< The layer type, as the prototxt spells it. */
                OperatorTranslator translate; /**< The translator of that layer type. */
            };

            /** @brief Returns the translator row of @p type, or null when none holds it. */
            [[nodiscard]] const OperatorTableRow* find_operator_translator(
                std::string_view type) noexcept;

            /** @brief Translates an Input layer into the model input. */
            [[nodiscard]] Status translate_input(const Interchange::CaffeLayer& layer,
                                                 std::int64_t index, TranslationProgress& progress);

            /**
             * @brief Declares the model input as the blob @p name of the extents @p dimensions.
             * @param dimensions [N, C, H, W] or [N, K]; the batch axis is left out of the model.
             * @param writer The index of the layer that declares it; -1 at net level.
             * @return Unsupported for a second input or another rank; InvalidDescription for an
             *         extent below 1.
             */
            [[nodiscard]] Status declare_input(TranslationProgress& progress,
                                               const std::string& name,
                                               const std::vector<std::int64_t>& dimensions,
                                               std::int64_t writer);

            /** @brief Translates a Convolution layer into a conv2d row. */
            [[nodiscard]] Status translate_convolution(const Interchange::CaffeLayer& layer,
                                                       std::int64_t index,
                                                       TranslationProgress& progress);

            /** @brief Translates a Pooling layer into a maxpool2d, avgpool2d or global row. */
            [[nodiscard]] Status translate_pooling(const Interchange::CaffeLayer& layer,
                                                   std::int64_t index,
                                                   TranslationProgress& progress);

            /** @brief Translates a ReLU layer into a relu or leaky_relu row. */
            [[nodiscard]] Status translate_relu(const Interchange::CaffeLayer& layer,
                                                std::int64_t index, TranslationProgress& progress);

            /** @brief Translates an InnerProduct layer into a linear row, flattened first. */
            [[nodiscard]] Status translate_inner_product(const Interchange::CaffeLayer& layer,
                                                         std::int64_t index,
                                                         TranslationProgress& progress);

            /** @brief Translates a Concat layer along the channels into a concat row. */
            [[nodiscard]] Status translate_concat(const Interchange::CaffeLayer& layer,
                                                  std::int64_t index,
                                                  TranslationProgress& progress);

            /** @brief Translates an Eltwise layer into a chain of add, mul or maximum rows. */
            [[nodiscard]] Status translate_eltwise(const Interchange::CaffeLayer& layer,
                                                   std::int64_t index,
                                                   TranslationProgress& progress);

            /** @brief Translates a Softmax layer into a softmax row. */
            [[nodiscard]] Status translate_softmax(const Interchange::CaffeLayer& layer,
                                                   std::int64_t index,
                                                   TranslationProgress& progress);

            /** @brief Translates a Dropout layer into a dropout row. */
            [[nodiscard]] Status translate_dropout(const Interchange::CaffeLayer& layer,
                                                   std::int64_t index,
                                                   TranslationProgress& progress);

            /**
             * @brief Translates a BatchNorm layer, with the Scale layer after it when one reads
             *        its top, into one batchnorm2d row and its four tensors.
             */
            [[nodiscard]] Status translate_batch_norm(const Interchange::CaffeLayer& layer,
                                                      std::int64_t index,
                                                      TranslationProgress& progress);

            /** @brief Refuses a Scale layer that no BatchNorm layer before it absorbed. */
            [[nodiscard]] Status translate_scale(const Interchange::CaffeLayer& layer,
                                                 std::int64_t index, TranslationProgress& progress);

            /** @brief Returns "layer 'name' (Type)", the words every message opens with. */
            [[nodiscard]] std::string describe_layer(const Interchange::CaffeLayer& layer);

            /** @brief Returns Unsupported unless @p layer has @p bottoms bottoms and one top. */
            [[nodiscard]] Status expect_shape(const Interchange::CaffeLayer& layer,
                                              std::size_t bottoms);

            /**
             * @brief Returns the value of bottom @p position of @p layer.
             * @return InvalidDescription when no layer above wrote the blob.
             */
            [[nodiscard]] Result<BlobValue> read_bottom(const TranslationProgress& progress,
                                                        const Interchange::CaffeLayer& layer,
                                                        std::size_t position);

            /** @brief Returns Unsupported unless @p value is an image, naming @p layer. */
            [[nodiscard]] Status expect_image(const Interchange::CaffeLayer& layer,
                                              const BlobValue& value);

            /**
             * @brief Sets the value of blob @p name, replacing what an earlier layer left.
             * @param writer The index of the layer that writes it.
             */
            void write_blob(TranslationProgress& progress, const std::string& name,
                            const BlobValue& value, std::int64_t writer);

            /**
             * @brief Returns the index of the first layer after @p after that reads the blob
             *        @p name, or -1 when none does.
             */
            [[nodiscard]] std::int64_t find_reader(const TranslationProgress& progress,
                                                   const std::string& name,
                                                   std::int64_t after) noexcept;

            /** @brief Returns an object value holding @p keys and @p values, matched by index. */
            [[nodiscard]] Config::Value make_attributes(std::vector<std::string> keys,
                                                        std::vector<Config::Value> values);

            /** @brief Returns one integer when both axes agree, else the array [height, width]. */
            [[nodiscard]] Config::Value pair_value(std::int64_t height, std::int64_t width);

            /**
             * @brief Appends a row of @p type that reads @p sources and returns its index.
             * @param sources Row indices or Config::K_MODEL_INPUT, in the order the row reads.
             */
            std::int64_t append_row(TranslationProgress& progress, std::string_view type,
                                    Config::Value attributes,
                                    SushiRuntime::span<const std::int64_t> sources);

            /** @brief Returns the state entry @p field of row @p row, "<row>.<field>". */
            [[nodiscard]] std::string entry_name(std::int64_t row, std::string_view field);

            /**
             * @brief Returns Incompatible when the net carries weights and @p layer holds fewer
             *        than @p needed blobs.
             */
            [[nodiscard]] Status expect_blobs(const TranslationProgress& progress,
                                              const Interchange::CaffeLayer& layer,
                                              std::size_t needed);

            /**
             * @brief Returns the floats of blob @p position of @p layer.
             * @return Incompatible when the blob does not hold @p expected floats.
             */
            [[nodiscard]] Result<std::vector<float>> read_blob(
                const Interchange::CaffeLayer& layer, std::size_t position,
                std::size_t expected);

            /** @brief Appends a float32 tensor named @p name that holds a copy of @p values. */
            void append_tensor(TranslationProgress& progress, std::string name,
                               const Shape& shape, const std::vector<float>& values);
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
