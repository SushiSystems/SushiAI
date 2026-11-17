/**************************************************************************/
/* caffe_net.hpp                                                          */
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
 * @file caffe_net.hpp
 * @brief Declares the layers and weights of a Caffe net and the readers of its two files.
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
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/tensor_bytes.hpp>
#include <SushiAI/io/protobuf_text.hpp>

namespace SushiAI
{
    namespace Interchange
    {
        /** @brief Holds one blob of a layer: its extents and its float32 elements. */
        struct CaffeBlob
        {
            /** @brief Holds the extents, from BlobProto.shape or the four legacy extents. */
            std::vector<std::int64_t> dimensions;

            /** @brief Holds the elements, row-major, as little-endian float32. */
            TensorBytes data;

            /** @brief Returns the number of floats @p data holds. */
            [[nodiscard]] std::size_t element_count() const noexcept;
        };

        /** @brief Holds one layer of a net, read from the prototxt, the caffemodel or both. */
        struct CaffeLayer
        {
            std::string name;                 /**< The layer's name. */
            std::string type;                 /**< The layer type, such as "Convolution". */
            std::vector<std::string> bottoms; /**< The blobs it reads. */
            std::vector<std::string> tops;    /**< The blobs it writes. */
            std::vector<CaffeBlob> blobs;     /**< Its weights; filled from the caffemodel. */
            bool legacy = false;              /**< Whether the file spelled it as a V1 layer. */

            /**
             * @brief Holds the layer's message from the prototxt, with its parameter messages
             *        such as convolution_param; the root of an empty message otherwise.
             */
            IO::TextField parameters;
        };

        /** @brief Holds the layers of a net in file order, with the inputs the net declares. */
        struct CaffeNet
        {
            std::string name;                                     /**< The net's name. */
            std::vector<std::string> inputs;                      /**< The net-level input blobs. */
            std::vector<std::vector<std::int64_t>> input_shapes;  /**< Their shapes, by input. */
            std::vector<CaffeLayer> layers;                       /**< The layers, V1 and V2. */

            /** @brief Returns how many layers are named @p layer_name. */
            [[nodiscard]] std::size_t count_layers(std::string_view layer_name) const noexcept;

            /** @brief Returns the first layer named @p layer_name, or null when none is. */
            [[nodiscard]] const CaffeLayer* find_layer(std::string_view layer_name) const noexcept;
        };

        /**
         * @brief Reads the structure of a net from a prototxt.
         *
         * A `layers` (V1) message reads as a layer: its type enumerator becomes the V2 string
         * and the layer is marked legacy. `input_dim` or `input_shape` fills input_shapes.
         * @return InvalidFile for text the parser refuses, a layer with no name or type, an
         *         unknown V1 type or a number that is no integer, naming the line.
         */
        [[nodiscard]] Result<CaffeNet> read_caffe_prototxt(std::string_view text,
                                                           std::string_view origin) noexcept;

        /**
         * @brief Reads the layers of a caffemodel with their blobs, but not their parameters.
         * @param bytes The file; must outlive the net, whose packed blobs view it.
         * @param origin Names the file in a message.
         * @return InvalidFile for a malformed message, a blob whose floats do not match its
         *         extents or an unknown V1 type; Unsupported for a blob stored as doubles.
         */
        [[nodiscard]] Result<CaffeNet> read_caffe_model(SushiRuntime::span<const std::byte> bytes,
                                                        std::string_view origin) noexcept;

        /**
         * @brief Attaches the blobs of each layer of @p weights to the layer of @p net it names.
         *
         * Nothing changes unless every weights layer finds its place.
         * @return Incompatible when a layer with blobs names no layer of the net (the message
         *         lists each), names several, differs in type or meets a layer holding blobs.
         */
        [[nodiscard]] Status attach_caffe_weights(CaffeNet& net, const CaffeNet& weights);
    } // namespace Interchange
} // namespace SushiAI
