/**************************************************************************/
/* model_config.hpp                                                       */
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
 * @file model_config.hpp
 * @brief Declares model architecture configuration parsers and specifications.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/config/input_normalise.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Config
    {
        /** @brief Holds the model-file version this build writes and the newest it reads. */
        inline constexpr std::int64_t K_MODEL_VERSION = 2;

        /** @brief Names the fit that scales each axis of an image to the input on its own. */
        inline constexpr std::string_view K_FIT_RESIZE = "resize";

        /** @brief Names the fit that scales both axes by one ratio and pads the rest. */
        inline constexpr std::string_view K_FIT_LETTERBOX = "letterbox";

        /**
         * @brief Names the fit of Darknet's letterbox_image: one scale for both axes, the
         *        scaled sides truncated, the padding grey and the image scaled up as well as down.
         */
        inline constexpr std::string_view K_FIT_DARKNET_LETTERBOX = "darknet_letterbox";

        /** @brief Marks a resolved @ref SourceSpec that reads the model input, not a row. */
        inline constexpr std::int64_t K_MODEL_INPUT = -1;

        /** @brief Marks a resolved @ref SourceSpec that reads a named extra model input. */
        inline constexpr std::int64_t K_NAMED_INPUT = -2;

        /** @brief Holds the prefix of the "from" entry that reads a named extra model input. */
        inline constexpr std::string_view K_NAMED_INPUT_PREFIX = "inputs.";

        /**
         * @brief Names one value a layer reads: a port of a row above it, the model input, or a
         *        named extra input.
         */
        struct SourceSpec
        {
            /** @brief Holds the absolute row index, @ref K_MODEL_INPUT or @ref K_NAMED_INPUT. */
            std::int64_t node = K_MODEL_INPUT;

            /** @brief Holds the port name; empty selects the row's first port. */
            std::string port;

            /** @brief Holds the name of the extra input when @c node is @ref K_NAMED_INPUT. */
            std::string input{};
        };

        /** @brief Names one extra input of a model, fed from the batch role of the same name. */
        struct NamedInputSpec
        {
            /** @brief Holds the name a row reads it by and the batch role that feeds it. */
            std::string name;

            /** @brief Holds the shape of one sample of the input, batch axis left out. */
            Shape shape;

            /** @brief Names the element type of the input: float32, or int32. */
            DType dtype = DType::Float32;
        };

        /** @brief Names what a model publishes: every port of a row, or one of them. */
        struct OutputSpec
        {
            /** @brief Holds the absolute row index. */
            std::int64_t node = 0;

            /** @brief Holds the port name; empty publishes every port of the row. */
            std::string port;

            /** @brief Holds the published name; empty keeps the port's own name. */
            std::string name;
        };

        /**
         * @struct LayerSpec
         * @brief One layer as the file spelled it: a type, and everything else.
         */
        struct LayerSpec
        {
            /** @brief The type a registered layer or node answers to, e.g. "linear". */
            std::string type;

            /** @brief Holds the row's keys without "type" and "from", defaults merged in. */
            Value attributes;

            /** @brief Lists the values the row reads, in the order the file gave them. */
            std::vector<SourceSpec> from;
        };

        /**
         * @struct ModelSpec
         * @brief A parsed architecture.
         */
        struct ModelSpec
        {
            /** @brief Holds the shape of one sample entering the model, batch axis left out. */
            Shape input_shape;

            /** @brief Names the element type of the input: float32, or int32 for indices. */
            DType input_dtype = DType::Float32;

            /**
             * @brief Names how an image is fitted to the input: @ref K_FIT_RESIZE,
             *        @ref K_FIT_LETTERBOX, @ref K_FIT_DARKNET_LETTERBOX, or empty when the file
             *        states none.
             */
            std::string input_fit;

            /**
             * @brief Holds how an image's byte levels become input values, or nothing when the
             *        file states no "normalise"; the image then converts by its default.
             */
            std::optional<InputNormalise> input_normalise;

            /**
             * @brief Lists the extra inputs the file's "inputs" object names, in file order.
             *
             * A row reads one as "inputs.<name>" in its "from"; the trainer and the evaluator
             * feed it from the batch role of the same name.
             */
            std::vector<NamedInputSpec> inputs;

            /** @brief The layers, in application order. */
            std::vector<LayerSpec> layers;

            /** @brief Lists what the model publishes; never empty after a parse. */
            std::vector<OutputSpec> outputs;

            /**
             * @brief Holds the "task" object of the file, null when the file has none.
             *
             * This module reads its "type" and nothing else; the domain that owns the type
             * reads the rest.
             */
            Value task;
        };

        /**
         * @brief Parses a model document of any version this build reads into a @ref ModelSpec.
         * @param root Parsed configuration value tree; a version 1 tree is upgraded in memory.
         * @param origin Context description for diagnostics.
         * @return The parsed specification; InvalidFile for a version this build does not read,
         *         InvalidDescription for any other wrong content.
         */
        [[nodiscard]] Result<ModelSpec> parse_model(const Value& root, std::string_view origin) noexcept;

        /**
         * @brief Reads, parses and validates the model file at @p path.
         * @param path The file to read.
         * @return The spec; FileNotFound or InvalidFile when the file cannot be read or is not
         *         valid JSON, else what parse_model returns.
         */
        [[nodiscard]] Result<ModelSpec> load_model(const std::filesystem::path& path) noexcept;

    } // namespace Config
} // namespace SushiAI
