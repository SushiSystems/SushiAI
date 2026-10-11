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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file model_config.hpp
 * @brief Declares model architecture configuration parsers and specifications.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>

namespace SushiAI
{
    namespace Config
    {
        /** @brief Holds the model-file version this build writes and the newest it reads. */
        inline constexpr std::int64_t K_MODEL_VERSION = 2;

        /** @brief Marks a resolved @ref SourceSpec that reads the model input, not a row. */
        inline constexpr std::int64_t K_MODEL_INPUT = -1;

        /** @brief Names one value a layer reads: a port of a row above it, or the model input. */
        struct SourceSpec
        {
            /** @brief Holds the absolute row index, or @ref K_MODEL_INPUT. */
            std::int64_t node = K_MODEL_INPUT;

            /** @brief Holds the port name; empty selects the row's first port. */
            std::string port;
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
         * @return Parsed model specification.
         */
        ModelSpec parse_model(const Value& root, std::string_view origin);

        /**
         * @brief Reads, parses and validates the model file at @p path.
         * @param path The file to read.
         * @return The spec.
         * @throws Error If the file cannot be read, is not valid JSON, or is
         *         not a valid model document.
         */
        ModelSpec load_model(const std::filesystem::path& path);

    } // namespace Config
} // namespace SushiAI
