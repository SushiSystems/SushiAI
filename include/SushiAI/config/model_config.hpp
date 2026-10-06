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

namespace SushiAI
{
    namespace Config
    {
        /** @brief The model-file version this build reads. */
        inline constexpr std::int64_t MODEL_VERSION = 1;

        /**
         * @struct LayerSpec
         * @brief One layer as the file spelled it: a type, and everything else.
         */
        struct LayerSpec
        {
            /** @brief The @c NN::LAYER_REGISTRY key, e.g. "linear". */
            std::string type;

            /**
             * @brief Every other key the layer object carried.
             *
             * The layer object with @c "type" removed, so the key that chose
             * the factory is not also a key the factory has to allow.
             */
            Value attributes;
        };

        /**
         * @struct ModelSpec
         * @brief A parsed architecture.
         */
        struct ModelSpec
        {
            /** @brief Features per sample entering the first layer. */
            std::int64_t input_features = 0;

            /** @brief The layers, in application order. */
            std::vector<LayerSpec> layers;
        };

        /**
         * @brief Parses a model configuration value into a @ref ModelSpec.
         * @param root Parsed configuration value tree.
         * @param origin Context description for diagnostics.
         * @return Parsed model specification.
         * @throws Error If schema constraints or version requirements fail.
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
