/**************************************************************************/
/* layer_registry.hpp                                                     */
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
 * @file layer_registry.hpp
 * @brief Declares layer registry entries and factory functions for model building.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/nn/ilayer.hpp>
#include <SushiAI/nn/layer_stack.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @brief Builds one layer from its spec.
         * @param in_features How many features a sample carries entering it.
         * @param attributes The layer's own keys; the factory rejects any it does not know.
         * @param context What to call this layer in a diagnostic, e.g. "layer 3 'linear'".
         * @return The layer.
         */
        using LayerFactory = std::unique_ptr<ILayer> (*)(int64_t in_features,
                                                         const Config::Value& attributes,
                                                         std::string_view context);

        /**
         * @struct LayerEntry
         * @brief One row of @ref LAYER_REGISTRY.
         */
        struct LayerEntry
        {
            /** @brief The "type" a model.json spells. */
            std::string_view type;

            /** @brief What builds it. */
            LayerFactory make;
        };

        /**
         * @brief Builds a @c Linear from "units" and an optional "use_bias".
         * @param in_features The predecessor's output width.
         * @param attributes The layer's keys.
         * @param context What to call this layer in a diagnostic.
         * @return The layer.
         * @throws Error If a key is unknown, "units" is absent, or it is not positive.
         */
        std::unique_ptr<ILayer> make_linear(int64_t in_features,
                                            const Config::Value& attributes,
                                            std::string_view context);

        /**
         * @brief Builds a @c GELU, which takes no keys at all.
         * @param in_features The predecessor's output width; the activation preserves it.
         * @param attributes The layer's keys; any key whatsoever is an error.
         * @param context What to call this layer in a diagnostic.
         * @return The layer.
         * @throws Error If any key is present.
         */
        std::unique_ptr<ILayer> make_gelu(int64_t in_features,
                                          const Config::Value& attributes,
                                          std::string_view context);

        /**
         * @brief Every layer type a model.json may name.
         *
         * A file-scope constant table, in the shape of @c Ops::ALL and the
         * fusion @c CATALOG.
         */
        inline constexpr std::array<LayerEntry, 2> LAYER_REGISTRY = {
            LayerEntry{"linear", &make_linear},
            LayerEntry{"gelu",   &make_gelu},
        };

        /**
         * @brief Builds the whole layer stack, chaining each layer's input width.
         * @param model Parsed model architecture specification.
         * @return Constructed layer stack ready for tracing.
         * @throws Error If a layer type is unknown or attributes are invalid.
         */
        LayerStack build_layer_stack(const Config::ModelSpec& model);

    } // namespace NN
} // namespace SushiAI
