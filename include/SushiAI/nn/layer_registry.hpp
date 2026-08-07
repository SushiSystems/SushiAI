/**************************************************************************/
/* layer_registry.hpp                                                     */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file layer_registry.hpp
 * @brief Which layer types a model.json may name, and what builds each one.
 *
 * @section growth How this grows
 *
 * Adding a layer type is **one table row and one factory function**. No driver
 * changes, no parser changes, and no schema changes outside the factory - which
 * is the only thing that knows which keys its layer accepts, and rejects any it
 * does not. A schema change therefore stays local to the layer that caused it.
 *
 * Nothing self-registers; see config/registry.hpp for why that is forced rather
 * than preferred in a static library.
 *
 * Host lane: no SushiBLAS engine, no SYCL.
 */

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
         * @brief Builds the whole stack, chaining each layer's input width.
         *
         * @c units is an output width only: @c ModelSpec::input_features seeds
         * the chain and each layer's input is its predecessor's output, so a
         * dimension mismatch between adjacent layers is not expressible.
         *
         * @param model The parsed architecture.
         * @return The stack.
         * @throws Error If a type is unknown - the message lists every type
         *         this build has - or a factory rejects its attributes.
         */
        LayerStack build_layer_stack(const Config::ModelSpec& model);

    } // namespace NN
} // namespace SushiAI
