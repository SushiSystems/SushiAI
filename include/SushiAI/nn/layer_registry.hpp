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
#include <SushiAI/core/shape.hpp>
#include <SushiAI/nn/ilayer.hpp>
#include <SushiAI/nn/layer_stack.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @brief Builds one layer from its spec.
         * @param input The per-sample shape entering it, batch axis excluded.
         * @param attributes The layer's own keys; the factory rejects any it does not know.
         * @param context What to call this layer in a diagnostic, e.g. "layer 3 'linear'".
         * @return The layer.
         */
        using LayerFactory = std::unique_ptr<ILayer> (*)(const Shape& input,
                                                         const Config::Value& attributes,
                                                         std::string_view context);

        /**
         * @struct LayerEntry
         * @brief One row of @ref K_LAYER_REGISTRY.
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
         * @param input The predecessor's per-sample shape; must be rank 1, [features].
         * @param context What to call this layer in a diagnostic.
         * @throws Error If @p input is not rank 1, a key is unknown, "units" is absent, or it
         *         is not positive.
         */
        std::unique_ptr<ILayer> make_linear(const Shape& input,
                                            const Config::Value& attributes,
                                            std::string_view context);

        /**
         * @brief Builds a @c K_GELU, which takes no keys at all.
         * @param input The predecessor's per-sample shape; the activation preserves it.
         * @param attributes The layer's keys; any key whatsoever is an error.
         * @param context What to call this layer in a diagnostic.
         * @return The layer.
         * @throws Error If any key is present.
         */
        std::unique_ptr<ILayer> make_gelu(const Shape& input,
                                          const Config::Value& attributes,
                                          std::string_view context);

        /** @brief Builds a @c ReLU, which takes no key. */
        std::unique_ptr<ILayer> make_relu(const Shape& input,
                                          const Config::Value& attributes,
                                          std::string_view context);

        /** @brief Builds a @c Tanh, which takes no key. */
        std::unique_ptr<ILayer> make_tanh(const Shape& input,
                                          const Config::Value& attributes,
                                          std::string_view context);

        /** @brief Builds a @c Sigmoid, which takes no key. */
        std::unique_ptr<ILayer> make_sigmoid(const Shape& input,
                                             const Config::Value& attributes,
                                             std::string_view context);

        /** @brief Builds a @c LeakyReLU from an optional "slope", 0.01 when absent. */
        std::unique_ptr<ILayer> make_leaky_relu(const Shape& input,
                                                const Config::Value& attributes,
                                                std::string_view context);

        /**
         * @brief Builds an @c Unflatten from "height", "width" and "channels", all required.
         * @param input The predecessor's per-sample shape; must be [height * width * channels].
         */
        std::unique_ptr<ILayer> make_unflatten(const Shape& input,
                                               const Config::Value& attributes,
                                               std::string_view context);

        /**
         * @brief Builds a @c Conv2d from "filters" and "kernel", with optional "stride",
         *        "padding", "dilation" and "use_bias".
         * @param input The predecessor's per-sample [H, W, C] shape; C becomes in_channels.
         * @param attributes "kernel", "stride", "padding" and "dilation" each hold an integer
         *        or a two-entry [h, w] array.
         */
        std::unique_ptr<ILayer> make_conv2d(const Shape& input,
                                            const Config::Value& attributes,
                                            std::string_view context);

        /**
         * @brief Builds a @c MaxPool2d from "kernel", with optional "stride" and "padding".
         * @param input The predecessor's per-sample shape; must be rank 3, [H, W, C].
         * @param attributes "stride" defaults to "kernel"; each key holds an integer or a
         *        two-entry [h, w] array.
         */
        std::unique_ptr<ILayer> make_maxpool2d(const Shape& input,
                                               const Config::Value& attributes,
                                               std::string_view context);

        /**
         * @brief Builds an @c AvgPool2d from "kernel", with optional "stride" and "padding".
         * @param input The predecessor's per-sample shape; must be rank 3, [H, W, C].
         * @param attributes "stride" defaults to "kernel"; each key holds an integer or a
         *        two-entry [h, w] array.
         */
        std::unique_ptr<ILayer> make_avgpool2d(const Shape& input,
                                               const Config::Value& attributes,
                                               std::string_view context);

        /**
         * @brief Builds a @c GlobalAvgPool, which takes no key.
         * @param input The predecessor's per-sample shape; must be rank 3, [H, W, C].
         */
        std::unique_ptr<ILayer> make_global_avgpool(const Shape& input,
                                                    const Config::Value& attributes,
                                                    std::string_view context);

        /**
         * @brief Builds a @c Flatten, which takes no key.
         * @param input The predecessor's per-sample shape, of rank 1 or more.
         */
        std::unique_ptr<ILayer> make_flatten(const Shape& input,
                                             const Config::Value& attributes,
                                             std::string_view context);

        /**
         * @brief Builds an @c Upsample2d from an optional "scale", 2 when absent.
         * @param input The predecessor's per-sample shape; must be rank 3, [H, W, C].
         */
        std::unique_ptr<ILayer> make_upsample(const Shape& input,
                                              const Config::Value& attributes,
                                              std::string_view context);

        /** @brief Builds a @c SiLU, which takes no key. */
        std::unique_ptr<ILayer> make_silu(const Shape& input,
                                          const Config::Value& attributes,
                                          std::string_view context);

        /**
         * @brief Every layer type a model.json may name.
         *
         * A file-scope constant table, in the shape of @c Ops::K_ALL and the
         * fusion @c K_CATALOG.
         */
        inline constexpr std::array<LayerEntry, 14> K_LAYER_REGISTRY = {
            LayerEntry{"linear",         &make_linear},
            LayerEntry{"gelu",           &make_gelu},
            LayerEntry{"relu",           &make_relu},
            LayerEntry{"tanh",           &make_tanh},
            LayerEntry{"sigmoid",        &make_sigmoid},
            LayerEntry{"leaky_relu",     &make_leaky_relu},
            LayerEntry{"unflatten",      &make_unflatten},
            LayerEntry{"conv2d",         &make_conv2d},
            LayerEntry{"maxpool2d",      &make_maxpool2d},
            LayerEntry{"avgpool2d",      &make_avgpool2d},
            LayerEntry{"global_avgpool", &make_global_avgpool},
            LayerEntry{"flatten",        &make_flatten},
            LayerEntry{"upsample",       &make_upsample},
            LayerEntry{"silu",           &make_silu},
        };

        /**
         * @brief Builds the whole layer stack, chaining each layer's per-sample shape.
         * @param model Parsed model architecture specification; every row must read the row
         *        before it, row 0 the model input, and "outputs" must be empty or publish
         *        the last row whole.
         * @return Constructed layer stack ready for tracing.
         * @throws Error If a layer type is unknown or attributes are invalid.
         */
        LayerStack build_layer_stack(const Config::ModelSpec& model);

    } // namespace NN
} // namespace SushiAI
