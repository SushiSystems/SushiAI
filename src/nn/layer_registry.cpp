/**************************************************************************/
/* layer_registry.cpp                                                     */
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
 * @file layer_registry.cpp
 * @brief Implements layer registry factories and layer stack builder.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/layer_registry.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/registry.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/avg_pool2d.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/flatten.hpp>
#include <SushiAI/nn/global_avg_pool.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/leaky_relu.hpp>
#include <SushiAI/nn/max_pool2d.hpp>
#include <SushiAI/nn/unflatten.hpp>
#include <SushiAI/nn/upsample2d.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Every key a "linear" layer accepts. */
            constexpr std::string_view LINEAR_KEYS[] = {"units", "use_bias"};

            /** @brief Lists every key a "leaky_relu" layer accepts. */
            constexpr std::string_view K_LEAKY_RELU_KEYS[] = {"slope"};

            /** @brief Lists every key an "unflatten" layer accepts. */
            constexpr std::string_view K_UNFLATTEN_KEYS[] = {"height", "width", "channels"};

            /** @brief Lists every key a "conv2d" layer accepts. */
            constexpr std::string_view K_CONV2D_KEYS[] = {
                "filters", "kernel", "stride", "padding", "dilation", "use_bias"};

            /** @brief Lists every key a "maxpool2d" or an "avgpool2d" layer accepts. */
            constexpr std::string_view K_POOL_KEYS[] = {"kernel", "stride", "padding"};

            /** @brief Lists every key an "upsample" layer accepts. */
            constexpr std::string_view K_UPSAMPLE_KEYS[] = {"scale"};

            /** @brief Holds one setting of a 2-D window along its two axes. */
            struct AxisPair
            {
                int64_t h = 0; /**< @brief The value along the height axis. */
                int64_t w = 0; /**< @brief The value along the width axis. */
            };

            /** @brief Wraps a literal key list as a span, for reject_unknown_keys. */
            template<std::size_t N>
            SushiRuntime::span<const std::string_view> as_span(const std::string_view (&keys)[N])
            {
                return SushiRuntime::span<const std::string_view>(keys, N);
            }

            /**
             * @brief An empty allowed-key list, so any key at all is an error.
             *
             * The length is spelled with its type because a bare 0 is
             * ambiguous between span's (pointer, size) and (first, last)
             * constructors - the literal converts to either.
             */
            SushiRuntime::span<const std::string_view> no_keys()
            {
                return SushiRuntime::span<const std::string_view>(LINEAR_KEYS,
                                                                  static_cast<std::size_t>(0));
            }

            /** @brief Names the key @p key of the layer @p context, for a diagnostic. */
            std::string name_key(std::string_view context, std::string_view key)
            {
                std::string named(context);
                named.append(" '").append(key).append("'");
                return named;
            }

            /** @brief Reads the required whole number under @p key. */
            int64_t read_int(const Config::Value& attributes, std::string_view key,
                             std::string_view context)
            {
                return attributes.at(key, context).as_int(name_key(context, key));
            }

            /** @brief Reads the whole number under @p key, or @p fallback when it is absent. */
            int64_t read_int_or(const Config::Value& attributes, std::string_view key,
                                std::string_view context, int64_t fallback)
            {
                if (!attributes.has(key))
                    return fallback;

                return read_int(attributes, key, context);
            }

            /** @brief Reads the boolean under @p key, or @p fallback when it is absent. */
            bool read_bool_or(const Config::Value& attributes, std::string_view key,
                              std::string_view context, bool fallback)
            {
                if (!attributes.has(key))
                    return fallback;

                return attributes.at(key, context).as_bool(name_key(context, key));
            }

            /** @brief Reads the number under @p key, or @p fallback when it is absent. */
            double read_real_or(const Config::Value& attributes, std::string_view key,
                                std::string_view context, double fallback)
            {
                if (!attributes.has(key))
                    return fallback;

                return attributes.at(key, context).as_real(name_key(context, key));
            }

            /**
             * @brief Reads a window setting spelled as an integer or as a two-entry [h, w] array.
             * @param what Names the value in a diagnostic, e.g. "layer 3 'conv2d' 'kernel'".
             * @return The pair; an integer fills both axes.
             */
            AxisPair parse_axis_pair(const Config::Value& value, const std::string& what)
            {
                if (value.kind() == Config::Kind::INT)
                {
                    const int64_t both = value.as_int(what);
                    return AxisPair{both, both};
                }

                SA_THROW_IF(value.kind() != Config::Kind::ARRAY,
                            "{} must be an integer or a two-entry [h, w] array, but it is {}",
                            what, Config::kind_name(value.kind()));

                const std::vector<Config::Value>& entries = value.as_array(what);
                SA_THROW_IF(entries.size() != 2,
                            "{} must be an integer or a two-entry [h, w] array, but its array "
                            "holds {} entries",
                            what, entries.size());

                return AxisPair{entries[0].as_int(what + "[0]"), entries[1].as_int(what + "[1]")};
            }

            /** @brief Reads the required window setting under @p key. */
            AxisPair read_axis_pair(const Config::Value& attributes, std::string_view key,
                                    std::string_view context)
            {
                return parse_axis_pair(attributes.at(key, context), name_key(context, key));
            }

            /** @brief Reads the window setting under @p key, or @p fallback when it is absent. */
            AxisPair read_axis_pair_or(const Config::Value& attributes, std::string_view key,
                                       std::string_view context, AxisPair fallback)
            {
                if (!attributes.has(key))
                    return fallback;

                return read_axis_pair(attributes, key, context);
            }

            /** @brief Packs four window settings into the geometry the layers hold. */
            Graph::ConvGeometry make_geometry(AxisPair kernel, AxisPair stride, AxisPair padding,
                                              AxisPair dilation)
            {
                Graph::ConvGeometry geometry;
                geometry.kernel_h = kernel.h;
                geometry.kernel_w = kernel.w;
                geometry.stride_h = stride.h;
                geometry.stride_w = stride.w;
                geometry.dilation_h = dilation.h;
                geometry.dilation_w = dilation.w;
                geometry.pad_top = padding.h;
                geometry.pad_left = padding.w;
                return geometry;
            }

            /**
             * @brief Asks @p layer for its output shape and wraps both as an ILayer.
             * @param input The per-sample shape entering @p layer.
             * @param context Prefixes the layer's own complaint about @p input or its settings.
             */
            template<typename L>
            std::unique_ptr<ILayer> adapt(L layer, const Shape& input, std::string_view context)
            {
                Shape output;
                try
                {
                    output = layer.output_shape(input);
                }
                catch (const Error& error)
                {
                    SA_THROW_IF(true, "{}: {}", context, std::string_view(error.what()));
                }

                return std::make_unique<LayerAdapter<L>>(std::move(layer), output);
            }

            /** @brief Builds a layer of type @p L that accepts no key. */
            template<typename L>
            std::unique_ptr<ILayer> make_keyless(const Shape& input,
                                                 const Config::Value& attributes,
                                                 std::string_view context)
            {
                attributes.reject_unknown_keys(context, no_keys());
                return adapt(L{}, input, context);
            }

            /** @brief Builds a pooling layer of type @p Pool from the keys both pools share. */
            template<typename Pool>
            std::unique_ptr<ILayer> make_pool(const Shape& input,
                                              const Config::Value& attributes,
                                              std::string_view context)
            {
                attributes.reject_unknown_keys(context, as_span(K_POOL_KEYS));

                const AxisPair kernel = read_axis_pair(attributes, "kernel", context);
                const AxisPair stride = read_axis_pair_or(attributes, "stride", context, kernel);
                const AxisPair padding =
                    read_axis_pair_or(attributes, "padding", context, AxisPair{0, 0});

                return adapt(Pool{make_geometry(kernel, stride, padding, AxisPair{1, 1})},
                             input, context);
            }
        }

        std::unique_ptr<ILayer> make_linear(const Shape& input,
                                            const Config::Value& attributes,
                                            std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(LINEAR_KEYS));

            SA_THROW_IF(input.rank() != 1,
                        "{} was given a per-sample input of {}, but a linear layer needs a "
                        "rank-1 [features] input; put a 'flatten' layer before it",
                        context, input.to_string());

            const int64_t units = read_int(attributes, "units", context);
            SA_THROW_IF(units <= 0, "{} must be positive, but it is {}",
                        name_key(context, "units"), units);

            const bool use_bias = read_bool_or(attributes, "use_bias", context, true);

            return adapt(Linear{input[0], units, use_bias}, input, context);
        }

        std::unique_ptr<ILayer> make_gelu(const Shape& input,
                                          const Config::Value& attributes,
                                          std::string_view context)
        {
            return make_keyless<GELU>(input, attributes, context);
        }

        std::unique_ptr<ILayer> make_relu(const Shape& input,
                                          const Config::Value& attributes,
                                          std::string_view context)
        {
            return make_keyless<ReLU>(input, attributes, context);
        }

        std::unique_ptr<ILayer> make_tanh(const Shape& input,
                                          const Config::Value& attributes,
                                          std::string_view context)
        {
            return make_keyless<Tanh>(input, attributes, context);
        }

        std::unique_ptr<ILayer> make_sigmoid(const Shape& input,
                                             const Config::Value& attributes,
                                             std::string_view context)
        {
            return make_keyless<Sigmoid>(input, attributes, context);
        }

        std::unique_ptr<ILayer> make_leaky_relu(const Shape& input,
                                                const Config::Value& attributes,
                                                std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_LEAKY_RELU_KEYS));

            LeakyReLU layer{};
            layer.slope = read_real_or(attributes, "slope", context, layer.slope);

            return adapt(layer, input, context);
        }

        std::unique_ptr<ILayer> make_unflatten(const Shape& input,
                                               const Config::Value& attributes,
                                               std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_UNFLATTEN_KEYS));

            Unflatten layer{};
            layer.height = read_int(attributes, "height", context);
            layer.width = read_int(attributes, "width", context);
            layer.channels = read_int(attributes, "channels", context);

            return adapt(layer, input, context);
        }

        std::unique_ptr<ILayer> make_conv2d(const Shape& input,
                                            const Config::Value& attributes,
                                            std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_CONV2D_KEYS));

            SA_THROW_IF(input.rank() != 3,
                        "{} was given a per-sample input of {}, but a conv2d layer reads its "
                        "input channels from a rank-3 [H, W, C] input; put an 'unflatten' layer "
                        "before it",
                        context, input.to_string());

            const AxisPair kernel = read_axis_pair(attributes, "kernel", context);
            const AxisPair stride =
                read_axis_pair_or(attributes, "stride", context, AxisPair{1, 1});
            const AxisPair padding =
                read_axis_pair_or(attributes, "padding", context, AxisPair{0, 0});
            const AxisPair dilation =
                read_axis_pair_or(attributes, "dilation", context, AxisPair{1, 1});

            Conv2d layer{};
            layer.in_channels = input[2];
            layer.out_channels = read_int(attributes, "filters", context);
            layer.geometry = make_geometry(kernel, stride, padding, dilation);
            layer.use_bias = read_bool_or(attributes, "use_bias", context, layer.use_bias);

            return adapt(std::move(layer), input, context);
        }

        std::unique_ptr<ILayer> make_maxpool2d(const Shape& input,
                                               const Config::Value& attributes,
                                               std::string_view context)
        {
            return make_pool<MaxPool2d>(input, attributes, context);
        }

        std::unique_ptr<ILayer> make_avgpool2d(const Shape& input,
                                               const Config::Value& attributes,
                                               std::string_view context)
        {
            return make_pool<AvgPool2d>(input, attributes, context);
        }

        std::unique_ptr<ILayer> make_global_avgpool(const Shape& input,
                                                    const Config::Value& attributes,
                                                    std::string_view context)
        {
            return make_keyless<GlobalAvgPool>(input, attributes, context);
        }

        std::unique_ptr<ILayer> make_flatten(const Shape& input,
                                             const Config::Value& attributes,
                                             std::string_view context)
        {
            return make_keyless<Flatten>(input, attributes, context);
        }

        std::unique_ptr<ILayer> make_upsample(const Shape& input,
                                              const Config::Value& attributes,
                                              std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_UPSAMPLE_KEYS));

            Upsample2d layer{};
            layer.scale = read_int_or(attributes, "scale", context, layer.scale);

            return adapt(layer, input, context);
        }

        LayerStack build_layer_stack(const Config::ModelSpec& model)
        {
            LayerStack stack;

            Shape shape{model.input_features};
            for (std::size_t i = 0; i < model.layers.size(); ++i)
            {
                const Config::LayerSpec& spec = model.layers[i];
                const std::string context =
                    "layer " + std::to_string(i) + " '" + spec.type + "'";

                const LayerEntry& entry =
                    Config::find_entry(LAYER_REGISTRY, spec.type, "layer type");

                std::unique_ptr<ILayer> layer = entry.make(shape, spec.attributes, context);

                // Invariant: the next factory receives this layer's per-sample output shape.
                shape = layer->output_shape();
                stack.push(std::move(layer));
            }

            return stack;
        }

    } // namespace NN
} // namespace SushiAI
