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
 * @brief Implements the layer factories, register_layers and the layer stack builder.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/layer_registry.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/add.hpp>
#include <SushiAI/nn/avg_pool2d.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/channel_slice.hpp>
#include <SushiAI/nn/concat.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/dropout.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/flatten.hpp>
#include <SushiAI/nn/global_avg_pool.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/leaky_relu.hpp>
#include <SushiAI/nn/max_pool2d.hpp>
#include <SushiAI/nn/mish.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/silu.hpp>
#include <SushiAI/nn/unflatten.hpp>
#include <SushiAI/nn/upsample2d.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Every key a "linear" layer accepts. */
            constexpr std::string_view K_LINEAR_KEYS[] = {"units", "use_bias"};

            /** @brief Lists every key a "leaky_relu" layer accepts. */
            constexpr std::string_view K_LEAKY_RELU_KEYS[] = {"slope"};

            /** @brief Lists every key an "unflatten" layer accepts. */
            constexpr std::string_view K_UNFLATTEN_KEYS[] = {"height", "width", "channels"};

            /** @brief Lists every key a "conv2d" layer accepts. */
            constexpr std::string_view K_CONV2D_KEYS[] = {
                "filters", "kernel", "stride", "padding", "dilation", "use_bias"};

            /** @brief Lists every key an "avgpool2d" layer accepts. */
            constexpr std::string_view K_AVGPOOL2D_KEYS[] = {"kernel", "stride", "padding"};

            /** @brief Lists every key a "maxpool2d" layer accepts, trailing padding included. */
            constexpr std::string_view K_MAXPOOL2D_KEYS[] = {"kernel", "stride", "padding",
                                                             "trailing_padding"};

            /** @brief Lists every key an "upsample" layer accepts. */
            constexpr std::string_view K_UPSAMPLE_KEYS[] = {"scale"};

            /** @brief Lists every key a "slice_channels" layer accepts. */
            constexpr std::string_view K_SLICE_CHANNELS_KEYS[] = {"offset", "count"};

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
                return SushiRuntime::span<const std::string_view>(K_LINEAR_KEYS,
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
                if (value.kind() == Config::Kind::Int)
                {
                    const int64_t both = value.as_int(what);
                    return AxisPair{both, both};
                }

                SA_THROW_IF(value.kind() != Config::Kind::Array,
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
             * @brief Asks @p layer for its output shape and wraps both as a Layer.
             * @param input The per-sample shape entering @p layer.
             * @param context Prefixes the layer's own complaint about @p input or its settings.
             */
            template<typename L>
            std::unique_ptr<Layer> adapt(L layer, const Shape& input, std::string_view context)
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
            std::unique_ptr<Layer> make_keyless(const Shape& input,
                                                 const Config::Value& attributes,
                                                 std::string_view context)
            {
                attributes.reject_unknown_keys(context, no_keys());
                return adapt(L{}, input, context);
            }

            /** @brief Reads the window of a pooling row: its kernel, stride and padding. */
            Graph::ConvGeometry read_pool_geometry(const Config::Value& attributes,
                                                   std::string_view context)
            {
                const AxisPair kernel = read_axis_pair(attributes, "kernel", context);
                const AxisPair stride = read_axis_pair_or(attributes, "stride", context, kernel);
                const AxisPair padding =
                    read_axis_pair_or(attributes, "padding", context, AxisPair{0, 0});

                return make_geometry(kernel, stride, padding, AxisPair{1, 1});
            }

            /** @brief Reports whether row @p index reads port 0 of the row before it only. */
            bool reads_previous_row_only(const Config::LayerSpec& spec, std::size_t index)
            {
                if (spec.from.size() != 1 || !spec.from[0].port.empty())
                    return false;

                const std::int64_t previous = index == 0
                                                  ? Config::K_MODEL_INPUT
                                                  : static_cast<std::int64_t>(index) - 1;
                return spec.from[0].node == previous;
            }

            /** @brief Reports whether @p model publishes nothing, or its last row whole only. */
            bool publishes_last_row_only(const Config::ModelSpec& model)
            {
                if (model.outputs.empty())
                    return true;

                if (model.outputs.size() != 1 || !model.outputs[0].port.empty())
                    return false;

                return model.outputs[0].node ==
                       static_cast<std::int64_t>(model.layers.size()) - 1;
            }

            /** @brief Joins the layer types of @p registry into "'a', 'b'" for a diagnostic. */
            std::string list_layer_types(const NodeRegistry& registry)
            {
                std::string listed;
                for (const LayerEntry& entry : registry.layers())
                {
                    if (!listed.empty())
                        listed.append(", ");

                    listed.append("'").append(entry.type).append("'");
                }

                return listed;
            }

            /** @brief Builds a Mish, which takes no key. */
            std::unique_ptr<Layer> make_mish(const Shape& input, const Config::Value& attributes,
                                             std::string_view context)
            {
                return make_keyless<Mish>(input, attributes, context);
            }

            /** @brief Builds a ChannelSlice from "offset" and "count", both required. */
            std::unique_ptr<Layer> make_slice_channels(const Shape& input,
                                                       const Config::Value& attributes,
                                                       std::string_view context)
            {
                attributes.reject_unknown_keys(context, as_span(K_SLICE_CHANNELS_KEYS));

                ChannelSlice layer{};
                layer.offset = read_int(attributes, "offset", context);
                layer.count = read_int(attributes, "count", context);

                return adapt(layer, input, context);
            }
        }

        std::unique_ptr<Layer> make_linear(const Shape& input,
                                            const Config::Value& attributes,
                                            std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_LINEAR_KEYS));

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

        std::unique_ptr<Layer> make_gelu(const Shape& input,
                                          const Config::Value& attributes,
                                          std::string_view context)
        {
            return make_keyless<GELU>(input, attributes, context);
        }

        std::unique_ptr<Layer> make_relu(const Shape& input,
                                          const Config::Value& attributes,
                                          std::string_view context)
        {
            return make_keyless<ReLU>(input, attributes, context);
        }

        std::unique_ptr<Layer> make_tanh(const Shape& input,
                                          const Config::Value& attributes,
                                          std::string_view context)
        {
            return make_keyless<Tanh>(input, attributes, context);
        }

        std::unique_ptr<Layer> make_sigmoid(const Shape& input,
                                             const Config::Value& attributes,
                                             std::string_view context)
        {
            return make_keyless<Sigmoid>(input, attributes, context);
        }

        std::unique_ptr<Layer> make_leaky_relu(const Shape& input,
                                                const Config::Value& attributes,
                                                std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_LEAKY_RELU_KEYS));

            LeakyReLU layer{};
            layer.slope = read_real_or(attributes, "slope", context, layer.slope);

            return adapt(layer, input, context);
        }

        std::unique_ptr<Layer> make_unflatten(const Shape& input,
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

        std::unique_ptr<Layer> make_conv2d(const Shape& input,
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

        std::unique_ptr<Layer> make_maxpool2d(const Shape& input,
                                               const Config::Value& attributes,
                                               std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_MAXPOOL2D_KEYS));

            MaxPool2d layer{read_pool_geometry(attributes, context)};
            const AxisPair trailing =
                read_axis_pair_or(attributes, "trailing_padding", context, AxisPair{0, 0});
            layer.trailing_pad_h = trailing.h;
            layer.trailing_pad_w = trailing.w;

            return adapt(std::move(layer), input, context);
        }

        std::unique_ptr<Layer> make_avgpool2d(const Shape& input,
                                               const Config::Value& attributes,
                                               std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_AVGPOOL2D_KEYS));

            return adapt(AvgPool2d{read_pool_geometry(attributes, context)}, input, context);
        }

        std::unique_ptr<Layer> make_global_avgpool(const Shape& input,
                                                    const Config::Value& attributes,
                                                    std::string_view context)
        {
            return make_keyless<GlobalAvgPool>(input, attributes, context);
        }

        std::unique_ptr<Layer> make_flatten(const Shape& input,
                                             const Config::Value& attributes,
                                             std::string_view context)
        {
            return make_keyless<Flatten>(input, attributes, context);
        }

        std::unique_ptr<Layer> make_upsample(const Shape& input,
                                              const Config::Value& attributes,
                                              std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_UPSAMPLE_KEYS));

            Upsample2d layer{};
            layer.scale = read_int_or(attributes, "scale", context, layer.scale);

            return adapt(layer, input, context);
        }

        std::unique_ptr<Layer> make_silu(const Shape& input,
                                          const Config::Value& attributes,
                                          std::string_view context)
        {
            return make_keyless<SiLU>(input, attributes, context);
        }

        Status register_layers(NodeRegistry& registry)
        {
            SA_TRY(registry.add(LayerEntry{"linear", &make_linear}));
            SA_TRY(registry.add(LayerEntry{"gelu", &make_gelu}));
            SA_TRY(registry.add(LayerEntry{"relu", &make_relu}));
            SA_TRY(registry.add(LayerEntry{"tanh", &make_tanh}));
            SA_TRY(registry.add(LayerEntry{"sigmoid", &make_sigmoid}));
            SA_TRY(registry.add(LayerEntry{"leaky_relu", &make_leaky_relu}));
            SA_TRY(registry.add(LayerEntry{"unflatten", &make_unflatten}));
            SA_TRY(registry.add(LayerEntry{"conv2d", &make_conv2d}));
            SA_TRY(registry.add(LayerEntry{"maxpool2d", &make_maxpool2d}));
            SA_TRY(registry.add(LayerEntry{"avgpool2d", &make_avgpool2d}));
            SA_TRY(registry.add(LayerEntry{"global_avgpool", &make_global_avgpool}));
            SA_TRY(registry.add(LayerEntry{"flatten", &make_flatten}));
            SA_TRY(registry.add(LayerEntry{"upsample", &make_upsample}));
            SA_TRY(registry.add(LayerEntry{"silu", &make_silu}));
            SA_TRY(registry.add(LayerEntry{"mish", &make_mish}));
            SA_TRY(registry.add(LayerEntry{"slice_channels", &make_slice_channels}));
            SA_TRY(registry.add(NodeEntry{"batchnorm2d", &make_batch_norm2d_node}));
            SA_TRY(registry.add(NodeEntry{"concat", &make_concat_node}));
            SA_TRY(registry.add(NodeEntry{"add", &make_add_node}));
            SA_TRY(registry.add(NodeEntry{"embedding", &make_embedding_node}));
            SA_TRY(registry.add(NodeEntry{"layernorm", &make_layer_norm_node}));
            SA_TRY(registry.add(NodeEntry{"dropout", &make_dropout_node}));

            return Status();
        }

        LayerStack build_layer_stack(const NodeRegistry& registry,
                                     const Config::ModelSpec& model)
        {
            SA_THROW_IF(!publishes_last_row_only(model),
                        "the model's 'outputs' publishes something other than the whole of its "
                        "last row; a LayerStack is a chain and yields its last layer. Build "
                        "this file with build_model.");

            LayerStack stack;

            Shape shape = model.input_shape;
            for (std::size_t i = 0; i < model.layers.size(); ++i)
            {
                const Config::LayerSpec& spec = model.layers[i];
                const std::string context =
                    "layer " + std::to_string(i) + " '" + spec.type + "'";

                SA_THROW_IF(!reads_previous_row_only(spec, i),
                            "{} reads something other than the row before it; a LayerStack is "
                            "a chain. Build this file with build_model.",
                            context);

                const LayerEntry* const entry = registry.find_layer(spec.type);
                SA_THROW_IF(entry == nullptr, "unknown layer type '{}'; this build has: {}",
                            spec.type, list_layer_types(registry));

                std::unique_ptr<Layer> layer = entry->make(shape, spec.attributes, context);

                // Invariant: the next factory receives this layer's per-sample output shape.
                shape = layer->output_shape();
                stack.push(std::move(layer));
            }

            return stack;
        }

    } // namespace NN
} // namespace SushiAI
