/****************************************************************/
/* layer_registry.cpp                                           */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file layer_registry.cpp
 * @brief Implements layer registry factories and layer stack builder.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/layer_registry.hpp>

#include <string>
#include <utility>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/registry.hpp>
#include <SushiAI/nn/layers.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Every key a "linear" layer accepts. */
            constexpr std::string_view LINEAR_KEYS[] = {"units", "use_bias"};

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
        }

        std::unique_ptr<ILayer> make_linear(int64_t in_features,
                                            const Config::Value& attributes,
                                            std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(LINEAR_KEYS));

            const std::string units_context = std::string(context) + " 'units'";
            const int64_t units = attributes.at("units", context).as_int(units_context);
            SA_THROW_IF(units <= 0, "{} must be positive, but it is {}", units_context, units);

            const Config::Value use_bias_default = Config::Value::boolean(true);
            const bool use_bias = attributes.at_or("use_bias", use_bias_default)
                                            .as_bool(std::string(context) + " 'use_bias'");

            return std::make_unique<LayerAdapter<Linear>>(
                Linear{in_features, units, use_bias}, units);
        }

        std::unique_ptr<ILayer> make_gelu(int64_t in_features,
                                          const Config::Value& attributes,
                                          std::string_view context)
        {
            // Invariant: Activation layers forbid attributes; unknown keys trigger rejection.
            attributes.reject_unknown_keys(context, no_keys());

            // Invariant: Activations preserve input width for output dimension.
            return std::make_unique<LayerAdapter<GELU>>(GELU{}, in_features);
        }

        LayerStack build_layer_stack(const Config::ModelSpec& model)
        {
            LayerStack stack;

            int64_t width = model.input_features;
            for (std::size_t i = 0; i < model.layers.size(); ++i)
            {
                const Config::LayerSpec& spec = model.layers[i];
                const std::string context =
                    "layer " + std::to_string(i) + " '" + spec.type + "'";

                const LayerEntry& entry =
                    Config::find_entry(LAYER_REGISTRY, spec.type, "layer type");

                std::unique_ptr<ILayer> layer = entry.make(width, spec.attributes, context);

                // Invariant: Chains layer dimensions by setting next input width to current output.
                width = layer->output_features();
                stack.push(std::move(layer));
            }

            return stack;
        }

    } // namespace NN
} // namespace SushiAI
