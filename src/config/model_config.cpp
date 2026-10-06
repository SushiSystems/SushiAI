/**************************************************************************/
/* model_config.cpp                                                       */
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
 * @file model_config.cpp
 * @brief Implements model configuration parsing and validation.
 * @author Mustafa Garip
 */

#include <SushiAI/config/model_config.hpp>

#include <utility>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Config
    {
        namespace
        {
            /** @brief Every key a model document may carry at the top level. */
            constexpr std::string_view MODEL_KEYS[] = {"sushiai_model", "input", "layers"};

            /** @brief Every key the "input" object may carry. */
            constexpr std::string_view INPUT_KEYS[] = {"features"};

            /** @brief Wraps a literal key list as a span, for reject_unknown_keys. */
            template<std::size_t N>
            SushiRuntime::span<const std::string_view> as_span(const std::string_view (&keys)[N])
            {
                return SushiRuntime::span<const std::string_view>(keys, N);
            }
        }

        ModelSpec parse_model(const Value& root, std::string_view origin)
        {
            require_version(root, "sushiai_model", MODEL_VERSION, origin);
            root.reject_unknown_keys(origin, as_span(MODEL_KEYS));

            ModelSpec model;

            const Value& input = root.at("input", origin);
            const std::string input_context = "'" + std::string(origin) + "'s 'input'";
            input.reject_unknown_keys(input_context, as_span(INPUT_KEYS));

            model.input_features = input.at("features", input_context)
                                        .as_int(input_context + " 'features'");
            SA_THROW_IF(model.input_features <= 0,
                        "{} 'features' must be positive, but it is {}",
                        input_context, model.input_features);

            const std::vector<Value>& layers =
                root.at("layers", origin).as_array("'" + std::string(origin) + "'s 'layers'");
            SA_THROW_IF(layers.empty(),
                        "'{}' declares no layers, so it describes no model", origin);

            model.layers.reserve(layers.size());
            for (std::size_t i = 0; i < layers.size(); ++i)
            {
                // Positional naming ensures error messages remain clear if type is missing.
                const std::string context = "layer " + std::to_string(i) + " of '"
                                          + std::string(origin) + "'";

                SA_THROW_IF(layers[i].kind() != Kind::OBJECT,
                            "{} must be an object, but it is {}",
                            context, kind_name(layers[i].kind()));

                LayerSpec layer;
                layer.type = layers[i].at("type", context).as_string(context + " 'type'");

                // Stripping the type discriminator isolates layer attributes for factory parsing.
                layer.attributes = layers[i].without("type");

                model.layers.push_back(std::move(layer));
            }

            return model;
        }

        ModelSpec load_model(const std::filesystem::path& path)
        {
            return parse_model(parse_json_file(path), path.string());
        }

    } // namespace Config
} // namespace SushiAI
