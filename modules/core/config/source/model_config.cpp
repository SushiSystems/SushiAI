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

#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/registry.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Config
    {
        namespace
        {
            /** @brief Names the key that states the version of a model file. */
            constexpr std::string_view K_VERSION_KEY = "format_version";

            /** @brief Names the key a version 1 model file states its version under. */
            constexpr std::string_view K_FIRST_VERSION_KEY = "sushiai_model";

            /** @brief Holds the only version written under @ref K_FIRST_VERSION_KEY. */
            constexpr std::int64_t K_FIRST_VERSION = 1;

            /** @brief Holds the "from" entry that means the row before. */
            constexpr std::int64_t K_PREVIOUS_ROW = -1;

            /** @brief Holds the "from" entry that means the model input. */
            constexpr std::string_view K_INPUT_SOURCE = "input";

            /** @brief Every key a model document may carry at the top level. */
            constexpr std::string_view K_MODEL_KEYS[] = {"format_version", "input", "defaults",
                                                       "layers", "outputs", "task"};

            /** @brief Every key the "input" object may carry. */
            constexpr std::string_view K_INPUT_KEYS[] = {"shape", "dtype", "fit"};

            /** @brief Holds one element type the "input" object may name. */
            struct InputTypeEntry
            {
                std::string_view type; /**< The text a model file spells. */
                DType dtype;           /**< The element type it names. */
            };

            /**
             * @brief Lists every element type the "input" object may name.
             *
             * A file without the key has the first.
             */
            constexpr std::array<InputTypeEntry, 2> K_INPUT_TYPES = {{
                {"float32", DType::Float32},
                {"int32", DType::Int32},
            }};

            /** @brief Holds one way the "input" object may fit an image to its shape. */
            struct InputFitEntry
            {
                std::string_view type; /**< The text a model file spells. */
            };

            /** @brief Lists every fit the "input" object may name. */
            constexpr std::array<InputFitEntry, 2> K_INPUT_FITS = {{
                {K_FIT_RESIZE},
                {K_FIT_LETTERBOX},
            }};

            /** @brief Lists every key a version 1 document may carry at the top level. */
            constexpr std::string_view K_FIRST_MODEL_KEYS[] = {"sushiai_model", "input", "layers"};

            /** @brief Lists every key the "input" object of a version 1 document may carry. */
            constexpr std::string_view K_FIRST_INPUT_KEYS[] = {"features"};

            /** @brief Lists every key the object form of a "from" entry may carry. */
            constexpr std::string_view K_SOURCE_KEYS[] = {"node", "port"};

            /** @brief Lists every key an "outputs" entry may carry. */
            constexpr std::string_view K_OUTPUT_KEYS[] = {"node", "port", "name"};

            /** @brief Wraps a literal key list as a span, for reject_unknown_keys. */
            template<std::size_t N>
            SushiRuntime::span<const std::string_view> as_span(const std::string_view (&keys)[N])
            {
                return SushiRuntime::span<const std::string_view>(keys, N);
            }

            /** @brief Returns the text that names row @p index of @p origin in a diagnostic. */
            std::string layer_context(std::size_t index, std::string_view origin)
            {
                return "layer " + std::to_string(index) + " of '" + std::string(origin) + "'";
            }

            /** @brief Returns the version @p root states, under either version key. */
            Result<std::int64_t> stated_version(const Value& root, std::string_view origin)
            {
                SA_FAIL_IF(root.kind() != Kind::Object, StatusCode::InvalidFile,
                           "'{}' must be an object that states '{}', but it is {}",
                           origin, K_VERSION_KEY, kind_name(root.kind()));

                if (root.has(K_VERSION_KEY))
                {
                    const std::string context = "'" + std::string(origin) + "'s '"
                                              + std::string(K_VERSION_KEY) + "'";
                    SA_TRY_ASSIGN(const Value& stated, root.at(K_VERSION_KEY, origin));
                    SA_TRY_ASSIGN(const std::int64_t version, stated.as_int(context));

                    SA_FAIL_IF(version <= K_FIRST_VERSION, StatusCode::InvalidFile,
                               "{} is {}, but that key starts at 2; a version 1 model file "
                               "states '{}' instead", context, version, K_FIRST_VERSION_KEY);
                    return version;
                }

                SA_FAIL_IF(!root.has(K_FIRST_VERSION_KEY), StatusCode::InvalidFile,
                           "'{}' has neither '{}' nor the version 1 key '{}'. Every model "
                           "file leads with its version, so that a format change is an error "
                           "rather than a misread.",
                           origin, K_VERSION_KEY, K_FIRST_VERSION_KEY);

                const std::string context = "'" + std::string(origin) + "'s '"
                                          + std::string(K_FIRST_VERSION_KEY) + "'";
                SA_TRY_ASSIGN(const Value& stated, root.at(K_FIRST_VERSION_KEY, origin));
                SA_TRY_ASSIGN(const std::int64_t version, stated.as_int(context));

                SA_FAIL_IF(version != K_FIRST_VERSION, StatusCode::InvalidFile,
                           "'{}' is a {} version {} document, but that key marks version {} "
                           "only; a later model file states '{}'",
                           origin, K_FIRST_VERSION_KEY, version, K_FIRST_VERSION,
                           K_VERSION_KEY);
                return version;
            }

            /** @brief Returns a copy of the object @p row with one entry appended. */
            Value with_entry(const Value& row, std::string_view key, Value value)
            {
                std::vector<std::string> keys = row.keys();
                std::vector<Value> values = row.values();
                values.reserve(keys.size() + 1);

                keys.emplace_back(key);
                values.push_back(std::move(value));
                return Value::object(std::move(keys), std::move(values));
            }

            /**
             * @brief Rewrites the version 1 document @p root of @p origin as version 2.
             * @return The version 2 tree: "format_version" 2, "input.shape" [features], and
             *         "from" -1 on every row that is an object.
             */
            Result<Value> upgrade_model_v1_to_v2(const Value& root, std::string_view origin)
            {
                SA_TRY(root.reject_unknown_keys(origin, as_span(K_FIRST_MODEL_KEYS)));

                SA_TRY_ASSIGN(const Value& input, root.at("input", origin));
                const std::string input_context = "'" + std::string(origin) + "'s 'input'";
                SA_TRY(input.reject_unknown_keys(input_context, as_span(K_FIRST_INPUT_KEYS)));

                SA_TRY_ASSIGN(const Value& stated_features, input.at("features", input_context));
                SA_TRY_ASSIGN(const std::int64_t features,
                              stated_features.as_int(input_context + " 'features'"));
                SA_FAIL_IF(features <= 0, StatusCode::InvalidDescription,
                           "{} 'features' must be positive, but it is {}",
                           input_context, features);

                SA_TRY_ASSIGN(const Value& stated_layers, root.at("layers", origin));
                SA_TRY_ASSIGN(const std::vector<Value>& layers,
                              stated_layers.as_array("'" + std::string(origin) + "'s 'layers'"));

                std::vector<Value> rows;
                rows.reserve(layers.size());
                for (std::size_t i = 0; i < layers.size(); ++i)
                {
                    // A row that is not an object passes through; the version 2 reader names it.
                    if (layers[i].kind() != Kind::Object)
                    {
                        rows.push_back(layers[i]);
                        continue;
                    }

                    const std::string context = layer_context(i, origin);
                    SA_FAIL_IF(layers[i].has("from"), StatusCode::InvalidDescription,
                               "{} carries 'from', which a version 1 model file does not "
                               "have; state '{}' 2 to use it", context, K_VERSION_KEY);

                    rows.push_back(with_entry(layers[i], "from", Value::integer(K_PREVIOUS_ROW)));
                }

                Value shape = Value::object({"shape"}, {Value::array({Value::integer(features)})});
                return Value::object({std::string(K_VERSION_KEY), "input", "layers"},
                                     {Value::integer(K_MODEL_VERSION), std::move(shape),
                                      Value::array(std::move(rows))});
            }

            /** @brief Reads the "input" object of a version 2 document into the sample shape. */
            Result<Shape> parse_input_shape(const Value& input, const std::string& context)
            {
                SA_TRY(input.reject_unknown_keys(context, as_span(K_INPUT_KEYS)));

                const std::string shape_context = context + " 'shape'";
                SA_TRY_ASSIGN(const Value& stated_shape, input.at("shape", context));
                SA_TRY_ASSIGN(const std::vector<Value>& entries,
                              stated_shape.as_array(shape_context));

                SA_FAIL_IF(entries.empty(), StatusCode::InvalidDescription,
                           "{} is empty; it lists the extents of one sample", shape_context);
                SA_FAIL_IF(entries.size() > static_cast<std::size_t>(K_MAX_RANK),
                           StatusCode::InvalidDescription,
                           "{} lists {} extents, but a sample has at most {}",
                           shape_context, entries.size(), K_MAX_RANK);

                std::vector<std::int64_t> extents;
                extents.reserve(entries.size());
                std::int64_t count = 1;
                for (std::size_t i = 0; i < entries.size(); ++i)
                {
                    SA_TRY_ASSIGN(const std::int64_t extent,
                                  entries[i].as_int(shape_context + " entry " + std::to_string(i)));
                    SA_FAIL_IF(extent <= 0, StatusCode::InvalidDescription,
                               "{} entry {} must be positive, but it is {}",
                               shape_context, i, extent);
                    SA_FAIL_IF(extent > std::numeric_limits<std::int64_t>::max() / count,
                               StatusCode::InvalidDescription,
                               "{} describes more elements than a sample can hold; entry {} "
                               "is {}", shape_context, i, extent);
                    count *= extent;
                    extents.push_back(extent);
                }

                return Shape::make(SushiRuntime::span<const std::int64_t>(extents.data(),
                                                                          extents.size()));
            }

            /** @brief Reads "dtype" of the "input" object; an absent key gives float32. */
            Result<DType> parse_input_dtype(const Value& input, const std::string& context)
            {
                if (!input.has("dtype"))
                    return K_INPUT_TYPES.front().dtype;

                const std::string dtype_context = context + " 'dtype'";
                SA_TRY_ASSIGN(const Value& stated_dtype, input.at("dtype", context));
                SA_TRY_ASSIGN(const std::string& stated, stated_dtype.as_string(dtype_context));
                SA_TRY_ASSIGN(const InputTypeEntry& entry,
                              find_entry(K_INPUT_TYPES, stated, dtype_context));
                return entry.dtype;
            }

            /** @brief Reads "fit" of the "input" object; an absent key gives the empty name. */
            Result<std::string> parse_input_fit(const Value& input, const std::string& context)
            {
                if (!input.has("fit"))
                    return std::string();

                const std::string fit_context = context + " 'fit'";
                SA_TRY_ASSIGN(const Value& stated_fit, input.at("fit", context));
                SA_TRY_ASSIGN(const std::string& stated, stated_fit.as_string(fit_context));
                SA_TRY_ASSIGN(const InputFitEntry& entry,
                              find_entry(K_INPUT_FITS, stated, fit_context));
                return std::string(entry.type);
            }

            /** @brief Returns the text that names the "defaults" block of @p origin. */
            std::string defaults_context(std::string_view origin)
            {
                return "'" + std::string(origin) + "'s 'defaults'";
            }

            /** @brief Refuses a "defaults" block that is not a map from a type to plain keys. */
            Status check_defaults(const Value& defaults, std::string_view origin)
            {
                const std::string context = defaults_context(origin);
                SA_FAIL_IF(defaults.kind() != Kind::Object, StatusCode::InvalidDescription,
                           "{} must be an object, but it is {}",
                           context, kind_name(defaults.kind()));

                for (std::size_t i = 0; i < defaults.keys().size(); ++i)
                {
                    const std::string& type = defaults.keys()[i];
                    const Value& entry = defaults.values()[i];
                    SA_FAIL_IF(entry.kind() != Kind::Object, StatusCode::InvalidDescription,
                               "{} for '{}' must be an object, but it is {}",
                               context, type, kind_name(entry.kind()));
                    SA_FAIL_IF(entry.has("type") || entry.has("from"),
                               StatusCode::InvalidDescription,
                               "{} for '{}' states 'type' or 'from'; a default fills a "
                               "layer's other keys only", context, type);
                }
                return Status();
            }

            /** @brief Returns each distinct type of @p layers once, quoted, in row order. */
            std::string stated_types(const std::vector<LayerSpec>& layers)
            {
                std::vector<std::string_view> seen;
                std::string text;

                for (const LayerSpec& layer : layers)
                {
                    if (std::find(seen.begin(), seen.end(), layer.type) != seen.end())
                        continue;

                    seen.push_back(layer.type);
                    text += (text.empty() ? "'" : ", '") + layer.type + "'";
                }
                return text;
            }

            /** @brief Refuses a key of @p defaults that is the type of no row in @p layers. */
            Status check_defaults_are_used(const Value& defaults,
                                           const std::vector<LayerSpec>& layers,
                                           std::string_view origin)
            {
                for (const std::string& type : defaults.keys())
                {
                    const bool used = std::any_of(layers.begin(), layers.end(),
                                                  [&type](const LayerSpec& layer)
                                                  {
                                                      return layer.type == type;
                                                  });
                    SA_FAIL_IF(!used, StatusCode::InvalidDescription,
                               "{} names type '{}', but no layer has that type; the layers "
                               "state {}", defaults_context(origin), type, stated_types(layers));
                }
                return Status();
            }

            /**
             * @brief Builds a row's attributes: its keys without "type" and "from", then each
             *        key of @p defaults the row does not state.
             */
            Value layer_attributes(const Value& row, const Value& defaults)
            {
                std::vector<std::string> keys;
                std::vector<Value> values;

                for (std::size_t i = 0; i < row.keys().size(); ++i)
                {
                    const std::string& key = row.keys()[i];
                    if (key == "type" || key == "from")
                        continue;

                    keys.push_back(key);
                    values.push_back(row.values()[i]);
                }

                for (std::size_t i = 0; i < defaults.keys().size(); ++i)
                {
                    const std::string& key = defaults.keys()[i];
                    if (row.has(key))
                        continue;

                    keys.push_back(key);
                    values.push_back(defaults.values()[i]);
                }

                return Value::object(std::move(keys), std::move(values));
            }

            /**
             * @brief Resolves the row number @p stated by row @p index to an absolute one.
             * @return A row above @p index, or @ref K_MODEL_INPUT for the row before row 0.
             */
            Result<std::int64_t> resolve_row(std::int64_t stated, std::size_t index,
                                             std::string_view context)
            {
                const std::int64_t row = static_cast<std::int64_t>(index);

                if (stated == K_PREVIOUS_ROW)
                    return row == 0 ? K_MODEL_INPUT : row - 1;

                SA_FAIL_IF(stated < 0, StatusCode::InvalidDescription,
                           "{} names row {}; the only negative entry is {}, the row before",
                           context, stated, K_PREVIOUS_ROW);
                SA_FAIL_IF(stated >= row, StatusCode::InvalidDescription,
                           "{} names row {}, which is not above row {}; a layer reads only "
                           "the model input and the rows above it", context, stated, row);
                return stated;
            }

            /**
             * @brief Returns the string under @p key of @p entry, refusing an empty one.
             * @param absence What leaving the key out does, to end the refusal with.
             */
            Result<std::string> stated_text(const Value& entry, std::string_view key,
                                            const std::string& context,
                                            std::string_view absence)
            {
                const std::string key_context = context + " '" + std::string(key) + "'";
                SA_TRY_ASSIGN(const Value& member, entry.at(key, context));
                SA_TRY_ASSIGN(const std::string& text, member.as_string(key_context));

                SA_FAIL_IF(text.empty(), StatusCode::InvalidDescription,
                           "{} is an empty string; leave the key out to {}",
                           key_context, absence);
                return text;
            }

            /** @brief Reads one "from" entry of row @p index, in any of its four forms. */
            Result<SourceSpec> parse_source(const Value& entry, std::size_t index,
                                            const std::string& context)
            {
                SourceSpec source;

                if (entry.kind() == Kind::Int)
                {
                    SA_TRY_ASSIGN(const std::int64_t stated, entry.as_int(context));
                    SA_TRY_ASSIGN(source.node, resolve_row(stated, index, context));
                    return source;
                }

                if (entry.kind() == Kind::String)
                {
                    SA_TRY_ASSIGN(const std::string& stated, entry.as_string(context));
                    SA_FAIL_IF(stated != K_INPUT_SOURCE, StatusCode::InvalidDescription,
                               "{} is the string '{}'; the only string it accepts is '{}'",
                               context, stated, K_INPUT_SOURCE);
                    source.node = K_MODEL_INPUT;
                    return source;
                }

                SA_FAIL_IF(entry.kind() != Kind::Object, StatusCode::InvalidDescription,
                           "{} must be a row number, '{}' or an object carrying 'node', but "
                           "it is {}", context, K_INPUT_SOURCE, kind_name(entry.kind()));

                SA_TRY(entry.reject_unknown_keys(context, as_span(K_SOURCE_KEYS)));
                SA_TRY_ASSIGN(const Value& stated_node, entry.at("node", context));
                SA_TRY_ASSIGN(const std::int64_t node, stated_node.as_int(context + " 'node'"));
                SA_TRY_ASSIGN(source.node, resolve_row(node, index, context));
                if (entry.has("port"))
                {
                    SA_TRY_ASSIGN(source.port,
                                  stated_text(entry, "port", context, "read the row's first port"));
                }
                return source;
            }

            /** @brief Reads the "from" of row @p index; an absent key reads the row before. */
            Result<std::vector<SourceSpec>> parse_sources(const Value& row, std::size_t index,
                                                          const std::string& layer)
            {
                const std::string context = layer + " 'from'";
                std::vector<SourceSpec> sources;

                if (!row.has("from"))
                {
                    SA_TRY_ASSIGN(const std::int64_t previous,
                                  resolve_row(K_PREVIOUS_ROW, index, context));
                    sources.push_back(SourceSpec{previous, std::string()});
                    return sources;
                }

                SA_TRY_ASSIGN(const Value& from, row.at("from", layer));
                if (from.kind() != Kind::Array)
                {
                    SA_TRY_ASSIGN(SourceSpec source, parse_source(from, index, context));
                    sources.push_back(std::move(source));
                    return sources;
                }

                SA_TRY_ASSIGN(const std::vector<Value>& entries, from.as_array(context));
                SA_FAIL_IF(entries.empty(), StatusCode::InvalidDescription,
                           "{} is an empty list, so the layer would read nothing", context);

                sources.reserve(entries.size());
                for (const Value& entry : entries)
                {
                    SA_TRY_ASSIGN(SourceSpec source, parse_source(entry, index, context));
                    sources.push_back(std::move(source));
                }
                return sources;
            }

            /** @brief Reads one "outputs" entry of a model with @p rows rows. */
            Result<OutputSpec> parse_output(const Value& entry, std::size_t rows,
                                            const std::string& context)
            {
                SA_FAIL_IF(entry.kind() != Kind::Object, StatusCode::InvalidDescription,
                           "{} must be an object carrying 'node', but it is {}",
                           context, kind_name(entry.kind()));
                SA_TRY(entry.reject_unknown_keys(context, as_span(K_OUTPUT_KEYS)));

                OutputSpec output;
                SA_TRY_ASSIGN(const Value& stated_node, entry.at("node", context));
                SA_TRY_ASSIGN(output.node, stated_node.as_int(context + " 'node'"));
                SA_FAIL_IF(output.node < 0 || output.node >= static_cast<std::int64_t>(rows),
                           StatusCode::InvalidDescription,
                           "{} names row {}, but the model has rows 0 to {}",
                           context, output.node, rows - 1);

                if (entry.has("port"))
                {
                    SA_TRY_ASSIGN(output.port,
                                  stated_text(entry, "port", context,
                                              "publish every port of the row"));
                }

                SA_FAIL_IF(entry.has("name") && !entry.has("port"), StatusCode::InvalidDescription,
                           "{} states 'name' without 'port'; a row published whole keeps its "
                           "ports' own names", context);
                if (entry.has("name"))
                {
                    SA_TRY_ASSIGN(output.name,
                                  stated_text(entry, "name", context,
                                              "keep the port's own name"));
                }
                return output;
            }

            /** @brief Reads "outputs"; an absent key publishes every port of the last row. */
            Result<std::vector<OutputSpec>> parse_outputs(const Value& root, std::size_t rows,
                                                          std::string_view origin)
            {
                std::vector<OutputSpec> outputs;

                if (!root.has("outputs"))
                {
                    OutputSpec last;
                    last.node = static_cast<std::int64_t>(rows) - 1;
                    outputs.push_back(std::move(last));
                    return outputs;
                }

                const std::string context = "'" + std::string(origin) + "'s 'outputs'";
                SA_TRY_ASSIGN(const Value& stated, root.at("outputs", origin));
                SA_TRY_ASSIGN(const std::vector<Value>& entries, stated.as_array(context));
                SA_FAIL_IF(entries.empty(), StatusCode::InvalidDescription,
                           "{} is an empty list, so the model would publish nothing", context);

                outputs.reserve(entries.size());
                for (std::size_t i = 0; i < entries.size(); ++i)
                {
                    SA_TRY_ASSIGN(OutputSpec output,
                                  parse_output(entries[i], rows,
                                               context + " entry " + std::to_string(i)));
                    outputs.push_back(std::move(output));
                }
                return outputs;
            }

            /**
             * @brief Reads "task" of @p root; an absent key gives the null value.
             * @return The object as the file states it, checked for a string "type" only.
             */
            Result<Value> parse_task(const Value& root, std::string_view origin)
            {
                if (!root.has("task"))
                    return Value();

                const std::string context = "'" + std::string(origin) + "'s 'task'";
                SA_TRY_ASSIGN(const Value& task, root.at("task", origin));
                SA_TRY_ASSIGN(const Value& type, task.at("type", context));
                SA_TRY(type.as_string(context + " 'type'"));
                return task;
            }

            /** @brief Reads a version 2 model document into a @ref ModelSpec. */
            Result<ModelSpec> parse_current_model(const Value& root, std::string_view origin)
            {
                SA_TRY(root.reject_unknown_keys(origin, as_span(K_MODEL_KEYS)));

                ModelSpec model;
                const std::string input_context = "'" + std::string(origin) + "'s 'input'";
                SA_TRY_ASSIGN(const Value& input, root.at("input", origin));
                SA_TRY_ASSIGN(model.input_shape, parse_input_shape(input, input_context));
                SA_TRY_ASSIGN(model.input_dtype, parse_input_dtype(input, input_context));
                SA_TRY_ASSIGN(model.input_fit, parse_input_fit(input, input_context));

                const Value no_defaults = Value::object({}, {});
                const Value& defaults = root.at_or("defaults", no_defaults);
                SA_TRY(check_defaults(defaults, origin));

                SA_TRY_ASSIGN(const Value& stated_layers, root.at("layers", origin));
                SA_TRY_ASSIGN(const std::vector<Value>& layers,
                              stated_layers.as_array("'" + std::string(origin) + "'s 'layers'"));
                SA_FAIL_IF(layers.empty(), StatusCode::InvalidDescription,
                           "'{}' declares no layers, so it describes no model", origin);

                model.layers.reserve(layers.size());
                for (std::size_t i = 0; i < layers.size(); ++i)
                {
                    // Positional naming ensures error messages remain clear if type is missing.
                    const std::string context = layer_context(i, origin);

                    SA_FAIL_IF(layers[i].kind() != Kind::Object, StatusCode::InvalidDescription,
                               "{} must be an object, but it is {}",
                               context, kind_name(layers[i].kind()));

                    LayerSpec layer;
                    SA_TRY_ASSIGN(const Value& stated_type, layers[i].at("type", context));
                    SA_TRY_ASSIGN(const std::string& type,
                                  stated_type.as_string(context + " 'type'"));
                    layer.type = type;
                    SA_TRY_ASSIGN(layer.from, parse_sources(layers[i], i, context));
                    layer.attributes =
                        layer_attributes(layers[i], defaults.at_or(layer.type, no_defaults));

                    model.layers.push_back(std::move(layer));
                }

                SA_TRY(check_defaults_are_used(defaults, model.layers, origin));
                SA_TRY_ASSIGN(model.outputs, parse_outputs(root, layers.size(), origin));
                SA_TRY_ASSIGN(model.task, parse_task(root, origin));
                return model;
            }
        }

        Result<ModelSpec> parse_model(const Value& root, std::string_view origin) noexcept
        {
            SA_TRY_ASSIGN(const std::int64_t version, stated_version(root, origin));
            SA_FAIL_IF(version > K_MODEL_VERSION, StatusCode::InvalidFile,
                       "'{}' is a model file of version {}; this build reads up to version {}",
                       origin, version, K_MODEL_VERSION);

            if (version == K_FIRST_VERSION)
            {
                SA_TRY_ASSIGN(const Value upgraded, upgrade_model_v1_to_v2(root, origin));
                return parse_current_model(upgraded, origin);
            }

            return parse_current_model(root, origin);
        }

        Result<ModelSpec> load_model(const std::filesystem::path& path) noexcept
        {
            SA_TRY_ASSIGN(const Value root, parse_json_file(path));
            return parse_model(root, path.string());
        }

    } // namespace Config
} // namespace SushiAI
