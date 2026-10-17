/**************************************************************************/
/* model_writer.cpp                                                       */
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
 * @file model_writer.cpp
 * @brief Implements the writer that turns a model specification into the text of a model file.
 * @author Mustafa Garip
 */

#include <SushiAI/config/model_writer.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Config
    {
        namespace
        {
            /** @brief Holds the "from" entry that means the model input. */
            constexpr std::string_view K_INPUT_SOURCE = "input";

            /** @brief Holds the "node" of an object entry at row 0 that means the model input. */
            constexpr std::int64_t K_PREVIOUS_ROW = -1;

            /** @brief Names an attribute read by key, for a diagnostic no call can raise. */
            constexpr std::string_view K_WRITTEN_ATTRIBUTE = "an attribute being written";

            /** @brief Names the task read by key, for a diagnostic no call can raise. */
            constexpr std::string_view K_WRITTEN_TASK = "the task being written";

            /**
             * @brief Builds the "input" object from the sample shape @p shape.
             * @param dtype Float32, which is left out, or Int32, written as "dtype".
             */
            Result<Value> input_value(const Shape& shape, DType dtype)
            {
                SA_FAIL_IF(dtype != DType::Float32 && dtype != DType::Int32,
                           StatusCode::InvalidDescription,
                           "the model's input is {}, and a model file states float32 or int32",
                           dtype_name(dtype));

                SA_FAIL_IF(shape.rank() == 0, StatusCode::InvalidDescription,
                           "the model's input shape lists no extents, and a model file states "
                           "at least one");

                std::vector<Value> extents;
                extents.reserve(static_cast<std::size_t>(shape.rank()));

                std::int64_t count = 1;
                for (std::int32_t i = 0; i < shape.rank(); ++i)
                {
                    const std::int64_t extent = shape[i];
                    SA_FAIL_IF(extent > std::numeric_limits<std::int64_t>::max() / count,
                               StatusCode::InvalidDescription,
                               "the model's input shape describes more elements than a sample "
                               "can hold; extent {} is {}", i, extent);
                    count *= extent;
                    extents.push_back(Value::integer(extent));
                }

                if (dtype == DType::Float32)
                    return Value::object({"shape"}, {Value::array(std::move(extents))});

                return Value::object({"shape", "dtype"},
                                     {Value::array(std::move(extents)),
                                      Value::string(dtype_name(dtype))});
            }

            /** @brief Reports whether row @p row reads what a row without "from" reads. */
            bool reads_previous_row(const std::vector<SourceSpec>& sources, std::size_t row)
            {
                if (sources.size() != 1 || !sources.front().port.empty())
                    return false;

                const std::int64_t previous =
                    row == 0 ? K_MODEL_INPUT : static_cast<std::int64_t>(row) - 1;
                return sources.front().node == previous;
            }

            /** @brief Builds the "from" entry that names @p source, read by row @p row. */
            Result<Value> source_value(const SourceSpec& source, std::size_t row)
            {
                if (source.node == K_MODEL_INPUT)
                {
                    if (source.port.empty())
                        return Value::string(std::string(K_INPUT_SOURCE));

                    SA_FAIL_IF(row != 0, StatusCode::InvalidDescription,
                               "layer {} reads port '{}' of the model input, and a model file "
                               "names a port of the input at row 0 only", row, source.port);
                    return Value::object({"node", "port"}, {Value::integer(K_PREVIOUS_ROW),
                                                            Value::string(source.port)});
                }

                SA_FAIL_IF(source.node < 0 || source.node >= static_cast<std::int64_t>(row),
                           StatusCode::InvalidDescription,
                           "layer {} reads row {}, which is not above it; a layer reads only "
                           "the model input and the rows above it", row, source.node);

                if (source.port.empty())
                    return Value::integer(source.node);

                return Value::object({"node", "port"}, {Value::integer(source.node),
                                                        Value::string(source.port)});
            }

            /** @brief Builds the "from" of row @p row: one entry, or a list of several. */
            Result<Value> sources_value(const std::vector<SourceSpec>& sources, std::size_t row)
            {
                SA_FAIL_IF(sources.empty(), StatusCode::InvalidDescription,
                           "layer {} reads nothing; its 'from' lists no source", row);

                if (sources.size() == 1)
                    return source_value(sources.front(), row);

                std::vector<Value> entries;
                entries.reserve(sources.size());
                for (const SourceSpec& source : sources)
                {
                    SA_TRY_ASSIGN(Value entry, source_value(source, row));
                    entries.push_back(std::move(entry));
                }
                return Value::array(std::move(entries));
            }

            /** @brief Builds row @p row: "type", "from" unless the parser fills it, the rest. */
            Result<Value> row_value(const LayerSpec& layer, std::size_t row)
            {
                SA_FAIL_IF(layer.attributes.kind() != Kind::Object,
                           StatusCode::InvalidDescription,
                           "layer {} '{}' holds attributes that are {}; they are an object, "
                           "empty for a layer without keys",
                           row, layer.type, kind_name(layer.attributes.kind()));

                const std::vector<std::string>& attributes = layer.attributes.keys();

                std::vector<std::string> keys;
                std::vector<Value> values;
                keys.reserve(attributes.size() + 2);
                values.reserve(attributes.size() + 2);

                keys.emplace_back("type");
                values.push_back(Value::string(layer.type));

                if (!reads_previous_row(layer.from, row))
                {
                    SA_TRY_ASSIGN(Value from, sources_value(layer.from, row));
                    keys.emplace_back("from");
                    values.push_back(std::move(from));
                }

                for (const std::string& key : attributes)
                {
                    SA_FAIL_IF(key == "type" || key == "from", StatusCode::InvalidDescription,
                               "layer {} '{}' holds the attribute '{}', which a model file "
                               "reads as the row's own key", row, layer.type, key);

                    keys.push_back(key);
                    values.push_back(layer.attributes.at(key, K_WRITTEN_ATTRIBUTE));
                }

                return Value::object(std::move(keys), std::move(values));
            }

            /** @brief Builds "layers" from the rows of @p layers. */
            Result<Value> layers_value(const std::vector<LayerSpec>& layers)
            {
                SA_FAIL_IF(layers.empty(), StatusCode::InvalidDescription,
                           "the model has no layers, so it describes no model");

                std::vector<Value> rows;
                rows.reserve(layers.size());
                for (std::size_t i = 0; i < layers.size(); ++i)
                {
                    SA_TRY_ASSIGN(Value row, row_value(layers[i], i));
                    rows.push_back(std::move(row));
                }
                return Value::array(std::move(rows));
            }

            /** @brief Reports whether @p outputs is what a file without "outputs" publishes. */
            bool publishes_last_row(const std::vector<OutputSpec>& outputs, std::size_t rows)
            {
                if (outputs.empty())
                    return true;

                return outputs.size() == 1 && outputs.front().port.empty()
                    && outputs.front().name.empty()
                    && outputs.front().node == static_cast<std::int64_t>(rows) - 1;
            }

            /** @brief Builds entry @p index of "outputs" for a model of @p rows rows. */
            Result<Value> output_value(const OutputSpec& output, std::size_t index,
                                       std::size_t rows)
            {
                SA_FAIL_IF(output.node < 0 || output.node >= static_cast<std::int64_t>(rows),
                           StatusCode::InvalidDescription,
                           "output {} names row {}, but the model has rows 0 to {}",
                           index, output.node, rows - 1);
                SA_FAIL_IF(output.port.empty() && !output.name.empty(),
                           StatusCode::InvalidDescription,
                           "output {} is named '{}' without a port; a row published whole "
                           "keeps its ports' own names", index, output.name);

                std::vector<std::string> keys;
                std::vector<Value> values;

                keys.emplace_back("node");
                values.push_back(Value::integer(output.node));

                if (!output.port.empty())
                {
                    keys.emplace_back("port");
                    values.push_back(Value::string(output.port));
                }

                if (!output.name.empty())
                {
                    keys.emplace_back("name");
                    values.push_back(Value::string(output.name));
                }

                return Value::object(std::move(keys), std::move(values));
            }

            /** @brief Builds "outputs" from @p outputs for a model of @p rows rows. */
            Result<Value> outputs_value(const std::vector<OutputSpec>& outputs, std::size_t rows)
            {
                std::vector<Value> entries;
                entries.reserve(outputs.size());
                for (std::size_t i = 0; i < outputs.size(); ++i)
                {
                    SA_TRY_ASSIGN(Value entry, output_value(outputs[i], i, rows));
                    entries.push_back(std::move(entry));
                }
                return Value::array(std::move(entries));
            }

            /** @brief Checks that @p task is an object that states "type" as a string. */
            Status check_task(const Value& task)
            {
                SA_FAIL_IF(task.kind() != Kind::Object, StatusCode::InvalidDescription,
                           "the model's task is {}; a model file states it as an object "
                           "carrying 'type'", kind_name(task.kind()));
                SA_FAIL_IF(!task.has("type"), StatusCode::InvalidDescription,
                           "the model's task states no 'type', and a model file names the "
                           "task its object describes");

                const Value& type = task.at("type", K_WRITTEN_TASK);
                SA_FAIL_IF(type.kind() != Kind::String, StatusCode::InvalidDescription,
                           "the model's task states a 'type' that is {}; it is a string",
                           kind_name(type.kind()));
                SA_FAIL_IF(type.as_string(K_WRITTEN_TASK).empty(),
                           StatusCode::InvalidDescription,
                           "the model's task states a 'type' that is an empty string; it "
                           "names the task the rest of the object describes");
                return Status();
            }

            /** @brief Builds the document of @p model, its keys in the order a model file has. */
            Result<Value> model_value(const ModelSpec& model)
            {
                std::vector<std::string> keys;
                std::vector<Value> values;

                keys.emplace_back("format_version");
                values.push_back(Value::integer(K_MODEL_VERSION));

                SA_TRY_ASSIGN(Value input, input_value(model.input_shape, model.input_dtype));
                keys.emplace_back("input");
                values.push_back(std::move(input));

                SA_TRY_ASSIGN(Value layers, layers_value(model.layers));
                keys.emplace_back("layers");
                values.push_back(std::move(layers));

                if (!publishes_last_row(model.outputs, model.layers.size()))
                {
                    SA_TRY_ASSIGN(Value outputs,
                                  outputs_value(model.outputs, model.layers.size()));
                    keys.emplace_back("outputs");
                    values.push_back(std::move(outputs));
                }

                if (!model.task.is_null())
                {
                    SA_TRY(check_task(model.task));
                    keys.emplace_back("task");
                    values.push_back(model.task);
                }

                return Value::object(std::move(keys), std::move(values));
            }
        }

        Result<std::string> write_model(const ModelSpec& model) noexcept
        {
            try
            {
                SA_TRY_ASSIGN(const Value document, model_value(model));
                return write_json(document);
            }
            catch (const std::bad_alloc&)
            {
                return Status::from_message(StatusCode::OutOfMemory, []()
                {
                    return std::string("the host heap refused the document of a model file");
                });
            }
        }

    } // namespace Config
} // namespace SushiAI
