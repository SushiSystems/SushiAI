/**************************************************************************/
/* onnx_export.cpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file onnx_export.cpp
 * @brief Defines the ONNX exporter: the walk over the rows, the refusals and the notes.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/exporters/onnx_export.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>
#include <SushiAI/interchange/onnx_writer.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model.hpp>
#include "export_walk.hpp"
#include "onnx_export_progress.hpp"
#include "onnx_row_table.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace
        {
            /** @brief Holds the operator set the file imports. */
            constexpr std::int64_t K_OPSET = 13;

            /** @brief Holds the IR version of opset 13. */
            constexpr std::int64_t K_IR_VERSION = 7;

            /** @brief Holds the rank of the model input, [H, W, C]. */
            constexpr std::size_t K_INPUT_RANK = 3;

            /** @brief Names the graph input, as SPEC section 3.1 states it. */
            constexpr std::string_view K_INPUT_NAME = "input";

            /** @brief Names the symbolic batch extent of the input. */
            constexpr std::string_view K_BATCH_NAME = "batch";

            /** @brief Names the one port a row writes its value to. */
            constexpr std::string_view K_PORT = "output";

            /** @brief Says what an ONNX value is, as a refused port names it. */
            constexpr std::string_view K_WHOLE_VALUE = "an ONNX value is a whole row";

            /** @brief Returns Unsupported when the model states what no ONNX graph can hold. */
            Status check_model(const IO::ImportedModel& imported)
            {
                const Config::ModelSpec& model = imported.model;
                SA_FAIL_IF(!model.inputs.empty(), StatusCode::Unsupported,
                           "the model reads {} named inputs, and the ONNX exporter writes one "
                           "image input",
                           model.inputs.size());
                SA_FAIL_IF(model.input_dtype != DType::Float32, StatusCode::Unsupported,
                           "the model reads an input that is not float32, and the ONNX importer "
                           "reads a float32 image");
                SA_FAIL_IF(static_cast<std::size_t>(model.input_shape.rank()) != K_INPUT_RANK,
                           StatusCode::Unsupported,
                           "the model reads an input of rank {}, and ONNX states an image "
                           "[batch, C, H, W] from [H, W, C]",
                           model.input_shape.rank());
                SA_FAIL_IF(imported.weights.empty(), StatusCode::Unsupported,
                           "the model comes with no weights, and an ONNX file holds its weights");
                return Status();
            }

            /** @brief Returns the progress of an export of @p imported before any row. */
            Detail::OnnxExportProgress start_progress(const IO::ImportedModel& imported)
            {
                const Config::ModelSpec& model = imported.model;
                const std::size_t count = model.layers.size();
                Detail::OnnxExportProgress progress;
                progress.model = &imported;
                progress.input.name = std::string(K_INPUT_NAME);
                for (std::int32_t axis = 0; axis < model.input_shape.rank(); ++axis)
                    progress.input.shape.push_back(model.input_shape[axis]);
                progress.values.resize(count);
                progress.value_names.resize(count);
                progress.readers.resize(count);
                progress.published.assign(count, false);
                for (std::size_t row = 0; row < count; ++row)
                {
                    for (const Config::SourceSpec& source : model.layers[row].from)
                    {
                        if (source.node >= 0 && static_cast<std::size_t>(source.node) < count)
                            progress.readers[static_cast<std::size_t>(source.node)].push_back(row);
                    }
                }
                for (const Config::OutputSpec& output : model.outputs)
                {
                    if (output.node < 0 || static_cast<std::size_t>(output.node) >= count)
                        continue;
                    const std::size_t row = static_cast<std::size_t>(output.node);
                    if (!progress.published[row])
                        progress.value_names[row] = output.name;
                    progress.published[row] = true;
                }
                for (std::size_t row = 0; row < count; ++row)
                {
                    if (progress.value_names[row].empty())
                        progress.value_names[row] = Detail::row_tensor_name(row, K_PORT);
                }
                return progress;
            }

            /**
             * @brief Writes every row it can, collecting the refusals in @p refusals.
             * @return A failure other than Unsupported at once; else success.
             */
            Status write_rows(Detail::OnnxExportProgress& progress,
                              std::vector<std::string>& refusals)
            {
                const std::vector<Config::LayerSpec>& layers = progress.model->model.layers;
                const Detail::RowStep write_row = [&](std::size_t row) -> Status
                {
                    const Detail::OnnxRowEntry* entry = Detail::find_onnx_row(layers[row].type);
                    SA_FAIL_IF(entry == nullptr, StatusCode::Unsupported,
                               "the ONNX importer has no row of this type");
                    Detail::OnnxRowCall call;
                    call.index = row;
                    call.spec = &layers[row];
                    const Detail::SourceTaker take = [&](std::int64_t node)
                    {
                        if (node == Config::K_MODEL_INPUT)
                        {
                            call.sources.push_back(&progress.input);
                            return true;
                        }
                        const auto& value = progress.values[static_cast<std::size_t>(node)];
                        if (!value.has_value())
                            return false;
                        call.sources.push_back(&*value);
                        return true;
                    };
                    SA_TRY_ASSIGN(const bool gathered,
                                  Detail::gather_sources(layers[row], row, K_WHOLE_VALUE, take));
                    if (!gathered)
                        return Status();
                    return entry->write(call, progress);
                };
                return Detail::write_rows(layers, write_row, Detail::RowStep(), refusals);
            }

            /** @brief Adds the graph outputs; an extra name of one row is an Identity node. */
            void write_outputs(Detail::OnnxExportProgress& progress,
                               std::vector<std::string>& refusals)
            {
                const Config::ModelSpec& model = progress.model->model;
                for (std::size_t at = 0; at < model.outputs.size(); ++at)
                {
                    const Config::OutputSpec& output = model.outputs[at];
                    if (!Detail::is_whole_value(output.port) || output.node < 0)
                    {
                        refusals.push_back("output '" + output.name +
                                           "': it publishes a port or the input, and an ONNX "
                                           "output is the whole value of a row");
                        continue;
                    }
                    const std::size_t row = static_cast<std::size_t>(output.node);
                    if (row >= progress.values.size() || !progress.values[row].has_value())
                        continue;
                    const std::string& source = progress.values[row]->name;
                    if (source != output.name)
                    {
                        Interchange::OnnxNode alias;
                        alias.name = "output." + std::to_string(at);
                        alias.op_type = "Identity";
                        alias.inputs = {source};
                        alias.outputs = {output.name};
                        progress.graph.nodes.push_back(std::move(alias));
                    }
                    Interchange::OnnxValueInfo info;
                    info.name = output.name;
                    info.element_type = Interchange::OnnxDataType::Float;
                    progress.graph.outputs.push_back(std::move(info));
                }
            }

            /** @brief Adds the graph input [batch, C, H, W] and the model fields. */
            void write_header(Detail::OnnxExportProgress& progress)
            {
                const std::vector<std::int64_t>& shape = progress.input.shape;
                Interchange::OnnxValueInfo input;
                input.name = std::string(K_INPUT_NAME);
                input.element_type = Interchange::OnnxDataType::Float;
                input.has_shape = true;
                input.dimensions = {Interchange::OnnxDimension{-1, std::string(K_BATCH_NAME)},
                                    Interchange::OnnxDimension{shape[2], {}},
                                    Interchange::OnnxDimension{shape[0], {}},
                                    Interchange::OnnxDimension{shape[1], {}}};
                progress.graph.inputs.push_back(std::move(input));
                progress.graph.name = "sushiai";
                progress.graph.ir_version = K_IR_VERSION;
                progress.graph.producer_name = "SushiAI";
                progress.graph.opsets.push_back(Interchange::OnnxOpset{"", K_OPSET});
            }
        } // namespace

        Result<Interchange::OnnxGraph> export_onnx_graph(const IO::ImportedModel& model) noexcept
        {
            SA_TRY(check_model(model));
            Detail::OnnxExportProgress progress = start_progress(model);
            write_header(progress);
            std::vector<std::string> refusals;
            SA_TRY(write_rows(progress, refusals));
            write_outputs(progress, refusals);
            SA_TRY(Detail::check_refusals(refusals, "ONNX"));
            return std::move(progress.graph);
        }

        std::vector<std::string> describe_onnx_losses(const IO::ImportedModel& model) noexcept
        {
            std::vector<std::string> notes;
            if (!model.model.task.is_null())
                notes.emplace_back("an ONNX file states no task; the task section is left out");
            if (!model.model.input_fit.empty())
                notes.push_back("an ONNX file states no image fit; the fit '" +
                                model.model.input_fit + "' is left out");
            if (model.model.input_normalise.has_value())
                notes.emplace_back("an ONNX file states no input normalisation; it is left out");
            if (!model.training.is_null())
                notes.emplace_back("an ONNX file states no training recipe; it is left out");
            if (!model.data.is_null())
                notes.emplace_back("an ONNX file states no data block; it is left out");
            notes.emplace_back("the file reads and publishes images channel-first (C, H, W), "
                               "where the model is channel-last (H, W, C)");
            return notes;
        }

        Result<IO::ExportedModel> export_onnx_files(const IO::ImportedModel& model,
                                                    std::string_view stem) noexcept
        {
            SA_TRY_ASSIGN(const Interchange::OnnxGraph graph, export_onnx_graph(model));
            SA_TRY_ASSIGN(std::vector<std::byte> bytes, Interchange::write_onnx_graph(graph));
            IO::ExportedModel exported;
            exported.files.push_back(
                IO::ExportedFile{std::string(stem) + ".onnx", std::move(bytes)});
            exported.notes = describe_onnx_losses(model);
            return exported;
        }
    } // namespace Vision::Exporters
} // namespace SushiAI
