/**************************************************************************/
/* caffe_export.cpp                                                       */
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
 * @file caffe_export.cpp
 * @brief Defines the Caffe exporter: the walk over the rows, the outputs, refusals and notes.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/exporters/caffe_export.hpp>

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/interchange/caffe_writer.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/protobuf_text.hpp>
#include "caffe_export_progress.hpp"
#include "caffe_row_table.hpp"
#include "export_walk.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace
        {
            /** @brief Holds the rank of an image input, [H, W, C]. */
            constexpr std::size_t K_IMAGE_RANK = 3;

            /** @brief Holds the rank of a vector input, [K]. */
            constexpr std::size_t K_VECTOR_RANK = 1;

            /** @brief Names the blob of the Input layer. */
            constexpr std::string_view K_INPUT_NAME = "input";

            /** @brief Says what a Caffe value is, as a refused port names it. */
            constexpr std::string_view K_WHOLE_VALUE = "a Caffe blob is a whole row";

            /** @brief Returns Unsupported when the model states what no Caffe net can hold. */
            Status check_model(const IO::ImportedModel& imported)
            {
                const Config::ModelSpec& model = imported.model;
                const std::size_t rank = static_cast<std::size_t>(model.input_shape.rank());
                SA_FAIL_IF(!model.inputs.empty(), StatusCode::Unsupported,
                           "the model reads {} named inputs, and the Caffe exporter writes one "
                           "input",
                           model.inputs.size());
                SA_FAIL_IF(model.input_dtype != DType::Float32, StatusCode::Unsupported,
                           "the model reads an input that is not float32, and a Caffe blob is "
                           "float32");
                SA_FAIL_IF(rank != K_IMAGE_RANK && rank != K_VECTOR_RANK, StatusCode::Unsupported,
                           "the model reads an input of rank {}, and the Caffe importer reads "
                           "[N, C, H, W] from [H, W, C] or [N, K] from [K]",
                           rank);
                return Status();
            }

            /** @brief Returns the progress of an export of @p imported before any row. */
            Detail::CaffeExportProgress start_progress(const IO::ImportedModel& imported)
            {
                const Config::ModelSpec& model = imported.model;
                const std::size_t count = model.layers.size();
                Detail::CaffeExportProgress progress;
                progress.model = &imported;
                progress.has_weights = !imported.weights.empty();
                progress.input.blob = std::string(K_INPUT_NAME);
                for (std::int32_t axis = 0; axis < model.input_shape.rank(); ++axis)
                    progress.input.shape.push_back(model.input_shape[axis]);
                progress.values.resize(count);
                progress.blob_names.resize(count);
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
                        progress.blob_names[row] = output.name;
                    progress.published[row] = true;
                }
                for (std::size_t row = 0; row < count; ++row)
                {
                    if (progress.blob_names[row].empty())
                        progress.blob_names[row] = "row_" + std::to_string(row);
                }
                return progress;
            }

            /** @brief Appends the Input layer of the model's input shape, batch axis 1. */
            void write_input(Detail::CaffeExportProgress& progress)
            {
                const std::vector<std::int64_t>& shape = progress.input.shape;
                std::vector<IO::TextField> dimensions = {Detail::make_int_field("dim", 1)};
                if (shape.size() == K_IMAGE_RANK)
                {
                    dimensions.push_back(Detail::make_int_field("dim", shape[2]));
                    dimensions.push_back(Detail::make_int_field("dim", shape[0]));
                    dimensions.push_back(Detail::make_int_field("dim", shape[1]));
                }
                else
                {
                    dimensions.push_back(Detail::make_int_field("dim", shape[0]));
                }
                Interchange::CaffeLayer layer;
                layer.name = std::string(K_INPUT_NAME);
                layer.type = "Input";
                layer.tops = {std::string(K_INPUT_NAME)};
                layer.parameters.fields.push_back(Detail::make_message(
                    "input_param", {Detail::make_message("shape", std::move(dimensions))}));
                progress.net.name = "sushiai";
                progress.net.layers.push_back(std::move(layer));
            }

            /**
             * @brief Writes every row it can, collecting the refusals in @p refusals.
             * @return A failure other than Unsupported at once; else success.
             */
            Status write_rows(Detail::CaffeExportProgress& progress,
                              std::vector<std::string>& refusals)
            {
                const std::vector<Config::LayerSpec>& layers = progress.model->model.layers;
                const Detail::RowStep write_row = [&](std::size_t row) -> Status
                {
                    const Detail::CaffeRowEntry* entry = Detail::find_caffe_row(layers[row].type);
                    SA_FAIL_IF(entry == nullptr, StatusCode::Unsupported,
                               "the Caffe importer has no row of this type");
                    SA_FAIL_IF(progress.readers[row].empty() && !progress.published[row],
                               StatusCode::Unsupported,
                               "no row reads it and the model does not publish it, and the Caffe "
                               "importer publishes every blob no layer reads");
                    Detail::CaffeRowCall call;
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

            /**
             * @brief Refuses each output the Caffe importer would not publish back as it is:
             *        the whole blob of a row no layer reads, once, in row order, under a name
             *        no other blob takes.
             */
            void check_outputs(const Detail::CaffeExportProgress& progress,
                               std::vector<std::string>& refusals)
            {
                const Config::ModelSpec& model = progress.model->model;
                std::set<std::string> names = {std::string(K_INPUT_NAME)};
                for (const std::string& name : progress.blob_names)
                    names.insert(name);
                if (names.size() != progress.blob_names.size() + 1)
                    refusals.emplace_back("the outputs: two blobs take one name, as an output "
                                          "named 'input' or 'row_<i>' after another row does");
                std::int64_t previous = -1;
                for (const Config::OutputSpec& output : model.outputs)
                {
                    const std::string opening = "output '" + output.name + "': ";
                    if (!Detail::is_whole_value(output.port) || output.node < 0 ||
                        static_cast<std::size_t>(output.node) >= progress.blob_names.size())
                    {
                        refusals.push_back(opening + "it publishes a port or the input, and a "
                                                     "Caffe output is the whole blob of a row");
                        continue;
                    }
                    const std::size_t row = static_cast<std::size_t>(output.node);
                    if (output.node <= previous)
                        refusals.push_back(opening + "it is not after the output before it in "
                                                     "row order, the order the Caffe importer "
                                                     "publishes in");
                    if (!progress.readers[row].empty())
                        refusals.push_back(opening + "a later row reads it, and the Caffe "
                                                     "importer publishes only the blobs no "
                                                     "layer reads");
                    previous = output.node;
                }
            }
        } // namespace

        Result<Interchange::CaffeNet> export_caffe_net(const IO::ImportedModel& model) noexcept
        {
            SA_TRY(check_model(model));
            Detail::CaffeExportProgress progress = start_progress(model);
            write_input(progress);
            std::vector<std::string> refusals;
            SA_TRY(write_rows(progress, refusals));
            check_outputs(progress, refusals);
            SA_TRY(Detail::check_refusals(refusals, "Caffe"));
            return std::move(progress.net);
        }

        std::vector<std::string> describe_caffe_losses(const IO::ImportedModel& model) noexcept
        {
            std::vector<std::string> notes;
            if (!model.model.task.is_null())
                notes.emplace_back("a Caffe net states no task; the task section is left out");
            if (!model.model.input_fit.empty())
                notes.push_back("a Caffe net states no image fit; the fit '" +
                                model.model.input_fit + "' is left out");
            if (model.model.input_normalise.has_value())
                notes.emplace_back("a Caffe net states no input normalisation; it is left out");
            if (!model.training.is_null())
                notes.emplace_back("a Caffe net states no training recipe; it is left out");
            if (!model.data.is_null())
                notes.emplace_back("a Caffe net states no data block; it is left out");
            if (model.weights.empty())
                notes.emplace_back("the model comes with no weights; no .caffemodel is written");
            notes.emplace_back("the net reads and publishes images channel-first (C, H, W), "
                               "where the model is channel-last (H, W, C)");
            return notes;
        }

        Result<IO::ExportedModel> export_caffe_files(const IO::ImportedModel& model,
                                                     std::string_view stem) noexcept
        {
            SA_TRY_ASSIGN(const Interchange::CaffeNet net, export_caffe_net(model));
            SA_TRY_ASSIGN(const std::string text, Interchange::write_caffe_prototxt(net));
            IO::ExportedModel exported;
            std::vector<std::byte> prototxt(text.size());
            for (std::size_t at = 0; at < text.size(); ++at)
                prototxt[at] = static_cast<std::byte>(text[at]);
            exported.files.push_back(
                IO::ExportedFile{std::string(stem) + ".prototxt", std::move(prototxt)});
            if (!model.weights.empty())
            {
                SA_TRY_ASSIGN(std::vector<std::byte> weights, Interchange::write_caffe_model(net));
                exported.files.push_back(
                    IO::ExportedFile{std::string(stem) + ".caffemodel", std::move(weights)});
            }
            exported.notes = describe_caffe_losses(model);
            return exported;
        }
    } // namespace Vision::Exporters
} // namespace SushiAI
