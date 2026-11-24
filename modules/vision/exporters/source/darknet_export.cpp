/**************************************************************************/
/* darknet_export.cpp                                                     */
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
 * @file darknet_export.cpp
 * @brief Defines the Darknet exporter: the walk over the rows, the heads, the refusals and the
 *        notes.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/exporters/darknet_export.hpp>

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
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include "darknet_export_progress.hpp"
#include "darknet_files.hpp"
#include "darknet_head_sections.hpp"
#include "darknet_row_table.hpp"
#include "darknet_training_lines.hpp"
#include "export_walk.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace
        {
            using Detect::Anchor::AnchorTask;

            /** @brief Holds the rank of the model input, [H, W, C]. */
            constexpr std::size_t K_INPUT_RANK = 3;

            /** @brief Says what a Darknet value is, as a refused port names it. */
            constexpr std::string_view K_WHOLE_VALUE = "a Darknet layer is a whole value";

            /** @brief Returns Unsupported when the model's input is not a Darknet image. */
            Status check_model(const Config::ModelSpec& model)
            {
                SA_FAIL_IF(!model.inputs.empty(), StatusCode::Unsupported,
                           "the model reads {} named inputs, and a .cfg states one image input",
                           model.inputs.size());
                SA_FAIL_IF(model.input_dtype != DType::Float32, StatusCode::Unsupported,
                           "the model reads an input that is not float32, and Darknet reads a "
                           "float32 image");
                SA_FAIL_IF(static_cast<std::size_t>(model.input_shape.rank()) != K_INPUT_RANK,
                           StatusCode::Unsupported,
                           "the model reads an input of rank {}, and [net] states an image "
                           "[H, W, C]",
                           model.input_shape.rank());
                SA_FAIL_IF(!model.input_fit.empty() && model.input_fit != Config::K_FIT_RESIZE &&
                               model.input_fit != Config::K_FIT_DARKNET_LETTERBOX,
                           StatusCode::Unsupported,
                           "the model fits its image by '{}', and letter_box states '{}' or "
                           "'{}' alone",
                           model.input_fit, Config::K_FIT_RESIZE,
                           Config::K_FIT_DARKNET_LETTERBOX);
                return Status();
            }

            /** @brief Returns the progress of an export of @p imported before any row. */
            Detail::DarknetExportProgress start_progress(const IO::ImportedModel& imported)
            {
                const Config::ModelSpec& model = imported.model;
                const std::size_t count = model.layers.size();
                Detail::DarknetExportProgress progress;
                progress.model = &imported;
                for (std::int32_t axis = 0; axis < model.input_shape.rank(); ++axis)
                    progress.input.shape.push_back(model.input_shape[axis]);
                progress.values.resize(count);
                progress.readers.resize(count);
                progress.published.assign(count, false);
                progress.absorbed.assign(count, false);
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
                    if (output.node >= 0 && static_cast<std::size_t>(output.node) < count)
                        progress.published[static_cast<std::size_t>(output.node)] = true;
                }
                return progress;
            }

            /** @brief Holds the heads of the export and the next one to write. */
            struct HeadWriter
            {
                const AnchorTask* task = nullptr;                      /**< The heads. */
                const std::vector<Detail::DarknetOptionText>* training = nullptr; /**< Lines. */
                std::size_t next = 0; /**< The first head not yet written. */
            };

            /**
             * @brief Writes a [yolo] section for each next head whose row has a value by
             *        @p row; a [route] first when the head's row is not the last section's.
             */
            Status write_heads(Detail::DarknetExportProgress& progress, HeadWriter& heads,
                               std::size_t row)
            {
                const Config::ModelSpec& model = progress.model->model;
                while (heads.next < heads.task->heads.size())
                {
                    const std::int64_t node = model.outputs[heads.next].node;
                    const auto& value = progress.values[static_cast<std::size_t>(node)];
                    if (static_cast<std::size_t>(node) > row || !value.has_value())
                        return Status();
                    const std::int64_t expected =
                        Detect::Anchor::count_head_channels(heads.task->heads[heads.next].options,
                                                            heads.task->layout,
                                                            static_cast<std::int64_t>(heads.next));
                    SA_FAIL_IF(value->shape[2] != expected, StatusCode::InvalidDescription,
                               "head {} reads row {} of {} channels, and its anchors and classes "
                               "need {}",
                               heads.next, node, value->shape[2], expected);
                    SA_TRY(Detail::follow_row(progress, node));
                    Detail::add_section(progress, "yolo",
                                        Detail::describe_head(*heads.task, heads.next,
                                                              progress.input.shape,
                                                              *heads.training));
                    ++heads.next;
                }
                return Status();
            }

            /**
             * @brief Writes every row it can and the heads after them, collecting the refusals
             *        in @p refusals.
             * @return A failure other than Unsupported at once; else success.
             */
            Status write_rows(Detail::DarknetExportProgress& progress, HeadWriter& heads,
                              std::vector<std::string>& refusals)
            {
                const std::vector<Config::LayerSpec>& layers = progress.model->model.layers;
                const Detail::RowStep write_row = [&](std::size_t row) -> Status
                {
                    if (progress.absorbed[row])
                        return Status();
                    const Detail::DarknetRowEntry* entry =
                        Detail::find_darknet_row(layers[row].type);
                    SA_FAIL_IF(entry == nullptr, StatusCode::Unsupported,
                               "no Darknet section the importer reads gives this row back here");
                    Detail::DarknetRowCall call;
                    call.index = row;
                    call.spec = &layers[row];
                    const Detail::SourceTaker take = [&](std::int64_t node)
                    {
                        if (node == Config::K_MODEL_INPUT)
                        {
                            call.sources.push_back(Config::K_MODEL_INPUT);
                            call.values.push_back(&progress.input);
                            return true;
                        }
                        const auto& value = progress.values[static_cast<std::size_t>(node)];
                        if (!value.has_value())
                            return false;
                        call.sources.push_back(node);
                        call.values.push_back(&*value);
                        return true;
                    };
                    SA_TRY_ASSIGN(const bool gathered,
                                  Detail::gather_sources(layers[row], row, K_WHOLE_VALUE, take));
                    if (!gathered)
                        return Status();
                    return entry->write(call, progress);
                };
                const Detail::RowStep after_row = [&](std::size_t row)
                {
                    return write_heads(progress, heads, row);
                };
                return Detail::write_rows(layers, write_row, after_row, refusals);
            }

            /** @brief Returns the lines of [net]: the input, its fit and @p training. */
            std::vector<Detail::DarknetOptionText> describe_network(
                const Config::ModelSpec& model,
                const std::vector<Detail::DarknetOptionText>& training)
            {
                const bool letterbox = model.input_fit == Config::K_FIT_DARKNET_LETTERBOX;
                std::vector<Detail::DarknetOptionText> lines = {
                    {"width", Detail::spell_integer(model.input_shape[1])},
                    {"height", Detail::spell_integer(model.input_shape[0])},
                    {"channels", Detail::spell_integer(model.input_shape[2])},
                    {"letter_box", letterbox ? "1" : "0"}};
                lines.insert(lines.end(), training.begin(), training.end());
                return lines;
            }

            /** @brief Returns one note for each value of @p model no .cfg line holds. */
            std::vector<std::string> describe_losses(const Config::ModelSpec& model,
                                                     std::vector<std::string> training_notes)
            {
                std::vector<std::string> notes;
                if (model.input_fit.empty())
                    notes.emplace_back("the model states no image fit; the .cfg states "
                                       "letter_box=0, which the importer reads as 'resize'");
                if (model.input_normalise.has_value())
                    notes.emplace_back("a .cfg states no input normalisation; it is left out");
                for (std::string& note : training_notes)
                    notes.push_back(std::move(note));
                return notes;
            }
        } // namespace

        Result<IO::ExportedModel> export_darknet_files(const IO::ImportedModel& model,
                                                       std::string_view stem) noexcept
        {
            SA_TRY(check_model(model.model));
            SA_TRY_ASSIGN(const AnchorTask task, Detail::read_darknet_heads(model.model));
            SA_TRY_ASSIGN(Detail::DarknetTrainingLines training,
                          Detail::describe_darknet_training(model));
            Detail::DarknetExportProgress progress = start_progress(model);
            HeadWriter heads{&task, &training.head, 0};
            std::vector<std::string> refusals;
            SA_TRY(write_rows(progress, heads, refusals));
            SA_TRY(Detail::check_refusals(refusals, "Darknet"));

            const std::string text = Detail::write_darknet_cfg(
                describe_network(model.model, training.network), progress.sections);
            IO::ExportedModel exported;
            std::vector<std::byte> cfg(text.size());
            for (std::size_t at = 0; at < text.size(); ++at)
                cfg[at] = static_cast<std::byte>(text[at]);
            exported.files.push_back(IO::ExportedFile{std::string(stem) + ".cfg", std::move(cfg)});
            if (!model.weights.empty())
                exported.files.push_back(
                    IO::ExportedFile{std::string(stem) + ".weights",
                                     Detail::write_darknet_weights(progress.convolutions)});
            exported.notes = describe_losses(model.model, std::move(training.notes));
            return exported;
        }
    } // namespace Vision::Exporters
} // namespace SushiAI
