/**************************************************************************/
/* weights_export.cpp                                                     */
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
 * @file weights_export.cpp
 * @brief Defines `sushiai export weights` over the model records, the name map and the writers.
 * @author Mustafa Garip
 */

#include "weights_export.hpp"

#include <utility>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/exported_model_writer.hpp>
#include <SushiAI/io/model_files.hpp>
#include <SushiAI/io/name_map.hpp>
#include <SushiAI/io/safetensors_writer.hpp>
#include <SushiAI/io/torch_writer.hpp>
#include <SushiAI/io/weight_export.hpp>
#include <SushiAI/nn/model_state.hpp>
#include "model_state_entries.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Writes a safetensors file, which names no folder and no dictionary key. */
            Result<std::vector<std::byte>> write_safetensors_file(
                SushiRuntime::span<const IO::ImportedTensor> tensors, std::string_view,
                std::string_view)
            {
                return IO::write_safetensors(tensors);
            }

            /** @brief Writes a PyTorch state dictionary under @p folder and @p state_key. */
            Result<std::vector<std::byte>> write_torch_file(
                SushiRuntime::span<const IO::ImportedTensor> tensors, std::string_view folder,
                std::string_view state_key)
            {
                return IO::write_torch_state_dict(tensors, folder, state_key);
            }

            /**
             * @brief Returns the row of @p format.
             * @return InvalidDescription naming @p format and every format a row holds.
             */
            Result<WeightWriterEntry> find_writer(std::string_view format)
            {
                const std::vector<WeightWriterEntry> writers = compose_weight_writers();
                const WeightWriterEntry* found = nullptr;
                std::string known;
                for (const WeightWriterEntry& writer : writers)
                {
                    if (writer.format == format)
                        found = &writer;
                    known += (known.empty() ? "'" : ", '") + std::string(writer.format) + "'";
                }

                SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                           "no weight format '{}'; the formats are {}", format, known);
                return *found;
            }

            /** @brief Returns the name map of @p path, or an empty map for an empty path. */
            Result<IO::NameMap> read_map(const std::filesystem::path& path)
            {
                if (path.empty())
                    return IO::NameMap::parse("{}", "the empty name map");
                return IO::NameMap::read(path);
            }
        } // namespace

        std::vector<WeightWriterEntry> compose_weight_writers()
        {
            return {WeightWriterEntry{"safetensors", &write_safetensors_file},
                    WeightWriterEntry{"torch", &write_torch_file}};
        }

        Result<WeightsExportReport> export_weights_file(const WeightsExportRequest& request,
                                                        const Registries& registries)
        {
            SA_TRY_ASSIGN(const WeightWriterEntry writer, find_writer(request.format));

            IO::ModelFilePaths paths;
            paths.model = request.model;
            paths.weights = request.weights;
            SA_TRY_ASSIGN(const IO::ImportedModel model, IO::read_model_files(paths));
            SA_TRY_ASSIGN(const std::vector<NN::StateEntry> entries,
                          model_state_entries(registries.nodes, model.model));
            SA_TRY_ASSIGN(const IO::NameMap map, read_map(request.map));

            SA_TRY_ASSIGN(IO::ExportedWeights moved,
                          IO::export_state_weights(
                              SushiRuntime::span<const IO::ImportedTensor>(model.weights.data(),
                                                                           model.weights.size()),
                              SushiRuntime::span<const NN::StateEntry>(entries.data(),
                                                                       entries.size()),
                              map));

            const std::string stem = request.output.stem().string();
            IO::ExportedFile file;
            file.name = request.output.filename().string();
            SA_TRY_ASSIGN(file.bytes,
                          writer.write(SushiRuntime::span<const IO::ImportedTensor>(
                                           moved.tensors.data(), moved.tensors.size()),
                                       stem, map.state_key()));

            IO::ExportedModel exported;
            exported.files.push_back(std::move(file));
            const std::filesystem::path parent = request.output.parent_path();
            SA_TRY(IO::write_exported_model(exported, parent.empty() ? "." : parent,
                                            request.existing));

            WeightsExportReport report;
            report.written = moved.tensors.size();
            report.notes = std::move(moved.notes);
            return report;
        }

    } // namespace Apps
} // namespace SushiAI
