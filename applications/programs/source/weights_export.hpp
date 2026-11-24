/**************************************************************************/
/* weights_export.hpp                                                     */
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
 * @file weights_export.hpp
 * @brief Declares `sushiai export weights`, which writes a `.sushi` file as a foreign weight file.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/EXPORT.md
 */

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include "registries.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Writes tensors, already in the file's names and layouts, as one weight file.
         * @param folder The stem of the output file, for a container that names a folder.
         * @param state_key The key of the state dictionary, for a container that has one.
         * @return The bytes of the file, or the failure of the writer.
         */
        using WeightWriterFunction = Result<std::vector<std::byte>> (*)(
            SushiRuntime::span<const IO::ImportedTensor> tensors, std::string_view folder,
            std::string_view state_key);

        /** @brief One weight format `sushiai export weights` may name. */
        struct WeightWriterEntry
        {
            /** @brief The format name on the command line, such as "safetensors". */
            std::string_view format;

            /** @brief What writes the file of the format. */
            WeightWriterFunction write;
        };

        /** @brief Holds what the command line of one weights export states. */
        struct WeightsExportRequest
        {
            /** @brief The weight format, a row of compose_weight_writers. */
            std::string format;

            /** @brief The model file whose entries give each record its layout. */
            std::filesystem::path model;

            /** @brief The `.sushi` file whose model-state records are written. */
            std::filesystem::path weights;

            /** @brief The name map; empty states none, so every name stays. */
            std::filesystem::path map;

            /** @brief The weight file to write. */
            std::filesystem::path output;

            /** @brief Whether an output file that exists is refused or replaced. */
            IO::ExistingFiles existing = IO::ExistingFiles::Refuse;
        };

        /** @brief Holds what one weights export did. */
        struct WeightsExportReport
        {
            /** @brief The tensors written, one per state record. */
            std::size_t written = 0;

            /** @brief What the file would state that the export leaves out. */
            std::vector<std::string> notes;
        };

        /**
         * @brief Returns the weight formats a weights export may write: safetensors and torch.
         *
         * A new container adds one row here and edits no other source.
         */
        [[nodiscard]] std::vector<WeightWriterEntry> compose_weight_writers();

        /**
         * @brief Reads the model and its records, moves them to the file's layout and names and
         *        writes the weight file.
         * @param registries Supplies the rows a type of the model file may name.
         * @return The report; InvalidDescription naming every format for a format with no row;
         *         the failure of reading, of the move or of the writer. No file is written on a
         *         failure.
         */
        [[nodiscard]] Result<WeightsExportReport> export_weights_file(
            const WeightsExportRequest& request, const Registries& registries);

    } // namespace Apps
} // namespace SushiAI
