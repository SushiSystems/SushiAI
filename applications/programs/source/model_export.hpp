/**************************************************************************/
/* model_export.hpp                                                       */
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
 * @file model_export.hpp
 * @brief Declares `sushiai export <format>`, which writes a model as another framework's files.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/EXPORT.md
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include "registries.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /** @brief Holds what the command line of one model export states. */
        struct ModelExportRequest
        {
            /** @brief The format name, the row of the exporter registry. */
            std::string format;

            /** @brief The model file. */
            std::filesystem::path model;

            /** @brief The `.sushi` weight file; empty states none. */
            std::filesystem::path weights;

            /** @brief The training file; empty states none. */
            std::filesystem::path training;

            /** @brief The directory the files are written into. */
            std::filesystem::path output;

            /** @brief The stem every written file starts with. */
            std::string name = "model";

            /** @brief Whether a file that exists is refused or replaced. */
            IO::ExistingFiles existing = IO::ExistingFiles::Refuse;
        };

        /** @brief Holds what one model export wrote and what it could not state. */
        struct ModelExportReport
        {
            /** @brief The paths written, in file order. */
            std::vector<std::filesystem::path> written;

            /** @brief What the model states that none of the written files holds. */
            std::vector<std::string> notes;
        };

        /**
         * @brief Reads the model files, writes them through the exporter row and saves the files.
         * @param registries Supplies the exporter rows, looked up by the request's format.
         * @return The report; the failure of reading the files, of the exporter or of the writer.
         *         No file is written on a failure.
         */
        [[nodiscard]] Result<ModelExportReport> export_model_files(
            const ModelExportRequest& request, const Registries& registries);

    } // namespace Apps
} // namespace SushiAI
