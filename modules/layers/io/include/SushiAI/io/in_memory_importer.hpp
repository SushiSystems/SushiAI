/**************************************************************************/
/* in_memory_importer.hpp                                                 */
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
 * @file in_memory_importer.hpp
 * @brief Declares the importer rows that read no file, for a registry that needs a format.
 * @author Mustafa Garip
 * @see modules/layers/io/README.md
 */

#pragma once

#include <filesystem>
#include <string>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/importer_registry.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Returns a model whose only note is the number of files @p files holds.
         *
         * It opens no file, so a registry row over it shows which files a caller handed on.
         */
        inline Result<ImportedModel> count_files_importer(
            SushiRuntime::span<const std::filesystem::path> files)
        {
            ImportedModel imported;
            imported.notes.push_back(std::to_string(files.size()));
            return imported;
        }

        /** @brief Refuses a call without files, as an importer does for files it cannot read. */
        inline Result<ImportedModel> refuse_no_file_importer(
            SushiRuntime::span<const std::filesystem::path> files)
        {
            SA_FAIL_IF(files.empty(), StatusCode::InvalidFile, "the importer reads no file");
            return ImportedModel{};
        }

        /** @brief The row "counting", whose importer counts the files it is given. */
        inline constexpr ImporterEntry K_COUNTING_IMPORTER_ROW{"counting", &count_files_importer};

        /** @brief The row "refusing", whose importer refuses a call without files. */
        inline constexpr ImporterEntry K_REFUSING_IMPORTER_ROW{"refusing",
                                                               &refuse_no_file_importer};
    } // namespace IO
} // namespace SushiAI
