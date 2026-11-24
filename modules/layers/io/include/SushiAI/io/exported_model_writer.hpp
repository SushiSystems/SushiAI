/**************************************************************************/
/* exported_model_writer.hpp                                              */
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
 * @file exported_model_writer.hpp
 * @brief Declares the writer that puts the files of an export into a directory.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <filesystem>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Writes every file of @p exported into @p directory by partial file and rename.
         * @param directory Created when absent. No file is written when a check fails.
         * @param existing Refuse stops before any write when one of the files exists.
         * @return The paths written, in file order. InvalidDescription for an empty export or a
         *         name empty, with a directory part or twice; Incompatible naming an existing
         *         file under Refuse; InvalidFile for a failed write.
         */
        [[nodiscard]] Result<std::vector<std::filesystem::path>> write_exported_model(
            const ExportedModel& exported, const std::filesystem::path& directory,
            ExistingFiles existing) noexcept;

    } // namespace IO
} // namespace SushiAI
