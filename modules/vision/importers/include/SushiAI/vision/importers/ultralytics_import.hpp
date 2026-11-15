/**************************************************************************/
/* ultralytics_import.hpp                                                 */
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
 * @file ultralytics_import.hpp
 * @brief Declares the Ultralytics importer: a checkpoint in, an imported model out.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/ULTRALYTICS.md
 */

#pragma once

#include <filesystem>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/vision/importers/ultralytics_checkpoint.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds the format name `sushiai import` gives the Ultralytics importer. */
        inline constexpr std::string_view K_ULTRALYTICS_FORMAT = "ultralytics";

        /**
         * @brief Returns the model, the weights and the notes of @p checkpoint.
         * @return What translate_ultralytics_model and list_ultralytics_tensors return.
         */
        [[nodiscard]] Result<IO::ImportedModel> import_ultralytics_model(
            const UltralyticsCheckpoint& checkpoint, std::string_view origin) noexcept;

        /**
         * @brief Reads the one `.pt` file of @p files into an imported model.
         * @return InvalidDescription for a count of files other than one; else what
         *         load_ultralytics_checkpoint and import_ultralytics_model return.
         */
        [[nodiscard]] Result<IO::ImportedModel> import_ultralytics_files(
            SushiRuntime::span<const std::filesystem::path> files) noexcept;
    } // namespace Vision::Importers
} // namespace SushiAI
