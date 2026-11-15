/**************************************************************************/
/* ultralytics_import.cpp                                                 */
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
 * @file ultralytics_import.cpp
 * @brief Defines the Ultralytics importer over the checkpoint, the translation and the tensors.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/ultralytics_import.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/vision/importers/ultralytics_checkpoint.hpp>
#include <SushiAI/vision/importers/ultralytics_model.hpp>
#include <SushiAI/vision/importers/ultralytics_tensors.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /** @brief Returns the square input side the checkpoint trained at, or the default. */
            Result<std::int64_t> read_image_side(const UltralyticsCheckpoint& checkpoint)
            {
                const Config::Value& side =
                    checkpoint.training_arguments.at_or("imgsz", Config::Value());
                if (side.kind() != Config::Kind::Int)
                    return K_ULTRALYTICS_IMAGE_SIDE;
                return side.as_int("imgsz");
            }

            /** @brief Returns the notes the import prints: what none of the three files holds. */
            std::vector<std::string> write_notes(const UltralyticsCheckpoint& checkpoint)
            {
                std::vector<std::string> notes;
                if (!checkpoint.training_arguments.is_null())
                    notes.emplace_back("the training arguments of the checkpoint (train_args) are "
                                       "not translated; train.json states no recipe");
                if (!checkpoint.skipped.empty())
                    notes.push_back(std::to_string(checkpoint.skipped.size()) +
                                    " step counters (num_batches_tracked) are left out");
                return notes;
            }
        } // namespace

        Result<IO::ImportedModel> import_ultralytics_model(const UltralyticsCheckpoint& checkpoint,
                                                           std::string_view origin) noexcept
        {
            SA_TRY_ASSIGN(const std::int64_t side, read_image_side(checkpoint));
            SA_TRY_ASSIGN(const UltralyticsTranslation translation,
                          describe_ultralytics_model(
                              checkpoint.architecture, side,
                              SushiRuntime::span<const UltralyticsModuleClass>(
                                  checkpoint.modules.data(), checkpoint.modules.size()),
                              origin));
            SA_TRY_ASSIGN(Config::ModelSpec model,
                          Config::parse_model(translation.document, origin));
            SA_TRY_ASSIGN(std::vector<IO::ImportedTensor> weights,
                          list_ultralytics_tensors(SushiRuntime::span<const UltralyticsTensor>(
                                                       checkpoint.tensors.data(),
                                                       checkpoint.tensors.size()),
                                                   translation.tensor_rules));

            IO::ImportedModel imported;
            imported.model = std::move(model);
            imported.weights = std::move(weights);
            imported.notes = write_notes(checkpoint);
            return imported;
        }

        Result<IO::ImportedModel> import_ultralytics_files(
            SushiRuntime::span<const std::filesystem::path> files) noexcept
        {
            SA_FAIL_IF(files.size() != 1, StatusCode::InvalidDescription,
                       "the '{}' format takes one .pt file, but {} files were given",
                       K_ULTRALYTICS_FORMAT, files.size());
            SA_TRY_ASSIGN(const UltralyticsCheckpoint checkpoint,
                          load_ultralytics_checkpoint(files[0]));
            return import_ultralytics_model(checkpoint, files[0].string());
        }
    } // namespace Vision::Importers
} // namespace SushiAI
