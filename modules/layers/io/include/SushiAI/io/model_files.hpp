/**************************************************************************/
/* model_files.hpp                                                        */
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
 * @file model_files.hpp
 * @brief Declares the reader of a model's model, weight and training files into an ImportedModel.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <filesystem>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Names the files of one SushiAI model; an empty path states no such file. */
        struct ModelFilePaths
        {
            /** @brief Holds the model file. */
            std::filesystem::path model;

            /** @brief Holds the `.sushi` weight file, or nothing. */
            std::filesystem::path weights;

            /** @brief Holds the training file, or nothing. */
            std::filesystem::path training;
        };

        /**
         * @brief Reads the files @p paths names into an ImportedModel, the input of an exporter.
         * Weight-file model-state records become the weights (run state is skipped); the training
         * file's "data" block becomes the data and its keys outside K_TRAINING_WRITER_KEYS the
         * recipe, so write_imported_model writes the same three files back.
         * @return The model; what load_model returns for the model file; FileNotFound or
         *         InvalidFile naming a weight or training file that cannot be read.
         */
        [[nodiscard]] Result<ImportedModel> read_model_files(const ModelFilePaths& paths) noexcept;

    } // namespace IO
} // namespace SushiAI
