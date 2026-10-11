/**************************************************************************/
/* imported_model_writer.hpp                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file imported_model_writer.hpp
 * @brief Declares the writer of an imported model: a model file, a weight file, a training file.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT.md
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Holds the name of the model file an import writes. */
        inline constexpr std::string_view K_IMPORTED_MODEL_FILE = "model.json";

        /** @brief Holds the name of the weight file an import writes. */
        inline constexpr std::string_view K_IMPORTED_WEIGHTS_FILE = "weights.sushi";

        /** @brief Holds the name of the training file an import writes. */
        inline constexpr std::string_view K_IMPORTED_TRAINING_FILE = "train.json";

        /** @brief Holds the checkpoint the written training file asks a run to write. */
        inline constexpr std::string_view K_IMPORTED_CHECKPOINT_FILE = "trained.sushi";

        /** @brief Names what the writer does with a file that is already in the directory. */
        enum class ExistingFiles : std::uint8_t
        {
            Refuse,  /**< Nothing is written when one of the files exists. */
            Replace, /**< An existing file is overwritten. */
        };

        /** @brief Holds the files one import wrote, in the order they were written. */
        struct WrittenImport
        {
            /** @brief Holds the model file, the weight file if any, then the training file. */
            std::vector<std::filesystem::path> files;
        };

        /**
         * @brief Returns the training file of @p imported as a value.
         * @return The version, "model", "weights" when the model has weights, "data" as null,
         *         "checkpoint", then the keys of ImportedModel::training in their order; or
         *         InvalidDescription when the recipe is no object or states one of those keys.
         */
        [[nodiscard]] Result<Config::Value> compose_training_file(const ImportedModel& imported);

        /**
         * @brief Writes the model, weight and training files of @p imported into @p directory.
         * @param directory Created when absent. No file is written when a check fails.
         * @return The paths written. InvalidDescription for a model or a recipe no file can
         *         state, a tensor whose bytes differ from its record, or a name stated twice;
         *         Incompatible naming an existing file under Refuse; InvalidFile for a failed
         *         write.
         */
        [[nodiscard]] Result<WrittenImport> write_imported_model(
            const ImportedModel& imported, const std::filesystem::path& directory,
            ExistingFiles existing);

    } // namespace IO
} // namespace SushiAI
