/**************************************************************************/
/* darknet_import.hpp                                                     */
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
 * @file darknet_import.hpp
 * @brief Declares the Darknet importer: a .cfg and its .weights as an imported model.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <filesystem>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds the format name the Darknet importer is registered under. */
        inline constexpr std::string_view K_DARKNET_FORMAT = "darknet";

        /**
         * @brief Assembles a translated .cfg and its weights into an imported model, in memory.
         * @param model The translation of @p config, in the form @p weights are in.
         * @param weights The convolutions of the .weights file, or null for a model without.
         * @return The rows with an "anchor_detection" task, the tensors by state name, the
         *         recipe and its notes; or the failure of read_darknet_training,
         *         write_darknet_recipe, write_anchor_task or list_darknet_tensors.
         */
        [[nodiscard]] Result<IO::ImportedModel> import_darknet_model(
            const DarknetConfig& config, const DarknetModel& model,
            const DarknetWeights* weights);

        /**
         * @brief Reads a .cfg, and a .weights when one is given, into an imported model.
         * @param files The .cfg, then optionally the .weights; any other count is refused.
         * @return The model with every normalised convolution kept beside its batch-norm row;
         *         InvalidDescription for another count of files; or the failure of the load,
         *         of the translation or of import_darknet_model.
         */
        [[nodiscard]] Result<IO::ImportedModel> import_darknet_files(
            SushiRuntime::span<const std::filesystem::path> files);

    } // namespace Vision::Importers
} // namespace SushiAI
