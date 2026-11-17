/**************************************************************************/
/* weights_import.hpp                                                     */
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
 * @file weights_import.hpp
 * @brief Declares `sushiai import weights`, which writes a foreign weight file as `.sushi`.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <filesystem>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <SushiAI/io/weight_source.hpp>
#include "registries.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /** @brief Holds what the command line of one weights import states. */
        struct WeightsImportRequest
        {
            /** @brief The model file whose state records the weights must fill. */
            std::filesystem::path model;

            /** @brief The weight file of the other framework. */
            std::filesystem::path weights;

            /** @brief The name map: renames, drops and the key of the state dictionary. */
            std::filesystem::path map;

            /** @brief The `.sushi` file to write. */
            std::filesystem::path output;

            /** @brief Whether an output file that exists is refused or replaced. */
            IO::ExistingFiles existing = IO::ExistingFiles::Refuse;
        };

        /** @brief Holds what one weights import did. */
        struct WeightsImportReport
        {
            /** @brief The tensors written, one per state record of the model. */
            std::size_t written = 0;

            /** @brief The tensors of the file the name map dropped. */
            std::size_t dropped = 0;
        };

        /**
         * @brief Returns the containers a weight file may be in: the PyTorch zip and safetensors.
         *
         * A new container adds one row here and edits no other source.
         */
        [[nodiscard]] Result<IO::WeightSourceRegistry> compose_weight_sources();

        /**
         * @brief Reads the weight file, renames its tensors by the map and writes the `.sushi` file.
         * @param registries Supplies the rows a type of the model file may name.
         * @return The report; the failure of reading the model file, the map or the weights; or
         *         Incompatible naming every missing, extra and misshapen tensor at once. No file
         *         is written on a failure.
         */
        [[nodiscard]] Result<WeightsImportReport> import_weights_file(
            const WeightsImportRequest& request, const Registries& registries);

    } // namespace Apps
} // namespace SushiAI
