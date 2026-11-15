/**************************************************************************/
/* ultralytics_checkpoint.hpp                                             */
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
 * @file ultralytics_checkpoint.hpp
 * @brief Declares the reader of an Ultralytics checkpoint: its description and its tensors.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/ULTRALYTICS.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds one floating tensor of a checkpoint, widened to float32. */
        struct UltralyticsTensor
        {
            /** @brief Holds its path below the model's layer list, such as "0.conv.weight". */
            std::string name;

            /** @brief Holds the groups of the convolution that owns it; 1 for any other owner. */
            std::int64_t groups = 1;

            /** @brief Holds its extents as PyTorch states them. */
            Shape shape{};

            /** @brief Holds its elements, row-major. */
            std::vector<float> values;
        };

        /** @brief Holds the class of one module of the tree, such as "SiLU" at "9.cv1.act". */
        struct UltralyticsModuleClass
        {
            std::string path; /**< Its path below the layer list. */
            std::string name; /**< Its class name, without the module that defines it. */
        };

        /** @brief Holds what an Ultralytics checkpoint states that an import reads. */
        struct UltralyticsCheckpoint
        {
            /** @brief Holds the "yaml" dictionary of the model. */
            Config::Value architecture;

            /** @brief Holds the "train_args" dictionary, or null when the file has none. */
            Config::Value training_arguments;

            /** @brief Holds every floating parameter and persistent buffer, in module order. */
            std::vector<UltralyticsTensor> tensors;

            /** @brief Lists the paths of the buffers left out: the step counters of each norm. */
            std::vector<std::string> skipped;

            /** @brief Holds the class of every module below the layer list, in module order. */
            std::vector<UltralyticsModuleClass> modules;
        };

        /**
         * @brief Reads the model of the checkpoint @p bytes: its EMA copy when it holds one,
         *        else its "model".
         * @return What TorchArchive::read returns; InvalidFile for a file that holds no
         *         Ultralytics model; Unsupported for a tensor that is not floating.
         */
        [[nodiscard]] Result<UltralyticsCheckpoint> read_ultralytics_checkpoint(
            SushiRuntime::span<const std::byte> bytes, std::string_view origin) noexcept;

        /** @brief Returns read_ultralytics_checkpoint of the file at @p path. */
        [[nodiscard]] Result<UltralyticsCheckpoint> load_ultralytics_checkpoint(
            const std::filesystem::path& path) noexcept;
    } // namespace Vision::Importers
} // namespace SushiAI
