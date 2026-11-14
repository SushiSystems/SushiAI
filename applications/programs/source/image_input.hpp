/**************************************************************************/
/* image_input.hpp                                                        */
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
 * @file image_input.hpp
 * @brief Declares the data a single image file stands for in `sushiai inference`.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "image_input.hpp".
 */

#pragma once

#include <filesystem>

#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Reports whether @p path names an image file by its extension.
         *
         * A build without vision names none.
         */
        [[nodiscard]] bool is_image_path(const std::filesystem::path& path) noexcept;

        /**
         * @brief Returns the data block that reads @p image as the one sample of a run.
         * @param image The image file; its folder is the root of the block.
         * @param model Gives the class count by its task and the fitted size by its input.
         * @return The document with the `image_file` reader and a batch of one;
         *         InvalidDescription for a model whose task states no classes.
         */
        [[nodiscard]] Result<Config::DataDocument> image_data_document(
            const std::filesystem::path& image, const Config::ModelSpec& model);
    } // namespace Apps
} // namespace SushiAI
