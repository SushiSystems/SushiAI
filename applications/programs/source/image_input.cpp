/**************************************************************************/
/* image_input.cpp                                                        */
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
 * @file image_input.cpp
 * @brief Builds the data block that reads one image file, for a build with vision.
 * @author Mustafa Garip
 */

#include "image_input.hpp"

#include <string>
#include <utility>
#include <vector>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/image_extensions.hpp>

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Names the reader that serves one image file. */
            constexpr const char* K_IMAGE_FILE_READER = "image_file";

            /** @brief Names the key of the task section that counts the classes. */
            constexpr const char* K_CLASSES_KEY = "classes";
        } // namespace

        bool is_image_path(const std::filesystem::path& path) noexcept
        {
            return Vision::Datasets::has_image_extension(path);
        }

        Result<Config::DataDocument> image_data_document(const std::filesystem::path& image,
                                                         const Config::ModelSpec& model)
        {
            SA_FAIL_IF(model.input_shape.rank() != 3, StatusCode::InvalidDescription,
                       "an image file feeds a model whose input is [height, width, channels]; "
                       "this model's input is {}",
                       model.input_shape.to_string());
            SA_FAIL_IF(model.task.is_null() || !model.task.has(K_CLASSES_KEY),
                       StatusCode::InvalidDescription,
                       "an image file feeds a model whose task states its 'classes'");
            SA_TRY_ASSIGN(const Config::Value& stated,
                          model.task.at(K_CLASSES_KEY, "the task section"));
            SA_TRY_ASSIGN(const std::int64_t classes, stated.as_int("the task classes"));

            const std::filesystem::path folder = image.parent_path();
            std::vector<std::string> keys = {"reader",  "task",       "root",       "file",
                                             "batch_size", "input",   "max_objects", "classes"};
            std::vector<Config::Value> values;
            values.push_back(Config::Value::string(K_IMAGE_FILE_READER));
            values.push_back(Config::Value::string("detection"));
            values.push_back(Config::Value::string(
                (folder.empty() ? std::filesystem::path(".") : folder).generic_string()));
            values.push_back(Config::Value::string(image.filename().generic_string()));
            values.push_back(Config::Value::integer(1));
            values.push_back(Config::Value::array(
                {Config::Value::integer(model.input_shape[0]),
                 Config::Value::integer(model.input_shape[1])}));
            values.push_back(Config::Value::integer(1));
            values.push_back(Config::Value::integer(classes));

            Config::DataDocument document;
            document.data.type = std::string(Config::K_READER_DATA_TYPE);
            document.data.attributes = Config::Value::object(std::move(keys), std::move(values));
            return document;
        }
    } // namespace Apps
} // namespace SushiAI
