/**************************************************************************/
/* image_input_none.cpp                                                   */
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
 * @file image_input_none.cpp
 * @brief Names no image and refuses an image file, for a build that does not link vision.
 * @author Mustafa Garip
 */

#include "image_input.hpp"

#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Apps
    {
        bool is_image_path(const std::filesystem::path&) noexcept
        {
            return false;
        }

        Result<Config::DataDocument> image_data_document(const std::filesystem::path&,
                                                         const Config::ModelSpec&)
        {
            SA_FAIL_IF(true, StatusCode::Unsupported, "this build links no vision, so it reads no image file");
            return Config::DataDocument();
        }
    } // namespace Apps
} // namespace SushiAI
