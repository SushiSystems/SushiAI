/**************************************************************************/
/* fit_transform.cpp                                                      */
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
 * @file fit_transform.cpp
 * @brief Implements the placement of a fitted image and the map of a box back to its source.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/fit_transform.hpp>

#include <algorithm>

namespace SushiAI::Vision::Imaging
{
    FitTransform resize_transform(ImageSize source, ImageSize target) noexcept
    {
        FitTransform transform;
        transform.source = source;
        transform.scale_x = static_cast<float>(target.width) / static_cast<float>(source.width);
        transform.scale_y = static_cast<float>(target.height) / static_cast<float>(source.height);
        return transform;
    }

    FitTransform letterbox_transform(const LetterboxPlan& plan) noexcept
    {
        FitTransform transform;
        transform.source = plan.source;
        transform.scale_x = static_cast<float>(plan.ratio);
        transform.scale_y = static_cast<float>(plan.ratio);
        transform.offset_x = static_cast<float>(plan.left);
        transform.offset_y = static_cast<float>(plan.top);
        return transform;
    }

    PixelBox box_in_source(const FitTransform& transform, const PixelBox& box) noexcept
    {
        const auto width = static_cast<float>(transform.source.width);
        const auto height = static_cast<float>(transform.source.height);

        PixelBox mapped;
        mapped.x1 = std::clamp((box.x1 - transform.offset_x) / transform.scale_x, 0.0f, width);
        mapped.y1 = std::clamp((box.y1 - transform.offset_y) / transform.scale_y, 0.0f, height);
        mapped.x2 = std::clamp((box.x2 - transform.offset_x) / transform.scale_x, 0.0f, width);
        mapped.y2 = std::clamp((box.y2 - transform.offset_y) / transform.scale_y, 0.0f, height);
        return mapped;
    }
} // namespace SushiAI::Vision::Imaging
