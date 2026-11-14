/**************************************************************************/
/* fit_transform.hpp                                                      */
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
 * @file fit_transform.hpp
 * @brief Declares how a fitted image sits in its source and the map of a box back to the source.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/letterbox.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /** @brief Holds how a source image was placed in the fitted image, per axis. */
    struct FitTransform
    {
        /** @brief Holds the size of the image before it was fitted. */
        ImageSize source{};

        /** @brief Multiplies a source column; the fitted column is that times this plus the offset. */
        float scale_x = 1.0f;

        /** @brief Multiplies a source row; the fitted row is that times this plus the offset. */
        float scale_y = 1.0f;

        /** @brief Adds to the scaled column. */
        float offset_x = 0.0f;

        /** @brief Adds to the scaled row. */
        float offset_y = 0.0f;
    };

    /**
     * @brief Returns the placement of a source that a resize stretched to @p target.
     * @pre source and target have positive extents.
     */
    [[nodiscard]] FitTransform resize_transform(ImageSize source, ImageSize target) noexcept;

    /** @brief Returns the placement apply_letterbox gave @p plan.source: one scale, the padding as offset. */
    [[nodiscard]] FitTransform letterbox_transform(const LetterboxPlan& plan) noexcept;

    /**
     * @brief Returns @p box, in pixels of the fitted image, as pixels of the source image.
     *
     * The box is clipped to the source, so a box that reaches into the padding stops at its edge.
     * @pre The scales of @p transform are positive.
     */
    [[nodiscard]] PixelBox box_in_source(const FitTransform& transform,
                                         const PixelBox& box) noexcept;
} // namespace SushiAI::Vision::Imaging
