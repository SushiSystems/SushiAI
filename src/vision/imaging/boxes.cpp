/**************************************************************************/
/* boxes.cpp                                                              */
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
 * @file boxes.cpp
 * @brief Implements the two box formats' conversions and the box rules of the transforms.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/boxes.hpp>

#include <algorithm>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Clamps a coordinate to [0, upper]. */
        [[nodiscard]] float clamp_coordinate(float value, float upper) noexcept
        {
            return std::min(std::max(value, 0.0f), upper);
        }
    } // namespace

    PixelBox to_pixel_box(const NormalizedBox& box, ImageSize size) noexcept
    {
        const float image_width = static_cast<float>(size.width);
        const float image_height = static_cast<float>(size.height);
        const float half_width = box.width / 2.0f;
        const float half_height = box.height / 2.0f;

        PixelBox pixels;
        pixels.x1 = (box.x_center - half_width) * image_width;
        pixels.y1 = (box.y_center - half_height) * image_height;
        pixels.x2 = (box.x_center + half_width) * image_width;
        pixels.y2 = (box.y_center + half_height) * image_height;
        return pixels;
    }

    NormalizedBox to_normalized_box(const PixelBox& box, ImageSize size) noexcept
    {
        const float image_width = static_cast<float>(size.width);
        const float image_height = static_cast<float>(size.height);

        NormalizedBox normalized;
        normalized.x_center = ((box.x1 + box.x2) / 2.0f) / image_width;
        normalized.y_center = ((box.y1 + box.y2) / 2.0f) / image_height;
        normalized.width = (box.x2 - box.x1) / image_width;
        normalized.height = (box.y2 - box.y1) / image_height;
        return normalized;
    }

    PixelBox scale_box(const PixelBox& box, float scale_x, float scale_y) noexcept
    {
        PixelBox scaled;
        scaled.x1 = box.x1 * scale_x;
        scaled.y1 = box.y1 * scale_y;
        scaled.x2 = box.x2 * scale_x;
        scaled.y2 = box.y2 * scale_y;
        return scaled;
    }

    PixelBox translate_box(const PixelBox& box, float offset_x, float offset_y) noexcept
    {
        PixelBox moved;
        moved.x1 = box.x1 + offset_x;
        moved.y1 = box.y1 + offset_y;
        moved.x2 = box.x2 + offset_x;
        moved.y2 = box.y2 + offset_y;
        return moved;
    }

    PixelBox clip_box(const PixelBox& box, ImageSize size) noexcept
    {
        const float image_width = static_cast<float>(size.width);
        const float image_height = static_cast<float>(size.height);

        PixelBox clipped;
        clipped.x1 = clamp_coordinate(box.x1, image_width);
        clipped.y1 = clamp_coordinate(box.y1, image_height);
        clipped.x2 = clamp_coordinate(box.x2, image_width);
        clipped.y2 = clamp_coordinate(box.y2, image_height);
        return clipped;
    }

    float box_area(const PixelBox& box) noexcept
    {
        return (box.x2 - box.x1) * (box.y2 - box.y1);
    }

    bool is_box_candidate(const PixelBox& before, const PixelBox& after,
                          const CandidateFilter& filter) noexcept
    {
        const float width_before = before.x2 - before.x1;
        const float height_before = before.y2 - before.y1;
        const float width_after = after.x2 - after.x1;
        const float height_after = after.y2 - after.y1;

        const float aspect_ratio = std::max(width_after / (height_after + filter.epsilon),
                                            height_after / (width_after + filter.epsilon));
        const float area_ratio =
            width_after * height_after / (width_before * height_before + filter.epsilon);

        return width_after > filter.min_side && height_after > filter.min_side &&
               area_ratio > filter.min_area_ratio && aspect_ratio < filter.max_aspect_ratio;
    }
} // namespace SushiAI::Vision::Imaging
