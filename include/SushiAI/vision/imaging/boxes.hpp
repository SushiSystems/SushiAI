/**************************************************************************/
/* boxes.hpp                                                              */
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
 * @file boxes.hpp
 * @brief Declares the two box formats, their conversions and the box rules of the transforms.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /**
     * @brief Holds a box as a label file states it: centre and extent as fractions of the image.
     *
     * A value of 1 is the full width for x_center and width, and the full height for
     * y_center and height.
     */
    struct NormalizedBox
    {
        /** @brief Holds the centre's x as a fraction of the image width. */
        float x_center = 0.0f;

        /** @brief Holds the centre's y as a fraction of the image height. */
        float y_center = 0.0f;

        /** @brief Holds the box width as a fraction of the image width. */
        float width = 0.0f;

        /** @brief Holds the box height as a fraction of the image height. */
        float height = 0.0f;
    };

    /**
     * @brief Holds a box as the transforms move it: two corners in pixels.
     *
     * Coordinates are continuous. Pixel (x, y) of an image is the point (x, y), the image
     * spans [0, width] by [0, height], and x1 <= x2 and y1 <= y2 hold for every box a
     * function of this module returns.
     */
    struct PixelBox
    {
        /** @brief Holds the left edge. */
        float x1 = 0.0f;

        /** @brief Holds the top edge. */
        float y1 = 0.0f;

        /** @brief Holds the right edge. */
        float x2 = 0.0f;

        /** @brief Holds the bottom edge. */
        float y2 = 0.0f;
    };

    /**
     * @brief Holds the thresholds that decide whether a warped box is still worth training on.
     *
     * The defaults are the reference's box_candidates defaults (data/augment.py:1413-1416)
     * with the area ratio it passes for an object without a polygon (data/augment.py:1255).
     */
    struct CandidateFilter
    {
        /** @brief Holds the side length, in pixels, that both sides must exceed. */
        float min_side = 2.0f;

        /** @brief Holds the aspect ratio the box must stay below. */
        float max_aspect_ratio = 100.0f;

        /** @brief Holds the fraction of its area before the clip that the box must exceed. */
        float min_area_ratio = 0.10f;

        /** @brief Holds the term added to each denominator. */
        float epsilon = 1e-16f;
    };

    /** @brief Holds the area ratio of an object that has a polygon (data/augment.py:1255). */
    inline constexpr float K_POLYGON_MIN_AREA_RATIO = 0.01f;

    /** @brief Returns the default filter with min_area_ratio at K_POLYGON_MIN_AREA_RATIO. */
    [[nodiscard]] constexpr CandidateFilter make_polygon_filter() noexcept
    {
        CandidateFilter filter;
        filter.min_area_ratio = K_POLYGON_MIN_AREA_RATIO;
        return filter;
    }

    /**
     * @brief Converts a file box to pixel corners on an image of the given size.
     *
     * In float, in this order: x1 = (x_center - width / 2) * size.width and
     * x2 = (x_center + width / 2) * size.width; y1 and y2 likewise with the heights. Nothing
     * is clipped.
     */
    [[nodiscard]] PixelBox to_pixel_box(const NormalizedBox& box, ImageSize size) noexcept;

    /**
     * @brief Converts pixel corners to a file box on an image of the given size.
     *
     * In float, in this order: x_center = ((x1 + x2) / 2) / size.width and
     * width = (x2 - x1) / size.width; y_center and height likewise with size.height.
     */
    [[nodiscard]] NormalizedBox to_normalized_box(const PixelBox& box, ImageSize size) noexcept;

    /** @brief Multiplies x1 and x2 by scale_x and y1 and y2 by scale_y, in float. */
    [[nodiscard]] PixelBox scale_box(const PixelBox& box, float scale_x, float scale_y) noexcept;

    /** @brief Adds offset_x to x1 and x2 and offset_y to y1 and y2, in float. */
    [[nodiscard]] PixelBox translate_box(const PixelBox& box, float offset_x,
                                         float offset_y) noexcept;

    /** @brief Clamps x1 and x2 to [0, size.width] and y1 and y2 to [0, size.height]. */
    [[nodiscard]] PixelBox clip_box(const PixelBox& box, ImageSize size) noexcept;

    /** @brief Returns (x2 - x1) * (y2 - y1) in float. */
    [[nodiscard]] float box_area(const PixelBox& box) noexcept;

    /**
     * @brief Reports whether a warped and clipped box passes the reference's candidate test.
     *
     * With sides w1, h1 of before and w2, h2 of after, all hold strictly: w2 > min_side,
     * h2 > min_side, w2 * h2 / (w1 * h1 + epsilon) > min_area_ratio, and
     * max(w2 / (h2 + epsilon), h2 / (w2 + epsilon)) < max_aspect_ratio.
     * @param before The box before the warp, multiplied by the warp's scale and not clipped.
     */
    [[nodiscard]] bool is_box_candidate(const PixelBox& before, const PixelBox& after,
                                        const CandidateFilter& filter) noexcept;
} // namespace SushiAI::Vision::Imaging
