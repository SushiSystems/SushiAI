/**************************************************************************/
/* polygon.hpp                                                            */
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
 * @file polygon.hpp
 * @brief Declares the outline a label may carry beside its box, and the rules that move it.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <vector>

#include <SushiAI/imaging/boxes.hpp>
#include <SushiAI/imaging/image.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Imaging
{
    /** @brief Holds the point count a polygon is resampled to at load (data/dataset.py:416). */
    inline constexpr std::size_t K_POLYGON_POINTS = 1000;

    /** @brief Holds one vertex, in fractions of the image or in pixels as its owner states. */
    struct Point
    {
        /** @brief Holds the horizontal coordinate. */
        float x = 0.0f;

        /** @brief Holds the vertical coordinate. */
        float y = 0.0f;
    };

    /**
     * @brief Names the outline of one object: its vertices in order, closed implicitly.
     *
     * The last vertex joins the first. A polygon of no vertex means the object has none.
     */
    using Polygon = std::vector<Point>;

    /**
     * @brief Returns the point count every polygon of one image is resampled to.
     * @param longest The largest vertex count among the image's polygons.
     * @return K_POLYGON_POINTS, or longest + 1 when longest exceeds it (data/dataset.py:420).
     */
    [[nodiscard]] std::size_t resampled_point_count(std::size_t longest) noexcept;

    /**
     * @brief Resamples a closed polygon along its edges, as the reference's resample_segments.
     *
     * The first vertex is appended to close the outline of m points. With m below count the
     * result is the m points merged in order with count - m positions spread evenly over
     * [0, m - 1]; otherwise it is count such positions. A polygon of count vertices or of
     * none is returned unchanged; a count of zero throws Error. README.md has the rest.
     */
    [[nodiscard]] Polygon resample_polygon(const Polygon& polygon, std::size_t count);

    /**
     * @brief Moves every vertex: x becomes x * scale_x + offset_x, y likewise, in float.
     *
     * The multiply is made first. Every transform of the module moves a polygon through
     * this function, with the numbers it uses for a box edge.
     */
    void map_polygon(Polygon& polygon, float scale_x, float scale_y, float offset_x,
                     float offset_y) noexcept;

    /** @brief Clamps every vertex's x to [bounds.x1, bounds.x2] and y to [bounds.y1, bounds.y2]. */
    void clamp_polygon(Polygon& polygon, const PixelBox& bounds) noexcept;

    /**
     * @brief Reports whether a point lies inside a polygon or on its outline.
     *
     * The inside is decided by the even-odd rule on a ray towards larger x, in double.
     */
    [[nodiscard]] bool is_point_inside(const Polygon& polygon, Point point) noexcept;

    /**
     * @brief Returns the box around the part of a polygon an image shows, as segment2box.
     *
     * A polygon inside [0, width] by [0, height] gives its bounds. Otherwise the box bounds
     * its vertices inside the image, the points where its edges cross the image's four
     * lines, and the image corners it contains (utils/ops.py:78-124). A polygon that shows
     * nothing, or has no vertex, gives a box of four zeros.
     */
    [[nodiscard]] PixelBox visible_polygon_box(const Polygon& polygon, ImageSize size) noexcept;
} // namespace SushiAI::Imaging
