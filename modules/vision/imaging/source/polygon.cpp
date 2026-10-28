/**************************************************************************/
/* polygon.cpp                                                            */
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
 * @file polygon.cpp
 * @brief Implements the resampling, the mapping and the visible box of a label's polygon.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/polygon.hpp>

#include <algorithm>
#include <initializer_list>

#include <SushiAI/failure/check.hpp>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Holds count positions spread evenly over [0, last], as numpy.linspace. */
        struct EvenPositions
        {
            /** @brief Counts the positions. */
            std::size_t count = 0;

            /** @brief Holds the last position, which the final index returns exactly. */
            double last = 0.0;

            /** @brief Returns position index: index * last / (count - 1), or last at the end. */
            [[nodiscard]] double at(std::size_t index) const noexcept
            {
                if (count < 2)
                    return 0.0;

                if (index + 1 == count)
                    return last;

                const double step = last / static_cast<double>(count - 1);
                return static_cast<double>(index) * step;
            }
        };

        /**
         * @brief Returns the point of a closed polygon at a position along its vertices.
         *
         * Position j is vertex j and position size() is the first vertex again; between two
         * vertices the point is slope * (position - j) + vertex j in double, as numpy.interp.
         * @param position In [0, polygon.size()].
         */
        [[nodiscard]] Point point_at(const Polygon& polygon, double position) noexcept
        {
            const std::size_t count = polygon.size();
            const std::size_t lower = std::min(static_cast<std::size_t>(position), count);
            const Point& first = polygon[lower % count];

            if (lower == count || static_cast<double>(lower) == position)
                return first;

            const Point& second = polygon[(lower + 1) % count];
            const double offset = position - static_cast<double>(lower);
            const double first_x = first.x;
            const double first_y = first.y;
            const double slope_x = static_cast<double>(second.x) - first_x;
            const double slope_y = static_cast<double>(second.y) - first_y;

            return Point{static_cast<float>(slope_x * offset + first_x),
                         static_cast<float>(slope_y * offset + first_y)};
        }

        /** @brief Holds the bounds of the points seen so far. */
        struct Extent
        {
            /** @brief Tells whether at least one point was included. */
            bool has_point = false;

            /** @brief Holds the bounds; meaningless until has_point. */
            PixelBox box{};

            /** @brief Widens the bounds to hold one more point. */
            void include(float x, float y) noexcept
            {
                if (!has_point)
                {
                    box = PixelBox{x, y, x, y};
                    has_point = true;
                    return;
                }

                box.x1 = std::min(box.x1, x);
                box.y1 = std::min(box.y1, y);
                box.x2 = std::max(box.x2, x);
                box.y2 = std::max(box.y2, y);
            }
        };

        /** @brief Returns the bounds of every vertex of a polygon that has at least one. */
        [[nodiscard]] PixelBox bounds_of(const Polygon& polygon) noexcept
        {
            Extent extent;
            for (const Point& point : polygon)
                extent.include(point.x, point.y);

            return extent.box;
        }

        /** @brief Holds one edge of a polygon: its first vertex and its extent on each axis. */
        struct Edge
        {
            /** @brief Holds the vertex the edge starts at. */
            Point from{};

            /** @brief Holds the next vertex's x minus from.x, in float. */
            float delta_x = 0.0f;

            /** @brief Holds the next vertex's y minus from.y, in float. */
            float delta_y = 0.0f;

            /** @brief Returns the point at fraction t of the edge: from + t * delta, in float. */
            [[nodiscard]] Point at(float t) const noexcept
            {
                return Point{from.x + t * delta_x, from.y + t * delta_y};
            }
        };

        /** @brief Reports whether a fraction along an edge lies on the edge, in [0, 1]. */
        [[nodiscard]] bool is_on_edge(float t) noexcept
        {
            return t >= 0.0f && t <= 1.0f;
        }

        /**
         * @brief Includes the point where an edge crosses the line x = bound, if it does.
         * @param height The largest y the crossing may have; the smallest is zero.
         */
        void include_column_crossing(const Edge& edge, float bound, float height,
                                     Extent& extent) noexcept
        {
            if (edge.delta_x == 0.0f)
                return;

            const float t = (bound - edge.from.x) / edge.delta_x;
            const Point crossing = edge.at(t);

            if (is_on_edge(t) && crossing.y >= 0.0f && crossing.y <= height)
                extent.include(crossing.x, crossing.y);
        }

        /**
         * @brief Includes the point where an edge crosses the line y = bound, if it does.
         * @param width The largest x the crossing may have; the smallest is zero.
         */
        void include_row_crossing(const Edge& edge, float bound, float width,
                                  Extent& extent) noexcept
        {
            if (edge.delta_y == 0.0f)
                return;

            const float t = (bound - edge.from.y) / edge.delta_y;
            const Point crossing = edge.at(t);

            if (is_on_edge(t) && crossing.x >= 0.0f && crossing.x <= width)
                extent.include(crossing.x, crossing.y);
        }
    } // namespace

    std::size_t resampled_point_count(std::size_t longest) noexcept
    {
        return longest > K_POLYGON_POINTS ? longest + 1 : K_POLYGON_POINTS;
    }

    Polygon resample_polygon(const Polygon& polygon, std::size_t count) noexcept
    {
        SA_FATAL_IF(count == 0, "resample_polygon: a point count of 0 leaves no polygon");

        if (polygon.empty() || polygon.size() == count)
            return polygon;

        const std::size_t closed = polygon.size() + 1;
        const bool is_merged = closed < count;
        const EvenPositions even{is_merged ? count - closed : count,
                                 static_cast<double>(closed - 1)};

        Polygon resampled;
        resampled.reserve(count);

        std::size_t vertex = 0;
        std::size_t position = 0;

        while (resampled.size() < count)
        {
            // A vertex goes before the first even position that is not below it.
            const bool is_vertex_next =
                is_merged && vertex < closed &&
                (position == even.count || even.at(position) >= static_cast<double>(vertex));

            if (is_vertex_next)
            {
                resampled.push_back(polygon[vertex % polygon.size()]);
                ++vertex;
                continue;
            }

            resampled.push_back(point_at(polygon, even.at(position)));
            ++position;
        }

        return resampled;
    }

    void map_polygon(Polygon& polygon, float scale_x, float scale_y, float offset_x,
                     float offset_y) noexcept
    {
        for (Point& point : polygon)
        {
            point.x = point.x * scale_x + offset_x;
            point.y = point.y * scale_y + offset_y;
        }
    }

    void clamp_polygon(Polygon& polygon, const PixelBox& bounds) noexcept
    {
        for (Point& point : polygon)
        {
            point.x = std::min(std::max(point.x, bounds.x1), bounds.x2);
            point.y = std::min(std::max(point.y, bounds.y1), bounds.y2);
        }
    }

    bool is_point_inside(const Polygon& polygon, Point point) noexcept
    {
        const double x = point.x;
        const double y = point.y;
        bool is_inside = false;

        for (std::size_t i = 0; i < polygon.size(); ++i)
        {
            const Point& from = polygon[i];
            const Point& to = polygon[(i + 1) % polygon.size()];
            const double from_x = from.x;
            const double from_y = from.y;
            const double to_x = to.x;
            const double to_y = to.y;
            const double cross = (to_x - from_x) * (y - from_y) - (to_y - from_y) * (x - from_x);

            const bool is_in_columns = x >= std::min(from_x, to_x) && x <= std::max(from_x, to_x);
            const bool is_in_rows = y >= std::min(from_y, to_y) && y <= std::max(from_y, to_y);
            if (cross == 0.0 && is_in_columns && is_in_rows)
                return true;

            // An edge that rises crosses the ray when the point is on its left.
            const bool is_rising = to_y > from_y;
            const bool is_straddling = (from_y > y) != (to_y > y);
            if (is_straddling && (is_rising ? cross > 0.0 : cross < 0.0))
                is_inside = !is_inside;
        }

        return is_inside;
    }

    PixelBox visible_polygon_box(const Polygon& polygon, ImageSize size) noexcept
    {
        if (polygon.empty())
            return PixelBox{};

        const float width = static_cast<float>(size.width);
        const float height = static_cast<float>(size.height);
        const PixelBox bounds = bounds_of(polygon);

        if (bounds.x1 >= 0.0f && bounds.y1 >= 0.0f && bounds.x2 <= width && bounds.y2 <= height)
            return bounds;

        if (bounds.x2 < 0.0f || bounds.y2 < 0.0f || bounds.x1 > width || bounds.y1 > height)
            return PixelBox{};

        Extent extent;
        for (std::size_t i = 0; i < polygon.size(); ++i)
        {
            const Point& from = polygon[i];
            const Point& to = polygon[(i + 1) % polygon.size()];
            const Edge edge{from, to.x - from.x, to.y - from.y};

            if (from.x >= 0.0f && from.y >= 0.0f && from.x <= width && from.y <= height)
                extent.include(from.x, from.y);

            include_column_crossing(edge, 0.0f, height, extent);
            include_column_crossing(edge, width, height, extent);
            include_row_crossing(edge, 0.0f, width, extent);
            include_row_crossing(edge, height, width, extent);
        }

        for (const Point& corner : {Point{0.0f, 0.0f}, Point{width, 0.0f}, Point{0.0f, height},
                                    Point{width, height}})
        {
            if (is_point_inside(polygon, corner))
                extent.include(corner.x, corner.y);
        }

        return extent.has_point ? extent.box : PixelBox{};
    }
} // namespace SushiAI::Vision::Imaging
