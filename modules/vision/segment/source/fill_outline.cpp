/**************************************************************************/
/* fill_outline.cpp                                                       */
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
 * @file fill_outline.cpp
 * @brief Implements the scanline fill of an outline by the even-odd rule.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/segment/fill_outline.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>
#include <SushiBLAS/core/format.hpp>
#include <SushiAI/core/seam.hpp>

namespace SushiAI::Vision::Segment
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_UNIT = "fill_outline";

        /** @brief Holds the channel count of a mask. */
        constexpr std::int32_t K_MASK_CHANNELS = 1;

        /** @brief Holds the offset of a pixel's centre from its index, on both axes. */
        constexpr double K_PIXEL_CENTRE = 0.5;

        /** @brief Holds a run of rows: the first, and one past the last. */
        struct RowRange
        {
            std::int32_t first = 0; /**< First row of the run. */
            std::int32_t end = 0;   /**< One past the last row of the run. */
        };

        /**
         * @brief Returns the first index whose centre is at or after @p position, in [0, limit].
         * @pre @p position is finite and @p limit is at least 0.
         */
        [[nodiscard]] std::int32_t first_centre_at(double position, std::int32_t limit) noexcept
        {
            const double index = std::ceil(position - K_PIXEL_CENTRE);

            return static_cast<std::int32_t>(std::clamp(index, 0.0, static_cast<double>(limit)));
        }

        /** @brief Returns the rows of a mask of @p height rows whose centres the parts span. */
        [[nodiscard]] RowRange rows_spanned(SushiRuntime::span<const Imaging::Polygon> parts,
                                            std::int32_t height) noexcept
        {
            double lowest = std::numeric_limits<double>::infinity();
            double highest = -std::numeric_limits<double>::infinity();
            for (const Imaging::Polygon& part : parts)
            {
                for (const Imaging::Point& point : part)
                {
                    lowest = std::min(lowest, static_cast<double>(point.y));
                    highest = std::max(highest, static_cast<double>(point.y));
                }
            }

            if (lowest > highest)
                return RowRange{};

            return RowRange{first_centre_at(lowest, height), first_centre_at(highest, height)};
        }

        /**
         * @brief Appends the x at which each edge of @p part crosses the height @p centre.
         *
         * An edge counts when one end is at or above @p centre and the other below it, so a
         * vertex on the line is counted for one of its two edges.
         */
        void append_crossings(const Imaging::Polygon& part, double centre,
                              std::vector<double>& crossings)
        {
            for (std::size_t index = 0; index < part.size(); ++index)
            {
                const Imaging::Point& a = part[index];
                const Imaging::Point& b = part[(index + 1) % part.size()];
                const double ay = static_cast<double>(a.y);
                const double by = static_cast<double>(b.y);
                if ((ay <= centre) == (by <= centre))
                    continue;

                const double ax = static_cast<double>(a.x);
                const double bx = static_cast<double>(b.x);
                crossings.push_back(ax + (centre - ay) * (bx - ax) / (by - ay));
            }
        }

        /** @brief Fills, row by row, the columns between each pair of sorted crossings. */
        void paint(SushiRuntime::span<const Imaging::Polygon> parts, std::uint8_t value,
                   Imaging::Image& mask)
        {
            const RowRange rows = rows_spanned(parts, mask.height());
            std::vector<double> crossings;
            for (std::int32_t y = rows.first; y < rows.end; ++y)
            {
                crossings.clear();
                for (const Imaging::Polygon& part : parts)
                    append_crossings(part, static_cast<double>(y) + K_PIXEL_CENTRE, crossings);
                std::sort(crossings.begin(), crossings.end());

                // Every part is closed, so the crossings of a row come in pairs.
                std::uint8_t* const row = mask.row(y);
                for (std::size_t pair = 0; pair + 1 < crossings.size(); pair += 2)
                {
                    const std::int32_t first = first_centre_at(crossings[pair], mask.width());
                    const std::int32_t end = first_centre_at(crossings[pair + 1], mask.width());
                    if (first < end)
                        std::fill(row + first, row + end, value);
                }
            }
        }
    } // namespace

    Status fill_outline(SushiRuntime::span<const Imaging::Polygon> parts, std::uint8_t value,
                        Imaging::Image& mask)
    {
        SA_FAIL_IF(mask.channels() != K_MASK_CHANNELS || !Imaging::is_valid_size(mask.size()),
                   StatusCode::InvalidDescription,
                   "{}: the mask is {} by {} of {} channels; an outline is filled into a sized "
                   "mask of 1",
                   K_UNIT, mask.height(), mask.width(), mask.channels());

        for (std::size_t part = 0; part < parts.size(); ++part)
        {
            for (std::size_t vertex = 0; vertex < parts[part].size(); ++vertex)
            {
                const Imaging::Point& point = parts[part][vertex];
                SA_FAIL_IF(!std::isfinite(point.x) || !std::isfinite(point.y),
                           StatusCode::InvalidDescription,
                           "{}: vertex {} of part {} is ({}, {}); a vertex is a finite point",
                           K_UNIT, vertex, part, point.x, point.y);
            }
        }

        return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
        {
            paint(parts, value, mask);
            return Status();
        });
    }

} // namespace SushiAI::Vision::Segment
