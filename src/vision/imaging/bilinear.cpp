/**************************************************************************/
/* bilinear.cpp                                                           */
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
 * @file bilinear.cpp
 * @brief Implements the bilinear resize kernel over rows the caller owns.
 * @author Mustafa Garip
 */

#include "bilinear.hpp"

#include <algorithm>
#include <vector>

namespace SushiAI::Vision::Imaging::Detail
{
    namespace
    {
        /** @brief Holds the two source offsets one destination index reads and their blend. */
        struct AxisTap
        {
            /** @brief Holds the lower source index times the stride of the axis. */
            std::size_t first = 0;

            /** @brief Holds the upper source index times the stride, first at the last edge. */
            std::size_t second = 0;

            /** @brief Holds the weight of second, in [0, K_WEIGHT_ONE]. */
            std::uint32_t second_weight = 0;
        };

        /**
         * @brief Returns the taps of one axis, one per destination index.
         *
         * Destination index i reads source position (i + 0.5) * source_extent / extent - 0.5,
         * clamped to the edge indices. README.md, section resize.hpp, gives the arithmetic.
         * @param stride Counts the bytes between two neighbours along the axis.
         * @pre Both extents are positive.
         */
        [[nodiscard]] std::vector<AxisTap> make_axis_taps(std::int32_t source_extent,
                                                          std::int32_t extent, std::size_t stride)
        {
            const double scale = static_cast<double>(source_extent) / static_cast<double>(extent);
            const std::int32_t last = source_extent - 1;

            std::vector<AxisTap> taps(static_cast<std::size_t>(extent));

            for (std::int32_t i = 0; i < extent; ++i)
            {
                const double position = (static_cast<double>(i) + 0.5) * scale - 0.5;
                const double floored = std::floor(position);

                double fraction = position - floored;
                std::int32_t first = 0;

                if (floored < 0.0)
                {
                    fraction = 0.0;
                }
                else if (floored >= static_cast<double>(last))
                {
                    first = last;
                    fraction = 0.0;
                }
                else
                {
                    first = static_cast<std::int32_t>(floored);
                }

                AxisTap& tap = taps[static_cast<std::size_t>(i)];
                tap.first = static_cast<std::size_t>(first) * stride;
                tap.second = static_cast<std::size_t>(std::min(first + 1, last)) * stride;
                tap.second_weight = to_weight(fraction);
            }

            return taps;
        }
    } // namespace

    void resize_rows(ImageView source, ImageSize size, std::uint8_t* destination,
                     std::size_t row_stride)
    {
        const std::size_t channels = static_cast<std::size_t>(source.channels());

        // A row tap is a byte offset into the image and a column tap one into a row.
        const std::vector<AxisTap> rows =
            make_axis_taps(source.height(), size.height, source.row_bytes());
        const std::vector<AxisTap> columns = make_axis_taps(source.width(), size.width, channels);

        for (const AxisTap& row : rows)
        {
            const std::uint8_t* upper = source.data() + row.first;
            const std::uint8_t* lower = source.data() + row.second;
            std::uint8_t* out = destination;

            for (const AxisTap& column : columns)
            {
                for (std::size_t c = 0; c < channels; ++c)
                {
                    const std::uint32_t top = blend_pair(upper[column.first + c],
                                                         upper[column.second + c],
                                                         column.second_weight);
                    const std::uint32_t bottom = blend_pair(lower[column.first + c],
                                                            lower[column.second + c],
                                                            column.second_weight);

                    out[c] = blend_rows(top, bottom, row.second_weight);
                }

                out += channels;
            }

            destination += row_stride;
        }
    }
} // namespace SushiAI::Vision::Imaging::Detail
