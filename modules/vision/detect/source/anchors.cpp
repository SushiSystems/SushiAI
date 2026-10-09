/**************************************************************************/
/* anchors.cpp                                                            */
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
 * @file anchors.cpp
 * @brief Implements the anchor points and per-anchor strides of a detector's feature levels.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchors.hpp>

#include <cstddef>

#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        std::int64_t count_anchors(SushiRuntime::span<const FeatureLevel> levels)
        {
            std::int64_t count = 0;
            for (const FeatureLevel& level : levels)
            {
                SA_THROW_IF(level.height <= 0 || level.width <= 0 || !(level.stride > 0.0f),
                            "a feature level needs positive extents and a positive stride; got "
                            "height {}, width {}, stride {}",
                            level.height, level.width, level.stride);
                count += level.height * level.width;
            }
            return count;
        }

        void make_anchors(SushiRuntime::span<const FeatureLevel> levels,
                          SushiRuntime::span<float> points,
                          SushiRuntime::span<float> strides)
        {
            const std::size_t count = static_cast<std::size_t>(count_anchors(levels));
            SA_THROW_IF(points.size() != count * K_POINT_VALUES || strides.size() != count,
                        "make_anchors needs {} point values and {} strides; got {} and {}",
                        count * K_POINT_VALUES, count, points.size(), strides.size());

            std::size_t anchor = 0;
            for (const FeatureLevel& level : levels)
            {
                for (std::int64_t row = 0; row < level.height; ++row)
                {
                    for (std::int64_t column = 0; column < level.width; ++column)
                    {
                        float* point = points.data() + anchor * K_POINT_VALUES;
                        point[0] = static_cast<float>(column) + K_GRID_CELL_OFFSET;
                        point[1] = static_cast<float>(row) + K_GRID_CELL_OFFSET;
                        strides[anchor] = level.stride;
                        ++anchor;
                    }
                }
            }
        }

        void scale_points(SushiRuntime::span<const float> points,
                          SushiRuntime::span<const float> strides,
                          SushiRuntime::span<float> scaled)
        {
            SA_THROW_IF(points.size() != strides.size() * K_POINT_VALUES ||
                            scaled.size() != points.size(),
                        "scale_points needs [A, 2] points, [A] strides and [A, 2] output; got "
                        "{}, {} and {} values",
                        points.size(), strides.size(), scaled.size());

            for (std::size_t anchor = 0; anchor < strides.size(); ++anchor)
            {
                const std::size_t at = anchor * K_POINT_VALUES;
                scaled[at] = points[at] * strides[anchor];
                scaled[at + 1] = points[at + 1] * strides[anchor];
            }
        }

    } // namespace Vision::Detect
} // namespace SushiAI
