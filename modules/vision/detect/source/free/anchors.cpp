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
 * @file anchors.cpp
 * @brief Implements the anchor points and per-anchor strides of a detector's feature levels.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/free/anchors.hpp>

#include <cstddef>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        Result<std::int64_t> count_anchors(SushiRuntime::span<const FeatureLevel> levels) noexcept
        {
            std::int64_t count = 0;
            for (const FeatureLevel& level : levels)
            {
                SA_FAIL_IF(level.height <= 0 || level.width <= 0 || !(level.stride > 0.0f),
                           StatusCode::InvalidDescription,
                           "a feature level needs positive extents and a positive stride; got "
                           "height {}, width {}, stride {}",
                           level.height, level.width, level.stride);
                count += level.height * level.width;
            }
            return count;
        }

        Status make_anchors(SushiRuntime::span<const FeatureLevel> levels,
                            SushiRuntime::span<float> points,
                            SushiRuntime::span<float> strides) noexcept
        {
            SA_TRY_ASSIGN(const std::int64_t anchor_count, count_anchors(levels));
            const std::size_t count = static_cast<std::size_t>(anchor_count);
            SA_FAIL_IF(points.size() != count * K_POINT_VALUES || strides.size() != count,
                       StatusCode::InvalidDescription,
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
            return Status();
        }

        Status check_scale_sizes(std::size_t points, std::size_t strides,
                                 std::size_t scaled) noexcept
        {
            SA_FAIL_IF(points != strides * K_POINT_VALUES || scaled != points,
                       StatusCode::InvalidDescription,
                       "scale_points needs [A, 2] points, [A] strides and [A, 2] output; got "
                       "{}, {} and {} values",
                       points, strides, scaled);
            return Status();
        }

        void scale_points(SushiRuntime::span<const float> points,
                          SushiRuntime::span<const float> strides,
                          SushiRuntime::span<float> scaled) noexcept
        {
            const Status checked = check_scale_sizes(points.size(), strides.size(), scaled.size());
            SA_FATAL_IF(!checked.is_ok(), "{}", checked.message());

            for (std::size_t anchor = 0; anchor < strides.size(); ++anchor)
            {
                const std::size_t at = anchor * K_POINT_VALUES;
                scaled[at] = points[at] * strides[anchor];
                scaled[at + 1] = points[at + 1] * strides[anchor];
            }
        }

    } // namespace Vision::Detect::Free
} // namespace SushiAI
