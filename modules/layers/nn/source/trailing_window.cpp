/**************************************************************************/
/* trailing_window.cpp                                                    */
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
 * @file trailing_window.cpp
 * @brief Implements the window rule of the layers that pad an axis more at its end than at its start.
 * @author Mustafa Garip
 */

#include "trailing_window.hpp"

#include <limits>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace Detail
        {
            Result<Graph::SpatialExtents> trailing_window_extents(
                std::string_view layer, const Shape& sample, const Graph::ConvGeometry& geometry,
                int64_t trailing_h, int64_t trailing_w) noexcept
            {
                SA_FAIL_IF(sample.rank() != 3, StatusCode::InvalidDescription,
                           "{} needs a per-sample shape of [H, W, C]; got {}", layer,
                           sample.to_string());
                SA_FAIL_IF(trailing_h < 0 || trailing_w < 0, StatusCode::InvalidDescription,
                           "{} needs a non-negative trailing padding; got {},{}", layer, trailing_h,
                           trailing_w);

                constexpr int64_t K_LARGEST_EXTENT = std::numeric_limits<int64_t>::max();
                SA_FAIL_IF(trailing_h > K_LARGEST_EXTENT - sample[0] ||
                               trailing_w > K_LARGEST_EXTENT - sample[1],
                           StatusCode::InvalidDescription,
                           "{} needs a trailing padding that keeps the input within 64 bits; got "
                           "{},{} over {}",
                           layer, trailing_h, trailing_w, sample.to_string());

                // The trailing pad lengthens the input, so the window rule covers it.
                const Result<Graph::SpatialExtents> fitted =
                    Graph::window_extents(sample[0] + trailing_h, sample[1] + trailing_w, geometry);
                if (!fitted.has_value())
                    return SA_ANNOTATE(fitted.status(),
                                       "{} needs a legal window over its per-sample shape; got {}",
                                       layer, sample.to_string());

                SA_TRY_ASSIGN(const Graph::SpatialExtents extents, fitted);

                // A window that starts at or past the input's end holds padding alone.
                SA_FAIL_IF((extents.height - 1) * geometry.stride_h - geometry.pad_top >= sample[0] ||
                               (extents.width - 1) * geometry.stride_w - geometry.pad_left >=
                                   sample[1],
                           StatusCode::InvalidDescription,
                           "{} needs a trailing padding that leaves every window a tap inside the "
                           "input; got {},{} over {}",
                           layer, trailing_h, trailing_w, sample.to_string());

                return extents;
            }
        } // namespace Detail
    } // namespace NN
} // namespace SushiAI
