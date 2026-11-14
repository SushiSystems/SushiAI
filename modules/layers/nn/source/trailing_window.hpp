/**************************************************************************/
/* trailing_window.hpp                                                    */
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
 * @file trailing_window.hpp
 * @brief Declares the window rule of the layers that pad an axis more at its end than at its start.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace Detail
        {
            /**
             * @brief Computes the window extents of a [H, W, C] sample under a trailing pad.
             * @param layer Names the layer in a message.
             * @param trailing_h The zero rows added below the input.
             * @param trailing_w The zero columns added right of the input.
             * @return The window rows and columns; InvalidDescription for a sample that is no
             *         [H, W, C], a bad pad, an illegal window, or a window with no tap inside.
             */
            [[nodiscard]] Result<Graph::SpatialExtents> trailing_window_extents(
                std::string_view layer, const Shape& sample, const Graph::ConvGeometry& geometry,
                int64_t trailing_h, int64_t trailing_w) noexcept;
        } // namespace Detail
    } // namespace NN
} // namespace SushiAI
