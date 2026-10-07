/**************************************************************************/
/* anchors.hpp                                                            */
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
 * @file anchors.hpp
 * @brief Declares the anchor points and per-anchor strides of a detector's feature levels.
 * @author Mustafa Garip
 * @see include/SushiAI/detect/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Detect
    {
        /** @brief Holds the distance from a cell's index to its centre, in grid units. */
        inline constexpr float K_GRID_CELL_OFFSET = 0.5f;

        /** @brief Holds the number of values in one anchor point: x then y. */
        inline constexpr std::size_t K_POINT_VALUES = 2;

        /** @brief Holds the extents and the stride of one feature level. */
        struct FeatureLevel
        {
            std::int64_t height = 0; /**< Rows of the feature map. */
            std::int64_t width = 0;  /**< Columns of the feature map. */
            float stride = 0.0f;     /**< Input pixels per cell. */
        };

        /**
         * @brief Returns the number of anchors over @p levels, one per cell.
         * @see include/SushiAI/detect/README.md
         */
        [[nodiscard]] std::int64_t count_anchors(SushiRuntime::span<const FeatureLevel> levels);

        /**
         * @brief Writes each anchor's cell centre in grid units and its stride.
         * @param points Receives `[A, 2]` as x then y, levels in order, row-major in a level.
         * @param strides Receives `[A]`.
         * @see include/SushiAI/detect/README.md
         */
        void make_anchors(SushiRuntime::span<const FeatureLevel> levels,
                          SushiRuntime::span<float> points,
                          SushiRuntime::span<float> strides);

        /**
         * @brief Multiplies each anchor point by its stride, which gives the point in pixels.
         * @param points `[A, 2]` in grid units.
         * @param strides `[A]`.
         * @param scaled Receives `[A, 2]`; may be @p points.
         */
        void scale_points(SushiRuntime::span<const float> points,
                          SushiRuntime::span<const float> strides,
                          SushiRuntime::span<float> scaled);

    } // namespace Detect
} // namespace SushiAI
