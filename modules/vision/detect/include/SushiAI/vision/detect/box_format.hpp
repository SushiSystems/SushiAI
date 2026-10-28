/**************************************************************************/
/* box_format.hpp                                                         */
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
 * @file box_format.hpp
 * @brief Declares the conversions between corner boxes and centre-size boxes.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstddef>

#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        /** @brief Holds the number of values in one box. */
        inline constexpr std::size_t K_BOX_VALUES = 4;

        /**
         * @brief Converts `[N, 4]` boxes from x1 y1 x2 y2 to centre x, centre y, width, height.
         * @param converted Receives `[N, 4]`; may be @p boxes.
         * @pre @p boxes and @p converted hold the same whole number of boxes.
         * @see modules/vision/detect/README.md
         */
        void xyxy_to_xywh(SushiRuntime::span<const float> boxes,
                          SushiRuntime::span<float> converted);

        /**
         * @brief Converts `[N, 4]` boxes from centre x, centre y, width, height to x1 y1 x2 y2.
         * @param converted Receives `[N, 4]`; may be @p boxes.
         * @pre @p boxes and @p converted hold the same whole number of boxes.
         * @see modules/vision/detect/README.md
         */
        void xywh_to_xyxy(SushiRuntime::span<const float> boxes,
                          SushiRuntime::span<float> converted);

    } // namespace Vision::Detect
} // namespace SushiAI
