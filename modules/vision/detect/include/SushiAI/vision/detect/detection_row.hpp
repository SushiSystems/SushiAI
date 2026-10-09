/**************************************************************************/
/* detection_row.hpp                                                      */
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
 * @file detection_row.hpp
 * @brief Declares the columns of a detection row and the reference's limits on an image's rows.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace SushiAI
{
    namespace Vision::Detect
    {
        /** @brief Holds the number of values in one detection: x1 y1 x2 y2 score class. */
        inline constexpr std::size_t K_DETECTION_VALUES = 6;

        /** @brief Holds the column of a detection's score. */
        inline constexpr std::size_t K_DETECTION_SCORE = 4;

        /** @brief Holds the column of a detection's class index, stored as a float. */
        inline constexpr std::size_t K_DETECTION_CLASS = 5;

        /** @brief Holds the reference's limit on detections per image. */
        inline constexpr std::int64_t K_MAX_DETECTIONS = 300;

        /** @brief Holds the score a detection must exceed to enter validation. */
        inline constexpr float K_VALIDATION_CONFIDENCE = 0.001f;

    } // namespace Vision::Detect
} // namespace SushiAI
