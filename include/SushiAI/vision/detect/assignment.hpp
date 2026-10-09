/**************************************************************************/
/* assignment.hpp                                                         */
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
 * @file assignment.hpp
 * @brief Declares Assignment, the per-anchor targets a label assigner produces.
 * @author Mustafa Garip
 * @see include/SushiAI/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

namespace SushiAI
{
    namespace Vision::Detect
    {
        /**
         * @brief Holds the targets of every anchor of a batch, at full shape.
         * @see include/SushiAI/vision/detect/README.md
         */
        struct Assignment
        {
            std::int64_t batch = 0;       /**< Images, B. */
            std::int64_t anchors = 0;     /**< Anchors over all levels, A. */
            std::int64_t num_classes = 0; /**< Classes, C. */

            std::vector<std::uint8_t> foreground_mask; /**< `[B, A]`; 1 on an assigned anchor. */
            std::vector<float> target_boxes;           /**< `[B, A, 4]` x1 y1 x2 y2 in pixels. */
            std::vector<float> target_scores;          /**< `[B, A, C]` soft class targets. */
            std::vector<std::int64_t> target_labels;   /**< `[B, A]` class of the assigned box. */
            std::vector<std::int64_t> target_indices;  /**< `[B, A]` row of the assigned box. */
        };

    } // namespace Vision::Detect
} // namespace SushiAI
