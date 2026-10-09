/**************************************************************************/
/* box_format.cpp                                                         */
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
 * @file box_format.cpp
 * @brief Implements the conversions between corner boxes and centre-size boxes.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/box_format.hpp>

#include <cstddef>

#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        namespace
        {
            /** @brief Throws unless both arrays hold the same whole number of boxes. */
            void check_box_arrays(std::size_t source_size, std::size_t converted_size)
            {
                SA_THROW_IF(source_size % K_BOX_VALUES != 0 || converted_size != source_size,
                            "a box conversion needs [N, 4] input and output; got {} and {} values",
                            source_size, converted_size);
            }
        } // namespace

        void xyxy_to_xywh(SushiRuntime::span<const float> boxes,
                          SushiRuntime::span<float> converted)
        {
            check_box_arrays(boxes.size(), converted.size());
            for (std::size_t at = 0; at < boxes.size(); at += K_BOX_VALUES)
            {
                const float x1 = boxes[at];
                const float y1 = boxes[at + 1];
                const float x2 = boxes[at + 2];
                const float y2 = boxes[at + 3];
                converted[at] = (x1 + x2) / 2.0f;
                converted[at + 1] = (y1 + y2) / 2.0f;
                converted[at + 2] = x2 - x1;
                converted[at + 3] = y2 - y1;
            }
        }

        void xywh_to_xyxy(SushiRuntime::span<const float> boxes,
                          SushiRuntime::span<float> converted)
        {
            check_box_arrays(boxes.size(), converted.size());
            for (std::size_t at = 0; at < boxes.size(); at += K_BOX_VALUES)
            {
                const float centre_x = boxes[at];
                const float centre_y = boxes[at + 1];
                const float half_width = boxes[at + 2] / 2.0f;
                const float half_height = boxes[at + 3] / 2.0f;
                converted[at] = centre_x - half_width;
                converted[at + 1] = centre_y - half_height;
                converted[at + 2] = centre_x + half_width;
                converted[at + 3] = centre_y + half_height;
            }
        }

    } // namespace Vision::Detect
} // namespace SushiAI
