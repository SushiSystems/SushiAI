/**************************************************************************/
/* anchor_size.cpp                                                        */
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
 * @file anchor_size.cpp
 * @brief Implements the IoU of two box sizes laid on one centre.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

#include <algorithm>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        float shape_iou(AnchorSize a, AnchorSize b) noexcept
        {
            const float intersection =
                std::min(a.width, b.width) * std::min(a.height, b.height);
            const float joined = a.width * a.height + b.width * b.height - intersection;

            // The comparison is false for a NaN union, which then reads as no overlap.
            if (!(joined > 0.0f))
                return 0.0f;

            return intersection / joined;
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
