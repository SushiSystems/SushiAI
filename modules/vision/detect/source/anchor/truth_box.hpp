/**************************************************************************/
/* truth_box.hpp                                                          */
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
 * @file truth_box.hpp
 * @brief Defines the reading of a truth row's pixel corners as a centre-and-size box.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/vision/detect/anchor/head_options.hpp>

#include "box_delta.hpp"

namespace SushiAI::Vision::Detect::Anchor::Detail
{
    /**
     * @brief Returns a truth's box as centre and size in fractions of the network input.
     * @param corners The row's `x1 y1 x2 y2` in pixels.
     * @param network Size the corners are divided by; both sides positive.
     */
    inline CentreBox read_truth_box(const float* corners, NetworkSize network) noexcept
    {
        const float network_width = static_cast<float>(network.width);
        const float network_height = static_cast<float>(network.height);

        CentreBox box;
        box.x = (corners[0] + corners[2]) / 2.0f / network_width;
        box.y = (corners[1] + corners[3]) / 2.0f / network_height;
        box.width = (corners[2] - corners[0]) / network_width;
        box.height = (corners[3] - corners[1]) / network_height;
        return box;
    }

} // namespace SushiAI::Vision::Detect::Anchor::Detail
