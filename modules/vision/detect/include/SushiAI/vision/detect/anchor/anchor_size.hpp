/**************************************************************************/
/* anchor_size.hpp                                                        */
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
 * @file anchor_size.hpp
 * @brief Declares the size of an anchor box and the IoU of two sizes.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the width and height of a box as fractions of the network input. */
        struct AnchorSize
        {
            float width = 0.0f;  /**< Width over the network input's width. */
            float height = 0.0f; /**< Height over the network input's height. */
        };

        /**
         * @brief Returns the IoU of two sizes laid on one centre.
         * @param a Sides that are not negative.
         * @param b Sides that are not negative.
         * @return A value in [0, 1]; 0 when the union has no area or is not a number.
         */
        [[nodiscard]] float shape_iou(AnchorSize a, AnchorSize b) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
