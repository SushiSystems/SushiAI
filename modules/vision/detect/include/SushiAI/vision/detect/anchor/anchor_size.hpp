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
