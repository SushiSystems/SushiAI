/**************************************************************************/
/* prediction_shape.hpp                                                   */
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
 * @file prediction_shape.hpp
 * @brief Declares the extents of a head's boxes and scores over all anchors.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>

namespace SushiAI
{
    namespace Vision::Detect
    {
        /**
         * @brief Holds the extents of `[B, A, 4]` boxes and the `[B, A, C]` scores beside them.
         * @see modules/vision/detect/README.md
         */
        struct PredictionShape
        {
            std::int64_t batch = 0;       /**< Images, B. */
            std::int64_t anchors = 0;     /**< Anchors over all levels, A. */
            std::int64_t num_classes = 0; /**< Classes, C. */
        };

    } // namespace Vision::Detect
} // namespace SushiAI
