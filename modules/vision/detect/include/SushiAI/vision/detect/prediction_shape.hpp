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
