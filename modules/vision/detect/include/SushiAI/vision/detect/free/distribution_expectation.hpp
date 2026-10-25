/**************************************************************************/
/* distribution_expectation.hpp                                           */
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

/**
 * @file distribution_expectation.hpp
 * @brief Declares the expected distance of each side of a box output that holds bins.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>

#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        /**
         * @brief Decodes `[B, A, 4 * reg_max]` bin logits into `[B, A, 4]` expected distances.
         * @param logits Channel `side * reg_max + bin`, sides in the order left, top, right,
         *        bottom; every value must be finite, and the first that is not is refused.
         * @param reg_max Bins per side; must be positive. 1 copies the logits.
         * @param distances Receives `[B, A, 4]`: the softmax of a side's bins times 0..reg_max-1.
         * @see modules/vision/detect/README.md
         */
        void expect_distances(SushiRuntime::span<const float> logits,
                              std::int64_t reg_max,
                              SushiRuntime::span<float> distances);

        /**
         * @brief Writes the bin values 0, 1, ..., reg_max - 1 into @p projection.
         * @param projection The `[reg_max, 1]` leaf the expectation graph multiplies by.
         * @see modules/vision/detect/README.md
         */
        void fill_distribution_projection(SushiRuntime::span<float> projection);

    } // namespace Vision::Detect::Free
} // namespace SushiAI
