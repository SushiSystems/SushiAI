/**************************************************************************/
/* distribution_targets.hpp                                               */
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
 * @file distribution_targets.hpp
 * @brief Declares the bin weights the distribution focal loss reads as a leaf.
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
        /** @brief Holds how far below reg_max - 1 a target distance is clamped. */
        inline constexpr float K_DISTRIBUTION_MARGIN = 0.01f;

        /**
         * @brief Writes the `[B, A, 4 * reg_max]` bin weights of `[B, A, 4]` target distances.
         * @param normalized_weight `[B, A]`; a NaN or infinity here or in a target throws Error.
         * @param reg_max Bins per side; must be at least 2.
         * @param weights Receives the two linear weights of each clamped target, at the bins
         *        around it, times a quarter of its row's normalized weight; 0 elsewhere.
         * @see modules/vision/detect/README.md
         */
        void fill_distribution_weights(SushiRuntime::span<const float> target_distances,
                                       SushiRuntime::span<const float> normalized_weight,
                                       std::int64_t reg_max,
                                       SushiRuntime::span<float> weights);

    } // namespace Vision::Detect::Free
} // namespace SushiAI
