/**************************************************************************/
/* anchor_leaves.hpp                                                      */
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
 * @file anchor_leaves.hpp
 * @brief Declares the anchor points and strides a detection graph reads as leaves.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>

#include <SushiAI/vision/detect/free/anchors.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        /**
         * @brief Holds the anchor points and strides of every image of a batch, as FLOAT32 leaves.
         * @see modules/vision/detect/include/SushiAI/vision/detect/free/target_leaves.hpp
         */
        struct AnchorLeaves
        {
            std::vector<float> points;  /**< `[B, A, 2]`, a level's `[B, H, W, 2]`; grid units. */
            std::vector<float> strides; /**< `[B, A, 4]`, a level's `[B, H, W, 4]`; the stride. */
        };

        /**
         * @brief Writes the points and the strides of @p levels once for each image.
         * @param batch Images, B.
         * @return InvalidDescription for a batch below 1, an empty level list or a level
         *         count_anchors() refuses.
         * @see modules/vision/detect/README.md
         */
        [[nodiscard]] Status build_anchor_leaves(SushiRuntime::span<const FeatureLevel> levels,
                                                 std::int64_t batch,
                                                 AnchorLeaves& leaves);

    } // namespace Vision::Detect::Free
} // namespace SushiAI
