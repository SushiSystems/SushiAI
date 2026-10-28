/**************************************************************************/
/* target_leaves.hpp                                                      */
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
 * @file target_leaves.hpp
 * @brief Declares the full-shape arrays a masked detection loss reads from an assignment.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <array>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>

#include <SushiAI/vision/detect/free/assignment.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        /** @brief Holds the box an unassigned row carries: x1 y1 x2 y2 in grid units. */
        inline constexpr std::array<float, 4> K_SAFE_BOX = {0.0f, 0.0f, 1.0f, 1.0f};

        /** @brief Holds the smallest value the sum of target scores divides by. */
        inline constexpr float K_MIN_TARGET_SCORES_SUM = 1.0f;

        /** @brief Holds the geometry the leaves are scaled by. */
        struct TargetLeafInputs
        {
            SushiRuntime::span<const float> anchor_points; /**< `[A, 2]` x then y, grid units. */
            SushiRuntime::span<const float> strides;       /**< `[A]` pixels per cell. */
            float image_height = 0.0f;                     /**< Input height in pixels. */
            float image_width = 0.0f;                      /**< Input width in pixels. */
        };

        /**
         * @brief Holds the FLOAT32 arrays one branch's loss reads, over every anchor of the batch.
         * @see modules/vision/detect/README.md
         */
        struct TargetLeaves
        {
            /** @brief Holds `[B, A, 4]`, a level's `[B, H, W, 4]`: 1 on a foreground row. */
            std::vector<float> box_mask;

            /** @brief Holds `[B, A]`, a level's `[B, H, W, 1]`: the sum of a row's scores. */
            std::vector<float> weight;

            /** @brief Holds `[B, A]`, a level's `[B, H, W, 1]`: weight over target_scores_sum. */
            std::vector<float> normalized_weight;

            /** @brief Holds `[B, A, 4]`, a level's `[B, H, W, 4]`: x1 y1 x2 y2 in grid units. */
            std::vector<float> target_boxes;

            /** @brief Holds `[B, A, 4]`, a level's `[B, H, W, 4]`: left, top, right, bottom. */
            std::vector<float> target_distances;

            /** @brief Holds `[B, A, 4]`, a level's `[B, H, W, 4]`: the factor of each L1 error. */
            std::vector<float> l1_scale;

            /** @brief Holds `[B, A, 4]`, a level's `[B, H, W, 4]`: K_SAFE_BOX in every row. */
            std::vector<float> safe_boxes;

            /** @brief Holds `[1]`, bound whole: the sum of all target scores, at least 1. */
            std::array<float, 1> target_scores_sum = {K_MIN_TARGET_SCORES_SUM};
        };

        /**
         * @brief Writes the loss leaves of @p assignment into @p leaves.
         * @return InvalidDescription for anchor points or strides that do not match the
         *         assignment's anchors, or an image size that is not positive.
         * @pre @p assignment came from TaskAlignedAssigner::assign().
         * @see modules/vision/detect/README.md
         */
        [[nodiscard]] Status build_target_leaves(const Assignment& assignment,
                                                 const TargetLeafInputs& inputs,
                                                 TargetLeaves& leaves) noexcept;

    } // namespace Vision::Detect::Free
} // namespace SushiAI
