/**************************************************************************/
/* distribution_targets.cpp                                               */
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
 * @file distribution_targets.cpp
 * @brief Implements the bin weights the distribution focal loss reads as a leaf.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/free/distribution_targets.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <SushiAI/failure/error.hpp>
#include <SushiAI/vision/detect/box_format.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        namespace
        {
            /** @brief Holds the mean over the four sides as a factor. */
            constexpr double K_SIDE_MEAN = 1.0 / static_cast<double>(K_BOX_VALUES);
        } // namespace

        void fill_distribution_weights(SushiRuntime::span<const float> target_distances,
                                       SushiRuntime::span<const float> normalized_weight,
                                       std::int64_t reg_max,
                                       SushiRuntime::span<float> weights)
        {
            SA_THROW_IF(reg_max < 2, "the distribution focal loss needs reg_max of at least 2; "
                                     "got {}",
                        reg_max);
            const std::size_t bins = static_cast<std::size_t>(reg_max);
            const std::size_t rows = normalized_weight.size();
            SA_THROW_IF(target_distances.size() != rows * K_BOX_VALUES ||
                            weights.size() != rows * K_BOX_VALUES * bins,
                        "distribution targets must be [B, A, 4] over [B, A] weights with "
                        "[B, A, 4 * {}] bin weights; got {}, {} and {} values",
                        bins, target_distances.size(), rows, weights.size());

            std::fill(weights.begin(), weights.end(), 0.0f);
            const double ceiling = static_cast<double>(reg_max - 1) - K_DISTRIBUTION_MARGIN;
            for (std::size_t side = 0; side < target_distances.size(); ++side)
            {
                const double scale =
                    K_SIDE_MEAN * static_cast<double>(normalized_weight[side / K_BOX_VALUES]);
                const double target =
                    std::clamp(static_cast<double>(target_distances[side]), 0.0, ceiling);

                // The ceiling keeps the right bin at most reg_max - 1.
                const std::size_t left = static_cast<std::size_t>(std::floor(target));
                const double left_weight = static_cast<double>(left + 1) - target;
                float* row = weights.data() + side * bins;
                row[left] = static_cast<float>(left_weight * scale);
                row[left + 1] = static_cast<float>((1.0 - left_weight) * scale);
            }
        }

    } // namespace Vision::Detect::Free
} // namespace SushiAI
