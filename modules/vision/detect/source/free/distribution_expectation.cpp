/**************************************************************************/
/* distribution_expectation.cpp                                           */
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
 * @file distribution_expectation.cpp
 * @brief Implements the expected distance of each side of a box output that holds bins.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/free/distribution_expectation.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/float_order.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        Status expect_distances(SushiRuntime::span<const float> logits,
                                std::int64_t reg_max,
                                SushiRuntime::span<float> distances) noexcept
        {
            SA_FAIL_IF(reg_max < 1, StatusCode::InvalidDescription,
                       "reg_max must be positive; got {}", reg_max);
            const std::size_t bins = static_cast<std::size_t>(reg_max);
            const std::size_t sides = distances.size();
            SA_FAIL_IF(sides % K_BOX_VALUES != 0 || logits.size() != sides * bins,
                       StatusCode::InvalidDescription,
                       "bin logits must be [B, A, 4 * {}] over [B, A, 4] distances; got {} "
                       "and {} values",
                       bins, logits.size(), sides);
            const std::int64_t bad = first_non_finite(logits);
            SA_FAIL_IF(bad >= 0, StatusCode::InvalidDescription,
                       "bin logit {} is {}; every logit must be finite", bad,
                       logits[static_cast<std::size_t>(bad)]);

            for (std::size_t side = 0; side < sides; ++side)
            {
                const float* row = logits.data() + side * bins;
                if (bins == 1)
                {
                    distances[side] = row[0];
                    continue;
                }

                // The maximum shifts every exponent to at most 0, so none overflows.
                const double peak = *std::max_element(row, row + bins);
                double mass = 0.0;
                double moment = 0.0;
                for (std::size_t bin = 0; bin < bins; ++bin)
                {
                    const double weight = std::exp(static_cast<double>(row[bin]) - peak);
                    mass += weight;
                    moment += weight * static_cast<double>(bin);
                }
                distances[side] = static_cast<float>(moment / mass);
            }

            return Status();
        }

        void fill_distribution_projection(SushiRuntime::span<float> projection) noexcept
        {
            for (std::size_t bin = 0; bin < projection.size(); ++bin)
                projection[bin] = static_cast<float>(bin);
        }

    } // namespace Vision::Detect::Free
} // namespace SushiAI
