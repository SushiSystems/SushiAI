/**************************************************************************/
/* calibration_error.cpp                                                  */
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
 * @file calibration_error.cpp
 * @brief Implements the expected calibration error over equal-width confidence bins.
 * @author Mustafa Garip
 */

#include <SushiAI/language/training/calibration_error.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "logit_rows.hpp"

namespace SushiAI::Language::Training
{
    namespace
    {
        /** @brief Names the metric in a message. */
        constexpr std::string_view K_METRIC = "expected_calibration_error";

        /** @brief Holds the bins as an index. */
        constexpr std::size_t K_BINS = static_cast<std::size_t>(K_CALIBRATION_BINS);

        /** @brief Holds what the rows of one confidence bin add up to. */
        struct BinTotals
        {
            /** @brief Counts the rows of the bin that the logits get right. */
            double correct = 0.0;

            /** @brief Sums the confidence of the rows of the bin. */
            double confidence = 0.0;
        };

        /** @brief Returns the bin of a confidence in [0, 1]; 1 is in the last bin. */
        [[nodiscard]] std::size_t bin_of(double confidence) noexcept
        {
            const std::size_t bin = static_cast<std::size_t>(confidence * K_CALIBRATION_BINS);
            return std::min(bin, K_BINS - 1);
        }
    } // namespace

    Result<double> expected_calibration_error(SushiRuntime::span<const float> logits,
                                              SushiRuntime::span<const std::int32_t> labels,
                                              std::int64_t classes,
                                              double temperature) noexcept
    {
        SA_TRY(Detail::check_logit_rows(logits, labels, classes, temperature, K_METRIC));

        const std::size_t width = static_cast<std::size_t>(classes);
        std::array<BinTotals, K_BINS> bins{};

        for (std::size_t row = 0; row < labels.size(); ++row)
        {
            const float* const values = logits.data() + row * width;
            const std::size_t best = Detail::row_argmax(values, width);

            // The best class adds 1 to the normaliser, so its probability is 1 / (1 + excess).
            const double confidence =
                1.0 / (1.0 + Detail::exponential_excess(values, width, best, 1.0 / temperature));

            BinTotals& bin = bins[bin_of(confidence)];
            bin.confidence += confidence;
            if (best == static_cast<std::size_t>(labels[row]))
                bin.correct += 1.0;
        }

        // Rows of a bin times its accuracy gap is |correct - confidence sum|.
        double gap = 0.0;
        for (const BinTotals& bin : bins)
            gap += std::fabs(bin.correct - bin.confidence);

        return gap / static_cast<double>(labels.size());
    }
} // namespace SushiAI::Language::Training
