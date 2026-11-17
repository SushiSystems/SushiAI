/**************************************************************************/
/* temperature_scaling.cpp                                                */
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
 * @file temperature_scaling.cpp
 * @brief Implements the loss of a temperature and the search for the temperature that minimises it.
 * @author Mustafa Garip
 */

#include <SushiAI/language/training/temperature_scaling.hpp>

#include <cmath>
#include <cstddef>

#include "logit_rows.hpp"

namespace SushiAI::Language::Training
{
    namespace
    {
        /** @brief Names the loss in a message. */
        constexpr std::string_view K_LOSS = "mean_negative_log_likelihood";

        /** @brief Names the fit in a message. */
        constexpr std::string_view K_FIT = "fit_temperature";

        /** @brief Holds (sqrt(5) - 1) / 2, 1 over the golden ratio: the share of a bracket kept. */
        constexpr double K_KEPT_SHARE = 0.6180339887498949;

        /** @brief Returns the loss of checked rows at an inverse temperature. */
        [[nodiscard]] double loss_of(SushiRuntime::span<const float> logits,
                                     SushiRuntime::span<const std::int32_t> labels,
                                     std::size_t classes, double inverse_temperature) noexcept
        {
            double total = 0.0;

            for (std::size_t row = 0; row < labels.size(); ++row)
            {
                const float* const values = logits.data() + row * classes;
                const std::size_t best = Detail::row_argmax(values, classes);
                const double excess =
                    Detail::exponential_excess(values, classes, best, inverse_temperature);
                const double gap =
                    static_cast<double>(values[best]) - static_cast<double>(values[labels[row]]);

                // The row's loss is ln(1 + excess) + b * (best logit - label logit).
                total += std::log1p(excess) + inverse_temperature * gap;
            }

            return total / static_cast<double>(labels.size());
        }

        /** @brief Returns the loss of checked rows at the temperature exp(@p log_temperature). */
        [[nodiscard]] double loss_at_log(SushiRuntime::span<const float> logits,
                                         SushiRuntime::span<const std::int32_t> labels,
                                         std::size_t classes, double log_temperature) noexcept
        {
            return loss_of(logits, labels, classes, std::exp(-log_temperature));
        }
    } // namespace

    Result<double> mean_negative_log_likelihood(SushiRuntime::span<const float> logits,
                                                SushiRuntime::span<const std::int32_t> labels,
                                                std::int64_t classes,
                                                double temperature) noexcept
    {
        SA_TRY(Detail::check_logit_rows(logits, labels, classes, temperature, K_LOSS));

        return loss_of(logits, labels, static_cast<std::size_t>(classes), 1.0 / temperature);
    }

    Result<double> fit_temperature(SushiRuntime::span<const float> logits,
                                   SushiRuntime::span<const std::int32_t> labels,
                                   std::int64_t classes) noexcept
    {
        SA_TRY(Detail::check_logit_rows(logits, labels, classes, 1.0, K_FIT));

        const std::size_t width = static_cast<std::size_t>(classes);
        double low = std::log(K_MIN_TEMPERATURE);
        double high = std::log(K_MAX_TEMPERATURE);

        double left = high - K_KEPT_SHARE * (high - low);
        double right = low + K_KEPT_SHARE * (high - low);
        double left_loss = loss_at_log(logits, labels, width, left);
        double right_loss = loss_at_log(logits, labels, width, right);

        while (high - low > K_TEMPERATURE_LOG_TOLERANCE)
        {
            if (left_loss < right_loss)
            {
                // The minimum is in [low, right]; the old left point is the new right point.
                high = right;
                right = left;
                right_loss = left_loss;
                left = high - K_KEPT_SHARE * (high - low);
                left_loss = loss_at_log(logits, labels, width, left);
            }
            else
            {
                // The minimum is in [left, high]; the old right point is the new left point.
                low = left;
                left = right;
                left_loss = right_loss;
                right = low + K_KEPT_SHARE * (high - low);
                right_loss = loss_at_log(logits, labels, width, right);
            }
        }

        return std::exp((low + high) / 2.0);
    }
} // namespace SushiAI::Language::Training
