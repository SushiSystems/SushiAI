/**************************************************************************/
/* logit_rows.hpp                                                         */
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
 * @file logit_rows.hpp
 * @brief Defines the checks and the row sums the calibration units share over a set of logits.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>

/** @brief Holds the units of the language training module that no caller names. */
namespace SushiAI::Language::Training::Detail
{
    /**
     * @brief Fails unless the logits, the labels and the temperature describe a set to score.
     * @param who Names the calling unit in a message.
     * @return Success, or InvalidDescription for an empty set, fewer than two classes, a logit
     *         count that is not rows * classes, a non-finite logit, a label outside
     *         [0, classes) naming its row, or a temperature that is not positive and finite.
     */
    [[nodiscard]] inline Status check_logit_rows(SushiRuntime::span<const float> logits,
                                                 SushiRuntime::span<const std::int32_t> labels,
                                                 std::int64_t classes, double temperature,
                                                 std::string_view who)
    {
        SA_FAIL_IF(classes < 2, StatusCode::InvalidDescription,
                   "{}: {} classes; a decision needs at least 2", who, classes);
        SA_FAIL_IF(labels.empty(), StatusCode::InvalidDescription,
                   "{}: the set holds no row", who);
        SA_FAIL_IF(logits.size() != labels.size() * static_cast<std::size_t>(classes),
                   StatusCode::InvalidDescription,
                   "{}: {} logits are no {} rows of {} classes", who, logits.size(),
                   labels.size(), classes);
        SA_FAIL_IF(!std::isfinite(temperature) || temperature <= 0.0,
                   StatusCode::InvalidDescription,
                   "{}: a temperature of {}; it is finite and positive", who, temperature);

        for (std::size_t i = 0; i < logits.size(); ++i)
        {
            SA_FAIL_IF(!std::isfinite(logits[i]), StatusCode::InvalidDescription,
                       "{}: the logit at element {} is not finite", who, i);
        }

        for (std::size_t row = 0; row < labels.size(); ++row)
        {
            SA_FAIL_IF(labels[row] < 0 || labels[row] >= classes, StatusCode::InvalidDescription,
                       "{}: row {} has the label {}; it is in [0, {})", who, row, labels[row],
                       classes);
        }

        return Status();
    }

    /** @brief Returns the index of the first largest of the @p classes logits of a row. */
    [[nodiscard]] inline std::size_t row_argmax(const float* row, std::size_t classes) noexcept
    {
        return static_cast<std::size_t>(std::max_element(row, row + classes) - row);
    }

    /**
     * @brief Returns the sum of exp(inverse_temperature * (logit - best logit)) over the rest.
     *
     * The softmax normaliser of a row is 1 plus this sum, where @p best is the index of its largest
     * logit. Keeping the 1 apart lets a caller take log1p of the sum, so a tail too small to move
     * 1 still counts. Every exponent is at or below 0, so the sum never overflows.
     */
    [[nodiscard]] inline double exponential_excess(const float* row, std::size_t classes,
                                                   std::size_t best,
                                                   double inverse_temperature) noexcept
    {
        double sum = 0.0;
        for (std::size_t j = 0; j < classes; ++j)
        {
            if (j != best)
            {
                sum += std::exp(inverse_temperature
                                * (static_cast<double>(row[j]) - static_cast<double>(row[best])));
            }
        }

        return sum;
    }
} // namespace SushiAI::Language::Training::Detail
