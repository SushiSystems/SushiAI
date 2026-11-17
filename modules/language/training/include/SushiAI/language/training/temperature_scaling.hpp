/**************************************************************************/
/* temperature_scaling.hpp                                                */
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
 * @file temperature_scaling.hpp
 * @brief Declares the temperature fit of a held-out set and the loss it minimises.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>

namespace SushiAI::Language::Training
{
    /** @brief Holds the smallest temperature fit_temperature() returns. */
    inline constexpr double K_MIN_TEMPERATURE = 0.01;

    /** @brief Holds the largest temperature fit_temperature() returns. */
    inline constexpr double K_MAX_TEMPERATURE = 100.0;

    /** @brief Holds the width of the bracket of ln T at which fit_temperature() stops. */
    inline constexpr double K_TEMPERATURE_LOG_TOLERANCE = 1.0e-8;

    /**
     * @brief Returns the mean negative log-likelihood of the labels under softmax(logits / T).
     * @param logits [rows, classes] row-major, every value finite.
     * @param labels One class index per row, each in [0, classes).
     * @param temperature Finite and positive.
     * @return The loss in nats; InvalidDescription for a set that is no rows of at least two
     *         classes, a non-finite logit, a label outside [0, classes) or a bad temperature.
     */
    [[nodiscard]] Result<double> mean_negative_log_likelihood(
        SushiRuntime::span<const float> logits, SushiRuntime::span<const std::int32_t> labels,
        std::int64_t classes, double temperature) noexcept;

    /**
     * @brief Returns the temperature T that minimises the loss of softmax(logits / T).
     * @param logits [rows, classes] row-major, every value finite.
     * @param labels One class index per row, each in [0, classes).
     * @return T from a deterministic search of ln T, or what the loss refuses. All rows right
     *         gives K_MIN_TEMPERATURE; all rows wrong gives K_MAX_TEMPERATURE.
     * @see modules/language/training/README.md, section "Calibration".
     */
    [[nodiscard]] Result<double> fit_temperature(SushiRuntime::span<const float> logits,
                                                 SushiRuntime::span<const std::int32_t> labels,
                                                 std::int64_t classes) noexcept;
} // namespace SushiAI::Language::Training
