/**************************************************************************/
/* calibration_error.hpp                                                  */
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
 * @file calibration_error.hpp
 * @brief Declares the expected calibration error of a set of logits.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>

namespace SushiAI::Language::Training
{
    /** @brief Counts the equal-width confidence bins of the expected calibration error. */
    inline constexpr std::int64_t K_CALIBRATION_BINS = 15;

    /**
     * @brief Returns the expected calibration error of softmax(logits / T) against the labels.
     * @param logits [rows, classes] row-major, every value finite.
     * @param labels One class index per row, each in [0, classes).
     * @param temperature Finite and positive; 1 leaves the logits as they are.
     * @return A value in [0, 1]; InvalidDescription for what mean_negative_log_likelihood() refuses.
     * @see modules/language/training/README.md, section "Calibration", for the bins.
     */
    [[nodiscard]] Result<double> expected_calibration_error(
        SushiRuntime::span<const float> logits, SushiRuntime::span<const std::int32_t> labels,
        std::int64_t classes, double temperature = 1.0) noexcept;
} // namespace SushiAI::Language::Training
