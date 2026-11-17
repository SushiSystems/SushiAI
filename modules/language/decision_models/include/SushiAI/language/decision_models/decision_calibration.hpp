/**************************************************************************/
/* decision_calibration.hpp                                               */
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
 * @file decision_calibration.hpp
 * @brief Declares the temperatures of the two classifying heads, their fit and their place in
 *        the model file.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>

#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/decision_models/decision_predictions.hpp>

namespace SushiAI::Language::DecisionModels
{
    /** @brief Names the "type" of the "task" object of a decision model file. */
    inline constexpr std::string_view K_DECISION_TASK_TYPE = "decision";

    /** @brief Names the key of the "task" object that holds the temperatures. */
    inline constexpr std::string_view K_CALIBRATION_KEY = "calibration";

    /** @brief Names the key of the calibration object that holds the temperature of the choice. */
    inline constexpr std::string_view K_CHOICE_TEMPERATURE_KEY = "choice";

    /** @brief Names the key of the calibration object that holds the temperature of the yes-or-no. */
    inline constexpr std::string_view K_YES_NO_TEMPERATURE_KEY = "yes_no";

    /** @brief Holds one temperature for each classifying head; 1 leaves the logits as they are. */
    struct DecisionCalibration
    {
        double choice = 1.0; /**< Divides the logits of the options before their softmax. */
        double yes_no = 1.0; /**< Divides the logit of a yes before its sigmoid. */
    };

    /**
     * @brief Fits one temperature for each classifying head on the rows of its own type.
     * @param predictions What the model answered over the calibration split.
     * @return The temperatures, or InvalidDescription when the split holds no choice row or no
     *         yes-or-no row, or what fit_temperature() refuses.
     */
    [[nodiscard]] Result<DecisionCalibration> fit_decision_calibration(
        const DecisionPredictions& predictions) noexcept;

    /**
     * @brief Writes the temperatures into the "task" object of @p spec, as
     *        {"calibration": {"choice": T, "yes_no": T}}, and keeps its other keys.
     * @param spec A model file whose task is absent or of the type "decision".
     * @return Success, or InvalidDescription for a task of another type or a temperature that
     *         is not finite and inside the bounds of fit_temperature().
     * @see modules/language/decision_models/README.md, section "Calibration".
     */
    [[nodiscard]] Status store_calibration(Config::ModelSpec& spec,
                                           const DecisionCalibration& calibration) noexcept;

    /**
     * @brief Reads the temperatures store_calibration() wrote.
     * @return The temperatures, 1 and 1 for a "decision" task that holds none, or
     *         InvalidDescription for a task of another type, a key it does not accept or a
     *         temperature that is no number inside the bounds of fit_temperature().
     */
    [[nodiscard]] Result<DecisionCalibration> read_calibration(
        const Config::ModelSpec& spec) noexcept;
} // namespace SushiAI::Language::DecisionModels
