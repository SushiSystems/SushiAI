/**************************************************************************/
/* decision_scores.hpp                                                    */
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
 * @file decision_scores.hpp
 * @brief Declares the scores of a decision model over a dataset: the accuracy of each
 *        classifying head, the error of the number head and the calibration error.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/decision_models/decision_calibration.hpp>
#include <SushiAI/language/decision_models/decision_predictions.hpp>

namespace SushiAI::Language::DecisionModels
{
    /** @brief Holds what one dataset says of a model, each score over the rows of its own type. */
    struct DecisionScores
    {
        double choice_accuracy = 0.0;  /**< The share of choices whose largest logit is the label. */
        double yes_no_accuracy = 0.0;  /**< The share of yes-or-no rows whose logit has the label's sign. */
        double number_error = 0.0;     /**< The mean absolute error of the numbers. */
        double choice_ece = 0.0;       /**< The expected calibration error of the choice head. */
        double yes_no_ece = 0.0;       /**< The expected calibration error of the yes-or-no head. */
        std::int64_t choice_rows = 0;  /**< Counts the choice rows. */
        std::int64_t yes_no_rows = 0;  /**< Counts the yes-or-no rows. */
        std::int64_t number_rows = 0;  /**< Counts the number rows. */
    };

    /**
     * @brief Scores @p predictions with the logits of each head divided by its temperature.
     * @param calibration The temperatures; the accuracies do not depend on them.
     * @return The scores, or InvalidDescription for predictions that hold no row of a type or
     *         disagree in count, or what expected_calibration_error() refuses.
     */
    [[nodiscard]] Result<DecisionScores> score_decisions(
        const DecisionPredictions& predictions, const DecisionCalibration& calibration) noexcept;
} // namespace SushiAI::Language::DecisionModels
