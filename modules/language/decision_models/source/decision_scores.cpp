/**************************************************************************/
/* decision_scores.cpp                                                    */
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
 * @file decision_scores.cpp
 * @brief Implements the scores of a decision model over its predictions.
 * @author Mustafa Garip
 */

#include <SushiAI/language/decision_models/decision_scores.hpp>

#include <cmath>
#include <cstddef>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/training/calibration_error.hpp>

namespace SushiAI::Language::DecisionModels
{
    namespace
    {
        /** @brief Returns the share of rows whose largest logit, the first on a tie, is the label. */
        [[nodiscard]] double accuracy_of(const LogitRows& rows) noexcept
        {
            const std::size_t width = static_cast<std::size_t>(rows.classes);
            std::size_t hits = 0;
            for (std::size_t row = 0; row < rows.labels.size(); ++row)
            {
                std::size_t best = 0;
                for (std::size_t column = 1; column < width; ++column)
                {
                    if (rows.logits[row * width + column] > rows.logits[row * width + best])
                        best = column;
                }

                if (static_cast<std::int32_t>(best) == rows.labels[row])
                    ++hits;
            }

            return static_cast<double>(hits) / static_cast<double>(rows.labels.size());
        }

        /** @brief Returns the expected calibration error of @p rows at @p temperature. */
        [[nodiscard]] Result<double> ece_of(const LogitRows& rows, double temperature) noexcept
        {
            return Training::expected_calibration_error(
                SushiRuntime::span<const float>(rows.logits.data(), rows.logits.size()),
                SushiRuntime::span<const std::int32_t>(rows.labels.data(), rows.labels.size()),
                rows.classes, temperature);
        }

        /** @brief Returns the mean absolute error of the numbers of the rows of type Number. */
        [[nodiscard]] Result<double> number_error_of(const DecisionPredictions& predictions,
                                                     std::int64_t& rows) noexcept
        {
            double total = 0.0;
            rows = 0;
            for (std::size_t row = 0; row < predictions.types.size(); ++row)
            {
                if (predictions.types[row] != Datasets::DecisionType::Number)
                    continue;

                total += std::fabs(static_cast<double>(predictions.numbers[row])
                                   - static_cast<double>(predictions.values[row]));
                ++rows;
            }

            SA_FAIL_IF(rows == 0, StatusCode::InvalidDescription,
                       "the predictions hold no row of type Number");

            return total / static_cast<double>(rows);
        }
    } // namespace

    Result<DecisionScores> score_decisions(const DecisionPredictions& predictions,
                                           const DecisionCalibration& calibration) noexcept
    {
        SA_TRY_ASSIGN(const LogitRows choice, choice_rows(predictions));
        SA_TRY_ASSIGN(const LogitRows yes_no, yes_no_rows(predictions));
        SA_FAIL_IF(choice.labels.empty() || yes_no.labels.empty(), StatusCode::InvalidDescription,
                   "the predictions hold {} choice rows and {} yes-or-no rows; each head needs "
                   "one",
                   choice.labels.size(), yes_no.labels.size());

        DecisionScores scores;
        scores.choice_rows = static_cast<std::int64_t>(choice.labels.size());
        scores.yes_no_rows = static_cast<std::int64_t>(yes_no.labels.size());
        scores.choice_accuracy = accuracy_of(choice);
        scores.yes_no_accuracy = accuracy_of(yes_no);
        SA_TRY_ASSIGN(scores.number_error, number_error_of(predictions, scores.number_rows));
        SA_TRY_ASSIGN(scores.choice_ece, ece_of(choice, calibration.choice));
        SA_TRY_ASSIGN(scores.yes_no_ece, ece_of(yes_no, calibration.yes_no));

        return scores;
    }
} // namespace SushiAI::Language::DecisionModels
