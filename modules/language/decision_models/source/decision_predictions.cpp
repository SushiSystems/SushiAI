/**************************************************************************/
/* decision_predictions.cpp                                               */
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
 * @file decision_predictions.cpp
 * @brief Implements the rows of the choice head and of the yes-or-no head.
 * @author Mustafa Garip
 */

#include <SushiAI/language/decision_models/decision_predictions.hpp>

#include <cstddef>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI::Language::DecisionModels
{
    namespace
    {
        /** @brief Counts the classes of a yes-or-no row: no and yes. */
        constexpr std::int64_t K_YES_NO_CLASSES = 2;

        /** @brief Fails unless every vector of @p predictions agrees with its types. */
        [[nodiscard]] Status check_counts(const DecisionPredictions& predictions) noexcept
        {
            const std::size_t rows = predictions.types.size();
            SA_FAIL_IF(predictions.options < 1, StatusCode::InvalidDescription,
                       "the predictions state {} options for each choice", predictions.options);
            SA_FAIL_IF(predictions.choice_logits.size()
                           != rows * static_cast<std::size_t>(predictions.options)
                           || predictions.yes_no_logits.size() != rows
                           || predictions.numbers.size() != rows
                           || predictions.labels.size() != rows
                           || predictions.values.size() != rows,
                       StatusCode::InvalidDescription,
                       "the predictions hold {} types, {} choice logits of {} options, {} "
                       "yes-or-no logits, {} numbers, {} labels and {} values",
                       rows, predictions.choice_logits.size(), predictions.options,
                       predictions.yes_no_logits.size(), predictions.numbers.size(),
                       predictions.labels.size(), predictions.values.size());

            return Status();
        }
    } // namespace

    Result<LogitRows> choice_rows(const DecisionPredictions& predictions) noexcept
    {
        SA_TRY(check_counts(predictions));

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            LogitRows rows;
            rows.classes = predictions.options;
            const std::size_t width = static_cast<std::size_t>(predictions.options);
            for (std::size_t row = 0; row < predictions.types.size(); ++row)
            {
                if (predictions.types[row] != Datasets::DecisionType::Choice)
                    continue;

                const auto first = predictions.choice_logits.begin()
                                   + static_cast<std::ptrdiff_t>(row * width);
                rows.logits.insert(rows.logits.end(), first,
                                   first + static_cast<std::ptrdiff_t>(width));
                rows.labels.push_back(predictions.labels[row]);
            }

            return rows;
        });
    }

    Result<LogitRows> yes_no_rows(const DecisionPredictions& predictions) noexcept
    {
        SA_TRY(check_counts(predictions));

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            LogitRows rows;
            rows.classes = K_YES_NO_CLASSES;
            for (std::size_t row = 0; row < predictions.types.size(); ++row)
            {
                if (predictions.types[row] != Datasets::DecisionType::YesNo)
                    continue;

                rows.logits.push_back(0.0f);
                rows.logits.push_back(predictions.yes_no_logits[row]);
                rows.labels.push_back(predictions.values[row] > 0.5f ? 1 : 0);
            }

            return rows;
        });
    }
} // namespace SushiAI::Language::DecisionModels
