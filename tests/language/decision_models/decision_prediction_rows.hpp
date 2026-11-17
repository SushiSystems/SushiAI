/**************************************************************************/
/* decision_prediction_rows.hpp                                           */
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
 * @file decision_prediction_rows.hpp
 * @brief Declares the prediction rows the decision-model tests build by hand.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <vector>
#include <SushiAI/language/datasets/decision_dataset.hpp>
#include <SushiAI/language/decision_models/decision_predictions.hpp>

namespace DecisionTest
{
    using SushiAI::Language::DecisionModels::DecisionPredictions;
    using SushiAI::Language::Datasets::DecisionType;

    /** @brief Appends a row of every column to @p predictions; the columns of other types hold 0. */
    inline void add_row(DecisionPredictions& predictions, DecisionType type,
                        const std::vector<float>& choice_logits, std::int32_t label,
                        float yes_no_logit, float number, float value)
    {
        predictions.choice_logits.insert(predictions.choice_logits.end(), choice_logits.begin(),
                                         choice_logits.end());
        predictions.yes_no_logits.push_back(yes_no_logit);
        predictions.numbers.push_back(number);
        predictions.labels.push_back(label);
        predictions.values.push_back(value);
        predictions.types.push_back(type);
    }

    /**
     * @brief Returns the predictions of a model that is sure and often wrong: of 8 choices and of
     *        8 yes-or-no rows, 6 are answered right with a logit of 6 and 2 wrong with a logit
     *        of 6, and one number row.
     */
    inline DecisionPredictions overconfident_predictions()
    {
        DecisionPredictions predictions;
        predictions.options = 2;
        for (int row = 0; row < 8; ++row)
        {
            const std::int32_t label = row < 6 ? 0 : 1;
            add_row(predictions, DecisionType::Choice, {6.0f, 0.0f}, label, 0.0f, 0.0f, 0.0f);
        }

        for (int row = 0; row < 8; ++row)
        {
            const float answer = row < 6 ? 1.0f : 0.0f;
            add_row(predictions, DecisionType::YesNo, {0.0f, 0.0f}, 0, 6.0f, 0.0f, answer);
        }

        add_row(predictions, DecisionType::Number, {0.0f, 0.0f}, 0, 0.0f, 0.5f, 0.25f);
        return predictions;
    }

} // namespace DecisionTest
