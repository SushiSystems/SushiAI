/**************************************************************************/
/* decision_predictions.hpp                                               */
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
 * @file decision_predictions.hpp
 * @brief Declares what a decision model answered over a dataset, and the rows of each
 *        classifying head.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/datasets/decision_file.hpp>

namespace SushiAI::Language::DecisionModels
{
    /** @brief Holds the three outputs of a model for every record of a dataset, with the answers. */
    struct DecisionPredictions
    {
        /** @brief Counts K, the options of each row of the choice logits. */
        std::int64_t options = 0;

        /** @brief Holds [rows, K] logits of the options; a padded option holds a very low one. */
        std::vector<float> choice_logits;

        /** @brief Holds [rows] logits of a yes. */
        std::vector<float> yes_no_logits;

        /** @brief Holds [rows] numbers. */
        std::vector<float> numbers;

        /** @brief Holds [rows] indices of the right option of a choice; 0 for another type. */
        std::vector<std::int32_t> labels;

        /** @brief Holds [rows] answers: 1 or 0 of a yes-or-no, the value of a number, else 0. */
        std::vector<float> values;

        /** @brief Holds [rows] types of the records. */
        std::vector<Datasets::DecisionType> types;

        /** @brief Returns how many records the predictions hold. */
        [[nodiscard]] std::int64_t rows() const noexcept
        {
            return static_cast<std::int64_t>(types.size());
        }
    };

    /** @brief Holds the rows one classifying head answered: logits, their labels and their width. */
    struct LogitRows
    {
        /** @brief Holds [rows, classes] logits, row-major. */
        std::vector<float> logits;

        /** @brief Holds one class index per row. */
        std::vector<std::int32_t> labels;

        /** @brief Counts the classes of each row. */
        std::int64_t classes = 0;
    };

    /**
     * @brief Returns the choice logits of the rows of type Choice, with their labels.
     * @return The rows, or InvalidDescription for predictions whose vectors disagree in count,
     *         or OutOfMemory.
     */
    [[nodiscard]] Result<LogitRows> choice_rows(const DecisionPredictions& predictions) noexcept;

    /**
     * @brief Returns the yes-or-no logits of the rows of type YesNo as two-class rows.
     *
     * A row holds {0, z} for the logit z of a yes, so softmax(row / T) puts sigmoid(z / T) on
     * class 1, the yes. The label is 1 for a yes and 0 for a no.
     * @return The rows, or InvalidDescription for predictions whose vectors disagree in count,
     *         or OutOfMemory.
     */
    [[nodiscard]] Result<LogitRows> yes_no_rows(const DecisionPredictions& predictions) noexcept;
} // namespace SushiAI::Language::DecisionModels
