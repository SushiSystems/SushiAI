/**************************************************************************/
/* test_decision_predictions.cpp                                          */
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
 * @file test_decision_predictions.cpp
 * @brief Tests the rows of each classifying head that choice_rows and yes_no_rows cut out of the predictions.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/decision_dataset.hpp>
#include <SushiAI/language/decision_models/decision_predictions.hpp>
#include "decision_prediction_rows.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::DecisionModels;
using DecisionTest::add_row;
using Language::Datasets::DecisionType;

namespace
{
    /** @brief Returns a choice, a yes-or-no, a choice, a number and a yes-or-no row, in that order. */
    DecisionPredictions mixed_rows()
    {
        DecisionPredictions predictions;
        predictions.options = 2;
        add_row(predictions, DecisionType::Choice, {3.0f, 1.0f}, 0, 0.0f, 0.0f, 0.0f);
        add_row(predictions, DecisionType::YesNo, {0.0f, 0.0f}, 0, 2.0f, 0.0f, 1.0f);
        add_row(predictions, DecisionType::Choice, {1.0f, 3.0f}, 1, 0.0f, 0.0f, 0.0f);
        add_row(predictions, DecisionType::Number, {0.0f, 0.0f}, 0, 0.0f, 0.5f, 0.25f);
        add_row(predictions, DecisionType::YesNo, {0.0f, 0.0f}, 0, -2.0f, 0.0f, 0.0f);
        return predictions;
    }
} // namespace

/** @brief Tests that the predictions count their rows as the records they hold. */
TEST(Small_DecisionPredictions, TheRowsAreTheRecordsTheyHold)
{
    EXPECT_EQ(DecisionPredictions{}.rows(), 0);
    EXPECT_EQ(mixed_rows().rows(), 5);
}

/** @brief Tests that choice_rows keeps the choice rows, in their order, with their labels. */
TEST(Small_DecisionPredictions, ChoiceRowsHoldTheLogitsAndLabelsOfTheChoicesInOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const LogitRows rows, choice_rows(mixed_rows()));

    EXPECT_EQ(rows.classes, 2);
    EXPECT_EQ(rows.logits, (std::vector<float>{3.0f, 1.0f, 1.0f, 3.0f}));
    EXPECT_EQ(rows.labels, (std::vector<std::int32_t>{0, 1}));
}

/** @brief Tests that yes_no_rows turns each logit z into the row {0, z} and the value into a label. */
TEST(Small_DecisionPredictions, YesNoRowsHoldZeroAndTheLogitWithTheAnswerAsLabel)
{
    SA_ASSERT_OK_AND_ASSIGN(const LogitRows rows, yes_no_rows(mixed_rows()));

    EXPECT_EQ(rows.classes, 2);
    EXPECT_EQ(rows.logits, (std::vector<float>{0.0f, 2.0f, 0.0f, -2.0f}));
    EXPECT_EQ(rows.labels, (std::vector<std::int32_t>{1, 0}));
}

/** @brief Tests that a value above one half is a yes and a value of one half is a no. */
TEST(Small_DecisionPredictions, AValueAboveOneHalfIsAYes)
{
    DecisionPredictions predictions;
    predictions.options = 2;
    add_row(predictions, DecisionType::YesNo, {0.0f, 0.0f}, 0, 1.0f, 0.0f, 0.75f);
    add_row(predictions, DecisionType::YesNo, {0.0f, 0.0f}, 0, 1.0f, 0.0f, 0.5f);

    SA_ASSERT_OK_AND_ASSIGN(const LogitRows rows, yes_no_rows(predictions));

    EXPECT_EQ(rows.labels, (std::vector<std::int32_t>{1, 0}));
}

/** @brief Tests that predictions of another type only give no rows and still state the width. */
TEST(Small_DecisionPredictions, AHeadWithNoRowOfItsTypeHoldsNoRow)
{
    DecisionPredictions predictions;
    predictions.options = 3;
    add_row(predictions, DecisionType::Number, {0.0f, 0.0f, 0.0f}, 0, 0.0f, 0.5f, 0.25f);

    SA_ASSERT_OK_AND_ASSIGN(const LogitRows choices, choice_rows(predictions));
    SA_ASSERT_OK_AND_ASSIGN(const LogitRows answers, yes_no_rows(predictions));

    EXPECT_TRUE(choices.logits.empty());
    EXPECT_TRUE(choices.labels.empty());
    EXPECT_EQ(choices.classes, 3);
    EXPECT_TRUE(answers.logits.empty());
    EXPECT_TRUE(answers.labels.empty());
}

/** @brief Tests that both heads refuse predictions whose vectors disagree in count. */
TEST(Small_DecisionPredictions, PredictionsWhoseVectorsDisagreeInCountAreRefused)
{
    DecisionPredictions predictions = mixed_rows();
    predictions.numbers.pop_back();

    SA_EXPECT_FAILURE(choice_rows(predictions), StatusCode::InvalidDescription, "numbers");
    SA_EXPECT_FAILURE(yes_no_rows(predictions), StatusCode::InvalidDescription, "numbers");
}

/** @brief Tests that predictions that state no option are refused. */
TEST(Small_DecisionPredictions, PredictionsThatStateNoOptionAreRefused)
{
    DecisionPredictions predictions;

    SA_EXPECT_FAILURE(choice_rows(predictions), StatusCode::InvalidDescription, "options");
}
