/**************************************************************************/
/* test_decision_scores.cpp                                               */
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
 * @file test_decision_scores.cpp
 * @brief Tests the scores of a decision model against hand counts.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/decision_dataset.hpp>
#include <SushiAI/language/decision_models/decision_calibration.hpp>
#include <SushiAI/language/decision_models/decision_predictions.hpp>
#include <SushiAI/language/decision_models/decision_scores.hpp>
#include "decision_prediction_rows.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::DecisionModels;
using DecisionTest::add_row;
using DecisionTest::overconfident_predictions;
using Language::Datasets::DecisionType;

/** @brief Tests the scores of a hand-built set: 1 of 2 choices, 2 of 3 yes-or-no rows, two numbers. */
TEST(Small_DecisionScores, TheScoresEqualTheHandCounts)
{
    DecisionPredictions predictions;
    predictions.options = 2;
    add_row(predictions, DecisionType::Choice, {3.0f, 1.0f}, 0, 0.0f, 0.0f, 0.0f);
    add_row(predictions, DecisionType::Choice, {1.0f, 3.0f}, 0, 0.0f, 0.0f, 0.0f);
    add_row(predictions, DecisionType::YesNo, {0.0f, 0.0f}, 0, 2.0f, 0.0f, 1.0f);
    add_row(predictions, DecisionType::YesNo, {0.0f, 0.0f}, 0, -2.0f, 0.0f, 0.0f);
    add_row(predictions, DecisionType::YesNo, {0.0f, 0.0f}, 0, 1.0f, 0.0f, 0.0f);
    add_row(predictions, DecisionType::Number, {0.0f, 0.0f}, 0, 0.0f, 0.5f, 0.25f);
    add_row(predictions, DecisionType::Number, {0.0f, 0.0f}, 0, 0.0f, -1.0f, -0.5f);

    SA_ASSERT_OK_AND_ASSIGN(const DecisionScores scores,
                            score_decisions(predictions, DecisionCalibration{}));

    EXPECT_EQ(scores.choice_rows, 2);
    EXPECT_EQ(scores.yes_no_rows, 3);
    EXPECT_EQ(scores.number_rows, 2);
    EXPECT_EQ(scores.choice_accuracy, 0.5);
    EXPECT_EQ(scores.yes_no_accuracy, 2.0 / 3.0);
    // The errors are 0.25 and 0.5, both exact in float32, and so is their mean.
    EXPECT_EQ(scores.number_error, 0.375);
}

/** @brief Tests that dividing the logits by a temperature changes no answer. */
TEST(Small_DecisionScores, ATemperatureChangesNoAccuracy)
{
    const DecisionPredictions predictions = overconfident_predictions();

    SA_ASSERT_OK_AND_ASSIGN(const DecisionScores plain,
                            score_decisions(predictions, DecisionCalibration{}));
    SA_ASSERT_OK_AND_ASSIGN(const DecisionScores heated,
                            score_decisions(predictions, DecisionCalibration{5.0, 3.0}));

    EXPECT_EQ(heated.choice_accuracy, plain.choice_accuracy);
    EXPECT_EQ(heated.yes_no_accuracy, plain.yes_no_accuracy);
}
