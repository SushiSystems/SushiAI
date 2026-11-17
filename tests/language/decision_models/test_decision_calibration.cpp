/**************************************************************************/
/* test_decision_calibration.cpp                                          */
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
 * @file test_decision_calibration.cpp
 * @brief Tests the temperatures of a decision model: the fit, the model file and their bounds.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/model_writer.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/decision_dataset.hpp>
#include <SushiAI/language/decision_models/decision_calibration.hpp>
#include <SushiAI/language/decision_models/decision_predictions.hpp>
#include <SushiAI/language/decision_models/decision_scores.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include "decision_prediction_rows.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::DecisionModels;
using DecisionTest::add_row;
using DecisionTest::overconfident_predictions;
using Language::Datasets::DecisionType;
// SaTest stays reachable inside a TEST body; see tests/language/decision_models/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds a one-row model file the calibration cases store temperatures into. */
    constexpr std::string_view K_ONE_ROW_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [8], "dtype": "int32"},
        "layers": [{"type": "decision_heads", "vocabulary": 32, "dim": 16, "heads": 2,
                    "blocks": 1, "options": 4}],
        "task": {"type": "decision"}
    })";
} // namespace

/** @brief Tests that the temperatures a model file holds are those that were stored. */
TEST(Small_DecisionCalibration, TheTemperaturesSurviveTheModelFile)
{
    Config::ModelSpec spec = SaTest::model_from(K_ONE_ROW_MODEL);
    SA_EXPECT_OK(store_calibration(spec, DecisionCalibration{1.25, 0.5}));

    SA_ASSERT_OK_AND_ASSIGN(const std::string text, Config::write_model(spec));
    SA_ASSERT_OK_AND_ASSIGN(const Config::ModelSpec reread, SaTest::try_model_from(text));
    SA_ASSERT_OK_AND_ASSIGN(const DecisionCalibration read, read_calibration(reread));

    EXPECT_EQ(read.choice, 1.25);
    EXPECT_EQ(read.yes_no, 0.5);
}

/** @brief Tests that a file with no temperatures reads as the identity temperature. */
TEST(Small_DecisionCalibration, AFileWithNoTemperaturesReadsOne)
{
    const Config::ModelSpec spec = SaTest::model_from(K_ONE_ROW_MODEL);

    SA_ASSERT_OK_AND_ASSIGN(const DecisionCalibration read, read_calibration(spec));
    EXPECT_EQ(read.choice, 1.0);
    EXPECT_EQ(read.yes_no, 1.0);
}

/** @brief Tests that a task of another type is neither read nor overwritten. */
TEST(Small_DecisionCalibration, ATaskOfAnotherTypeIsRefused)
{
    Config::ModelSpec spec = SaTest::model_from(R"({
        "format_version": 2,
        "input": {"shape": [8], "dtype": "int32"},
        "layers": [{"type": "decision_heads", "vocabulary": 32, "dim": 16, "heads": 2,
                    "blocks": 1, "options": 4}],
        "task": {"type": "next_token_generation"}
    })");

    SA_EXPECT_FAILURE(store_calibration(spec, DecisionCalibration{}), StatusCode::InvalidDescription);
    SA_EXPECT_FAILURE(read_calibration(spec), StatusCode::InvalidDescription);
}

/** @brief Tests that a temperature outside the bounds of the fit is refused. */
TEST(Small_DecisionCalibration, ATemperatureOutsideTheBoundsIsRefused)
{
    Config::ModelSpec spec = SaTest::model_from(K_ONE_ROW_MODEL);

    SA_EXPECT_FAILURE(store_calibration(spec, DecisionCalibration{0.0, 1.0}),
                     StatusCode::InvalidDescription);
}

/**
 * @brief Tests that the fit heats a model that is sure and wrong a quarter of the time.
 *
 * With 6 of 8 rows right the maximiser of the likelihood puts the confidence at 3/4, so the
 * temperature divides a logit of 6 into ln 3 and is above 1; the expected calibration error falls.
 */
TEST(Small_DecisionCalibration, TheFitHeatsAnOverconfidentModel)
{
    const DecisionPredictions predictions = overconfident_predictions();

    SA_ASSERT_OK_AND_ASSIGN(const DecisionCalibration fitted,
                            fit_decision_calibration(predictions));
    SA_ASSERT_OK_AND_ASSIGN(const DecisionScores before,
                            score_decisions(predictions, DecisionCalibration{}));
    SA_ASSERT_OK_AND_ASSIGN(const DecisionScores after, score_decisions(predictions, fitted));

    EXPECT_GT(fitted.choice, 1.0);
    EXPECT_GT(fitted.yes_no, 1.0);
    EXPECT_LT(after.choice_ece, before.choice_ece);
    EXPECT_LT(after.yes_no_ece, before.yes_no_ece);
}

/** @brief Tests that a split with no yes-or-no row cannot calibrate that head. */
TEST(Small_DecisionCalibration, ASplitWithNoYesNoRowIsRefused)
{
    DecisionPredictions predictions;
    predictions.options = 2;
    add_row(predictions, DecisionType::Choice, {1.0f, 0.0f}, 0, 0.0f, 0.0f, 0.0f);
    add_row(predictions, DecisionType::Choice, {0.0f, 1.0f}, 1, 0.0f, 0.0f, 0.0f);

    SA_EXPECT_FAILURE(fit_decision_calibration(predictions), StatusCode::InvalidDescription);
}
