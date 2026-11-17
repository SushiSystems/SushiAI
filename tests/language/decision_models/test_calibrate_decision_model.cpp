/**************************************************************************/
/* test_calibrate_decision_model.cpp                                      */
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
 * @file test_calibrate_decision_model.cpp
 * @brief Tests that a trained decision model answers its held-out split and that calibration fits and stores its temperatures.
 * @author Mustafa Garip
 */

#include <optional>
#include <gtest/gtest.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/decision_models/calibrate_decision_model.hpp>
#include <SushiAI/language/decision_models/decision_model.hpp>
#include "decision_fixture.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::DecisionModels;
using DecisionTest::TrainedModel;
using DecisionTest::train_and_calibrate;
// SaTest stays reachable inside a TEST_F body; see tests/language/decision_models/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /**
     * @brief Holds the share of held-out choices answered right below which the model has
     *        learned nothing of the key rule. Unmeasured: chance is 1/3 over 2 to 4 options.
     */
    constexpr double K_CHOICE_ACCURACY_FLOOR = 0.85;

    /**
     * @brief Holds the share of held-out yes-or-no questions answered right below which the
     *        model has learned nothing of the majority rule. Unmeasured: chance is 1/2.
     */
    constexpr double K_YES_NO_ACCURACY_FLOOR = 0.90;

    /**
     * @brief Holds the mean absolute error of the held-out numbers above which the model has
     *        learned nothing of the mean rule. Unmeasured: predicting 0 gives about 0.29.
     */
    constexpr double K_NUMBER_ERROR_BOUND = 0.15;

    /** @brief Trains one session for the cases of the suite, which only read its scores. */
    class Large_DecisionModel : public SaTest::EngineTest
    {
        protected:
            /** @brief Trains, calibrates and scores once, before the first case. */
            static void SetUpTestSuite()
            {
                outcome_.emplace(train_and_calibrate());
            }

            /** @brief Holds what SetUpTestSuite() made, or why it could not. */
            inline static std::optional<Result<TrainedModel>> outcome_;
    };
} // namespace

/** @brief Tests that the held-out split is answered above the floor of each type of question. */
TEST_F(Large_DecisionModel, AnswersEachTypeOfQuestionOfTheHeldOutSplit)
{
    ASSERT_TRUE(outcome_->has_value()) << outcome_->status().message();
    const TrainedModel& trained = outcome_->value();

    EXPECT_GE(trained.calibrated.choice_accuracy, K_CHOICE_ACCURACY_FLOOR);
    EXPECT_GE(trained.calibrated.yes_no_accuracy, K_YES_NO_ACCURACY_FLOOR);
    EXPECT_LE(trained.calibrated.number_error, K_NUMBER_ERROR_BOUND);
}

/** @brief Tests the expected calibration error of each classifying head on the held-out split. */
TEST_F(Large_DecisionModel, TheCalibrationErrorAfterCalibrationIsAtMostTheBound)
{
    ASSERT_TRUE(outcome_->has_value()) << outcome_->status().message();
    const TrainedModel& trained = outcome_->value();

    EXPECT_LE(trained.calibrated.choice_ece, K_ECE_BOUND);
    EXPECT_LE(trained.calibrated.yes_no_ece, K_ECE_BOUND);
}

/** @brief Tests that the model file holds the temperatures the calibration step returned. */
TEST_F(Large_DecisionModel, TheModelFileHoldsTheFittedTemperatures)
{
    ASSERT_TRUE(outcome_->has_value()) << outcome_->status().message();
    const TrainedModel& trained = outcome_->value();

    EXPECT_EQ(trained.stored.choice, trained.fitted.choice);
    EXPECT_EQ(trained.stored.yes_no, trained.fitted.yes_no);
}
