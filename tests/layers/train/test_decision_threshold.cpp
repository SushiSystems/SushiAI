/**************************************************************************/
/* test_decision_threshold.cpp                                            */
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
 * @file test_decision_threshold.cpp
 * @brief Tests the range validate_threshold accepts.
 * @author Mustafa Garip
 */

#include <limits>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/train/decision_threshold.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Holds a name for a threshold row and the threshold. */
    struct ThresholdRow
    {
        std::string name;       /**< The row's name. */
        double threshold;       /**< The threshold validated. */
    };

    /** @brief Validates each threshold inside the open interval. */
    class Small_DecisionThresholdAccepted : public ::testing::TestWithParam<ThresholdRow>
    {
    };

    /** @brief Validates each threshold on or outside the bounds. */
    class Small_DecisionThresholdBounds : public ::testing::TestWithParam<ThresholdRow>
    {
    };

    /** @brief Names a row by its name. */
    std::string row_name(const ::testing::TestParamInfo<ThresholdRow>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that the open interval (0, 1) is accepted right up to its bounds. */
TEST_P(Small_DecisionThresholdAccepted, ProbabilitiesInsideTheOpenIntervalAreAccepted)
{
    SA_EXPECT_OK(Train::validate_threshold(GetParam().threshold));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DecisionThresholdAccepted,
                         ::testing::Values(ThresholdRow{"NearZero", 1e-9},
                                           ThresholdRow{"AQuarter", 0.25},
                                           ThresholdRow{"NearOne", 1.0 - 1e-9}),
                         row_name);

/** @brief Tests that both bounds, values outside them and a NaN are refused. */
TEST_P(Small_DecisionThresholdBounds, TheBoundsAndANaNAreRefused)
{
    SA_EXPECT_FAILURE(Train::validate_threshold(GetParam().threshold),
                      StatusCode::InvalidDescription, "above 0 and below 1");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DecisionThresholdBounds,
    ::testing::Values(ThresholdRow{"Zero", 0.0}, ThresholdRow{"One", 1.0},
                      ThresholdRow{"BelowZero", -0.1}, ThresholdRow{"AboveOne", 2.0},
                      ThresholdRow{"Infinity", std::numeric_limits<double>::infinity()},
                      ThresholdRow{"NaN", std::numeric_limits<double>::quiet_NaN()}),
    row_name);
