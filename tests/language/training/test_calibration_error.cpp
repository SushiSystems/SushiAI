/**************************************************************************/
/* test_calibration_error.cpp                                             */
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
 * @file test_calibration_error.cpp
 * @brief Tests the expected calibration error against hand-computed sets and after scaling.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "calibration_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/training/calibration_error.hpp>
#include <SushiAI/language/training/temperature_scaling.hpp>
#include <support/test_names.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Training;
namespace SaTest = SushiAI::Test;
using Support::LogitSet;
using Support::overconfident_groups;
using Support::three_to_one_margin;

namespace
{
    /** @brief Holds the threshold of 1: the derived bound itself is what the error is held to. */
    constexpr double K_WHOLE_BOUND = 1.0;

    /** @brief Holds the largest slope of the logistic function, at its middle. */
    constexpr double K_LARGEST_SLOPE = 0.25;

    /** @brief Holds the rounded operations of the sums of one evaluation. */
    constexpr std::size_t K_OPERATIONS = 8;

    /** @brief Returns four rows: margins 2 ln 3, 2 ln 3, ln 3, ln 3 with the labels 0, 1, 0, 0. */
    LogitSet hand_set()
    {
        const float wide = static_cast<float>(2.0 * three_to_one_margin());
        const float narrow = static_cast<float>(three_to_one_margin());

        LogitSet set;
        set.logits = {wide, 0.0f, wide, 0.0f, narrow, 0.0f, narrow, 0.0f};
        set.labels = {0, 1, 0, 0};
        return set;
    }

    /** @brief Names a set the metric refuses and what the refusal says. */
    struct RefusalCase
    {
        const char* label;                  /**< The name the row prints. */
        std::vector<float> logits;          /**< The logits. */
        std::vector<std::int32_t> labels;   /**< The labels. */
        double temperature;                 /**< The temperature. */
        const char* first;                  /**< A fragment the refusal names. */
        const char* second;                 /**< A second fragment, or an empty text. */
    };

    /** @brief Provides the sets and temperatures the metric refuses. */
    class Small_LanguageTrainingCalibrationErrorRefusal
        : public ::testing::TestWithParam<RefusalCase>
    {

    };

    /** @brief Returns the name a row prints. */
    template <typename Case>
    std::string label_of(const ::testing::TestParamInfo<Case>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.label);
    }

    /** @brief Returns the error of a set at a temperature, expecting the metric to succeed. */
    double expect_error_of(const LogitSet& set, double temperature)
    {
        const Result<double> error = expected_calibration_error(
            set.logits_span(), set.labels_span(), Support::K_CLASSES, temperature);
        EXPECT_TRUE(SaTest::status_is_ok(error.status()));
        return error.value();
    }
} // namespace

/** @brief Tests that the error of a hand-built set is the value worked out by hand. */
TEST(Small_LanguageTrainingCalibrationError, TheErrorOfTheHandSetIsTheHandValue)
{
    // Bin 13 holds the two rows at 9/10: |1 right - 1.8| = 0.8. Bin 11 holds the two at 3/4: 0.5.
    const double expected = (0.8 + 0.5) / 4.0;

    // The bound is the sidecar's: the float rounding of the widest margin plus the sums.
    const double bound = K_LARGEST_SLOPE * 2.0 * three_to_one_margin()
                             * TestSupport::unit_roundoff<float>()
                         + static_cast<double>(K_OPERATIONS) * TestSupport::unit_roundoff<double>()
                               * expected;

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
        expect_error_of(hand_set(), 1.0), expected, bound, K_WHOLE_BOUND)));
}

/** @brief Tests that a set calibrated exactly has no error. */
TEST(Small_LanguageTrainingCalibrationError, ACalibratedSetHasNoError)
{
    const double bound = K_LARGEST_SLOPE * three_to_one_margin()
                         * TestSupport::unit_roundoff<float>();

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
        expect_error_of(overconfident_groups(1.0, 2), 1.0), 0.0, bound, K_WHOLE_BOUND)));
}

/** @brief Tests that a confidence of exactly 1 lands in the last bin and counts a wrong row. */
TEST(Small_LanguageTrainingCalibrationError, AConfidenceOfOneIsInTheLastBin)
{
    // exp(-100) is under half a unit of 1, so the confidence is exactly 1 and the gap is 1.
    LogitSet set;
    set.logits = {100.0f, 0.0f};
    set.labels = {1};

    EXPECT_EQ(expect_error_of(set, 1.0), 1.0);
}

/** @brief Tests that dividing the logits by a temperature is the same as scaling them down. */
TEST(Small_LanguageTrainingCalibrationError, ATemperatureDividesTheLogits)
{
    // Doubling a float is exact, so factor 2 over temperature 2 is factor 1 bit for bit.
    EXPECT_EQ(expect_error_of(overconfident_groups(2.0, 2), 2.0),
              expect_error_of(overconfident_groups(1.0, 2), 1.0));
}

/** @brief Tests that scaling by the fitted temperature lowers the error of overconfident logits. */
TEST(Small_LanguageTrainingCalibrationError, ScalingByTheFittedTemperatureLowersTheError)
{
    // At factor 2 the confidence is 9/10 against an accuracy of 3/4: an error of 0.15.
    const LogitSet set = overconfident_groups(2.0, 2);
    const double expected_before = 0.9 - 0.75;
    const double bound = K_LARGEST_SLOPE * 2.0 * three_to_one_margin()
                             * TestSupport::unit_roundoff<float>()
                         + static_cast<double>(K_OPERATIONS) * TestSupport::unit_roundoff<double>()
                               * expected_before;

    SA_ASSERT_OK_AND_ASSIGN(
        const double fitted,
        fit_temperature(set.logits_span(), set.labels_span(), Support::K_CLASSES));
    const double before = expect_error_of(set, 1.0);
    const double after = expect_error_of(set, fitted);

    EXPECT_TRUE(TestSupport::is_within(
        TestSupport::measure_error(before, expected_before, bound, K_WHOLE_BOUND)));
    EXPECT_LT(after, before);
}

/** @brief Tests that a set or a temperature that does not fit is refused naming what is wrong. */
TEST_P(Small_LanguageTrainingCalibrationErrorRefusal, TheMetricRefusesWhatDoesNotFit)
{
    const RefusalCase& row = GetParam();

    SA_EXPECT_FAILURE(
        expected_calibration_error(
            SushiRuntime::span<const float>(row.logits.data(), row.logits.size()),
            SushiRuntime::span<const std::int32_t>(row.labels.data(), row.labels.size()),
            Support::K_CLASSES, row.temperature),
        StatusCode::InvalidDescription, "expected_calibration_error", row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageTrainingCalibrationErrorRefusal,
    ::testing::Values(
        RefusalCase{"NoRow", {}, {}, 1.0, "holds no row", ""},
        RefusalCase{"ALabelPastTheClasses", {1.0f, 2.0f}, {2}, 1.0, "row 0", "label 2"},
        RefusalCase{"ATemperatureOfZero", {1.0f, 2.0f}, {0}, 0.0, "temperature of 0",
                    "finite and positive"},
        RefusalCase{"ANaNTemperature", {1.0f, 2.0f}, {0},
                    std::numeric_limits<double>::quiet_NaN(), "temperature of",
                    "finite and positive"}),
    label_of<RefusalCase>);
