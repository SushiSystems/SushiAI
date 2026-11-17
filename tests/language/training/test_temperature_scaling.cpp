/**************************************************************************/
/* test_temperature_scaling.cpp                                           */
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
 * @file test_temperature_scaling.cpp
 * @brief Tests the temperature fit, the loss it minimises and their refusals.
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

    /** @brief Holds the groups of four rows the sets below have. */
    constexpr std::size_t K_GROUPS = 2;

    /** @brief Holds the rows of such a set. */
    constexpr std::size_t K_ROWS = K_GROUPS * Support::K_GROUP_ROWS;

    /** @brief Holds the rounded operations of one loss evaluation beyond one per row. */
    constexpr std::size_t K_EXTRA_OPERATIONS = 8;

    /** @brief Holds the share of the rows a calibrated group gets right: 3 of 4. */
    constexpr double K_RIGHT_SHARE = 0.75;

    /** @brief Returns the least loss of a calibrated group: three rows lose ln(4/3), one ln 4. */
    double least_loss()
    {
        return (3.0 * std::log(1.0 / K_RIGHT_SHARE) + std::log(1.0 / (1.0 - K_RIGHT_SHARE)))
               / 4.0;
    }

    /**
     * @brief Returns the bound on |ln T - ln factor| of a fit over the overconfident groups.
     * @see test_temperature_scaling.md, section "TheFittedTemperatureEqualsTheFactor".
     */
    double fit_bound()
    {
        const double margin = three_to_one_margin();
        const double second_derivative = margin * margin * K_RIGHT_SHARE * (1.0 - K_RIGHT_SHARE);
        const double evaluation_error =
            static_cast<double>(K_ROWS + K_EXTRA_OPERATIONS)
            * TestSupport::unit_roundoff<double>() * least_loss();

        return K_TEMPERATURE_LOG_TOLERANCE + TestSupport::unit_roundoff<float>()
               + std::sqrt(4.0 * evaluation_error / second_derivative);
    }

    /** @brief Holds a factor of overconfidence. */
    struct FactorCase
    {
        const char* label; /**< The name the row prints. */
        double factor;     /**< The factor the logits of a calibrated set are scaled by. */
    };

    /** @brief Provides the factors of overconfidence the fit must recover. */
    class Small_LanguageTrainingTemperatureScalingFactor
        : public ::testing::TestWithParam<FactorCase>
    {

    };

    /** @brief Names a set the calibration functions refuse and what the refusal says. */
    struct SetCase
    {
        const char* label;                  /**< The name the row prints. */
        std::vector<float> logits;          /**< The logits. */
        std::vector<std::int32_t> labels;   /**< The labels. */
        std::int64_t classes;               /**< The classes. */
        const char* first;                  /**< A fragment the refusal names. */
        const char* second;                 /**< A second fragment, or an empty text. */
    };

    /** @brief Provides the sets fit_temperature refuses. */
    class Small_LanguageTrainingTemperatureScalingSet : public ::testing::TestWithParam<SetCase>
    {

    };

    /** @brief Provides the temperatures the loss refuses. */
    class Small_LanguageTrainingTemperatureScalingTemperature
        : public ::testing::TestWithParam<double>
    {

    };

    /** @brief Returns the name a row prints. */
    template <typename Case>
    std::string label_of(const ::testing::TestParamInfo<Case>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.label);
    }
} // namespace

/** @brief Tests that the fitted temperature is the factor the logits were made overconfident by. */
TEST_P(Small_LanguageTrainingTemperatureScalingFactor, TheFittedTemperatureEqualsTheFactor)
{
    const double factor = GetParam().factor;
    const LogitSet set = overconfident_groups(factor, K_GROUPS);

    SA_ASSERT_OK_AND_ASSIGN(
        const double fitted,
        fit_temperature(set.logits_span(), set.labels_span(), Support::K_CLASSES));

    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
        std::log(fitted), std::log(factor), fit_bound(), K_WHOLE_BOUND)));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LanguageTrainingTemperatureScalingFactor,
                         ::testing::Values(FactorCase{"One", 1.0}, FactorCase{"OneAndAHalf", 1.5},
                                           FactorCase{"Two", 2.0}, FactorCase{"Four", 4.0}),
                         label_of<FactorCase>);

/** @brief Tests that the loss of an overconfident set is the value worked out by hand. */
TEST(Small_LanguageTrainingTemperatureScaling, TheLossOfAnOverconfidentSetIsTheHandValue)
{
    // At a factor of 2 the margin is 2 ln 3, so a row is right with probability 9/10.
    const LogitSet set = overconfident_groups(2.0, K_GROUPS);
    const double expected = (3.0 * std::log(10.0 / 9.0) + std::log(10.0)) / 4.0;

    // A row's loss moves by at most its margin's change; see the sidecar for the bound.
    const double bound =
        2.0 * three_to_one_margin() * TestSupport::unit_roundoff<float>()
        + static_cast<double>(K_ROWS + K_EXTRA_OPERATIONS) * TestSupport::unit_roundoff<double>()
              * expected;

    SA_ASSERT_OK_AND_ASSIGN(
        const double loss,
        mean_negative_log_likelihood(set.logits_span(), set.labels_span(), Support::K_CLASSES,
                                     1.0));

    EXPECT_TRUE(TestSupport::is_within(
        TestSupport::measure_error(loss, expected, bound, K_WHOLE_BOUND)));
}

/** @brief Tests that dividing the logits by a temperature is the same as scaling them down. */
TEST(Small_LanguageTrainingTemperatureScaling, ATemperatureDividesTheLogits)
{
    // Doubling a float is exact, so factor 2 over temperature 2 is factor 1 bit for bit.
    const LogitSet doubled = overconfident_groups(2.0, K_GROUPS);
    const LogitSet plain = overconfident_groups(1.0, K_GROUPS);

    SA_ASSERT_OK_AND_ASSIGN(
        const double divided,
        mean_negative_log_likelihood(doubled.logits_span(), doubled.labels_span(),
                                     Support::K_CLASSES, 2.0));
    SA_ASSERT_OK_AND_ASSIGN(
        const double reference,
        mean_negative_log_likelihood(plain.logits_span(), plain.labels_span(),
                                     Support::K_CLASSES, 1.0));

    EXPECT_EQ(divided, reference);
}

/** @brief Tests that scaling by the fitted temperature lowers the loss of an overconfident set. */
TEST(Small_LanguageTrainingTemperatureScaling, TheFittedTemperatureLowersTheLoss)
{
    const LogitSet set = overconfident_groups(2.0, K_GROUPS);

    SA_ASSERT_OK_AND_ASSIGN(
        const double fitted,
        fit_temperature(set.logits_span(), set.labels_span(), Support::K_CLASSES));
    SA_ASSERT_OK_AND_ASSIGN(
        const double before,
        mean_negative_log_likelihood(set.logits_span(), set.labels_span(), Support::K_CLASSES,
                                     1.0));
    SA_ASSERT_OK_AND_ASSIGN(
        const double after,
        mean_negative_log_likelihood(set.logits_span(), set.labels_span(), Support::K_CLASSES,
                                     fitted));

    EXPECT_LT(after, before);
}

/** @brief Tests that the same set gives the same temperature, bit for bit. */
TEST(Small_LanguageTrainingTemperatureScaling, TheSameSetGivesTheSameTemperature)
{
    const LogitSet set = overconfident_groups(3.0, K_GROUPS);

    SA_ASSERT_OK_AND_ASSIGN(
        const double first,
        fit_temperature(set.logits_span(), set.labels_span(), Support::K_CLASSES));
    SA_ASSERT_OK_AND_ASSIGN(
        const double second,
        fit_temperature(set.logits_span(), set.labels_span(), Support::K_CLASSES));

    EXPECT_EQ(first, second);
}

/** @brief Tests that a set the logits all predict has no finite minimum and fits the lowest T. */
TEST(Small_LanguageTrainingTemperatureScaling, ASetEveryRowOfWhichIsRightFitsTheLowestTemperature)
{
    const std::vector<float> logits = {5.0f, 0.0f, 5.0f, 0.0f};
    const std::vector<std::int32_t> labels = {0, 0};

    SA_ASSERT_OK_AND_ASSIGN(
        const double fitted,
        fit_temperature(SushiRuntime::span<const float>(logits.data(), logits.size()),
                        SushiRuntime::span<const std::int32_t>(labels.data(), labels.size()),
                        Support::K_CLASSES));

    // The search stops with the bracket K_TEMPERATURE_LOG_TOLERANCE wide at the lower end.
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
        std::log(fitted), std::log(K_MIN_TEMPERATURE), K_TEMPERATURE_LOG_TOLERANCE,
        K_WHOLE_BOUND)));
}

/** @brief Tests that a set the logits all miss fits the highest temperature. */
TEST(Small_LanguageTrainingTemperatureScaling, ASetEveryRowOfWhichIsWrongFitsTheHighestTemperature)
{
    const std::vector<float> logits = {5.0f, 0.0f, 5.0f, 0.0f};
    const std::vector<std::int32_t> labels = {1, 1};

    SA_ASSERT_OK_AND_ASSIGN(
        const double fitted,
        fit_temperature(SushiRuntime::span<const float>(logits.data(), logits.size()),
                        SushiRuntime::span<const std::int32_t>(labels.data(), labels.size()),
                        Support::K_CLASSES));

    // The search stops with the bracket K_TEMPERATURE_LOG_TOLERANCE wide at the upper end.
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
        std::log(fitted), std::log(K_MAX_TEMPERATURE), K_TEMPERATURE_LOG_TOLERANCE,
        K_WHOLE_BOUND)));
}

/** @brief Tests that a set that is no rows of classes is refused naming what is wrong. */
TEST_P(Small_LanguageTrainingTemperatureScalingSet, TheFitRefusesASetThatDoesNotFit)
{
    const SetCase& row = GetParam();

    SA_EXPECT_FAILURE(
        fit_temperature(SushiRuntime::span<const float>(row.logits.data(), row.logits.size()),
                        SushiRuntime::span<const std::int32_t>(row.labels.data(),
                                                               row.labels.size()),
                        row.classes),
        StatusCode::InvalidDescription, "fit_temperature", row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageTrainingTemperatureScalingSet,
    ::testing::Values(
        SetCase{"NoRow", {}, {}, 2, "holds no row", ""},
        SetCase{"OneClass", {1.0f, 2.0f}, {0, 0}, 1, "1 classes", "at least 2"},
        SetCase{"ALogitCountThatIsNoRows", {1.0f, 2.0f, 3.0f}, {0, 1}, 2, "3 logits",
                "2 rows of 2 classes"},
        SetCase{"ALabelPastTheClasses", {1.0f, 2.0f, 3.0f, 4.0f}, {0, 2}, 2, "row 1",
                "label 2"},
        SetCase{"ANegativeLabel", {1.0f, 2.0f}, {-1}, 2, "row 0", "label -1"},
        SetCase{"ANaNLogit", {1.0f, std::numeric_limits<float>::quiet_NaN()}, {0}, 2,
                "element 1", "not finite"},
        SetCase{"AnInfiniteLogit", {std::numeric_limits<float>::infinity(), 0.0f}, {0}, 2,
                "element 0", "not finite"}),
    label_of<SetCase>);

/** @brief Tests that the loss refuses a temperature that is not positive and finite. */
TEST_P(Small_LanguageTrainingTemperatureScalingTemperature, TheLossRefusesATemperatureOutOfRange)
{
    const LogitSet set = overconfident_groups(1.0, 1);

    SA_EXPECT_FAILURE(mean_negative_log_likelihood(set.logits_span(), set.labels_span(),
                                                   Support::K_CLASSES, GetParam()),
                      StatusCode::InvalidDescription, "mean_negative_log_likelihood",
                      "finite and positive");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LanguageTrainingTemperatureScalingTemperature,
    ::testing::Values(0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::infinity()));
