/**************************************************************************/
/* test_head_training_options.cpp                                         */
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
 * @file test_head_training_options.cpp
 * @brief Tests the training options of an anchor head.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <limits>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>

namespace
{
    using SushiAI::StatusCode;

    using SushiAI::Vision::Detect::BoxGradient;

    using SushiAI::Vision::Detect::Anchor::HeadTrainingOptions;

    using SushiAI::Vision::Detect::Anchor::validate_head_training;

    /** @brief Holds a value no enumeration of the training options names. */
    constexpr std::uint8_t K_UNKNOWN_ENUMERATOR = 200;

    /** @brief Holds training options outside their range and the name of their row. */
    struct RefusedTraining
    {
        std::string name;              /**< The row's name. */
        HeadTrainingOptions options;   /**< The options outside their range. */
    };

    /** @brief Runs one set of training options the check refuses. */
    class Small_DetectAnchorDeltaTraining : public ::testing::TestWithParam<RefusedTraining>
    {

    };

    /** @brief Names a row by its name field. */
    std::string training_name(const ::testing::TestParamInfo<RefusedTraining>& info)
    {
        return info.param.name;
    }

    /** @brief Returns the default training options after @p edit has changed them. */
    HeadTrainingOptions make_edited(void (*edit)(HeadTrainingOptions&))
    {
        HeadTrainingOptions options;
        edit(options);
        return options;
    }

} // namespace

/** @brief Tests that the default training options pass their check. */
TEST(Small_DetectAnchorDelta, DefaultTrainingOptionsValidate)
{
    EXPECT_TRUE(validate_head_training(HeadTrainingOptions{}).is_ok());
}

/** @brief Tests that an option outside its range is an invalid description. */
TEST_P(Small_DetectAnchorDeltaTraining, AnOptionOutsideItsRangeIsRefused)
{
    EXPECT_EQ(validate_head_training(GetParam().options).code(), StatusCode::InvalidDescription);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DetectAnchorDeltaTraining,
    ::testing::Values(
        RefusedTraining{"AMatchThresholdOfZero", make_edited([](HeadTrainingOptions& o)
                                                  { o.thresholds.match_threshold = 0.0f; })},
        RefusedTraining{"AMatchThresholdAboveOne", make_edited([](HeadTrainingOptions& o)
                                                     { o.thresholds.match_threshold = 1.5f; })},
        RefusedTraining{"ANegativeIgnoreThreshold", make_edited([](HeadTrainingOptions& o)
                                                      { o.thresholds.ignore_threshold = -0.1f; })},
        RefusedTraining{"AClassGateAboveOne",
                        make_edited([](HeadTrainingOptions& o) { o.class_gate = 1.5f; })},
        RefusedTraining{"ANegativeBoxWeight",
                        make_edited([](HeadTrainingOptions& o) { o.box_weight = -1.0f; })},
        RefusedTraining{"AnObjectWeightThatIsNotANumber",
                        make_edited([](HeadTrainingOptions& o)
                                    { o.object_weight = std::numeric_limits<float>::quiet_NaN(); })},
        RefusedTraining{"AnInfiniteBoxWeight",
                        make_edited([](HeadTrainingOptions& o)
                                    { o.box_weight = std::numeric_limits<float>::infinity(); })},
        RefusedTraining{"ANegativeMaxDelta",
                        make_edited([](HeadTrainingOptions& o) { o.max_delta = -2.0f; })},
        RefusedTraining{"AnUnknownBoxGradient",
                        make_edited([](HeadTrainingOptions& o)
                                    {
                                        o.box_gradient =
                                            static_cast<BoxGradient>(K_UNKNOWN_ENUMERATOR);
                                    })},
        RefusedTraining{"ANegativeBoxGain",
                        make_edited([](HeadTrainingOptions& o) { o.box_gain = -0.5f; })},
        RefusedTraining{"AClassGainThatIsNotANumber",
                        make_edited([](HeadTrainingOptions& o)
                                    { o.class_gain = std::numeric_limits<float>::quiet_NaN(); })},
        RefusedTraining{"AnInfiniteObjectnessGain",
                        make_edited([](HeadTrainingOptions& o)
                                    {
                                        o.objectness_gain =
                                            std::numeric_limits<float>::infinity();
                                    })}),
    training_name);
