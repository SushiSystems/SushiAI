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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/


/**
 * @file test_head_training_options.cpp
 * @brief Tests the training options of an anchor head.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/status.hpp>
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

} // namespace

/** @brief Tests that the default training options pass their check. */
TEST(Unit_DetectAnchorDelta, DefaultTrainingOptionsValidate)
{
    EXPECT_TRUE(validate_head_training(HeadTrainingOptions{}).is_ok());
}

/** @brief Tests that each option outside its range is an invalid description. */
TEST(Unit_DetectAnchorDelta, TrainingOptionsOutsideTheirRangeAreRefused)
{
    const float not_a_number = std::numeric_limits<float>::quiet_NaN();
    std::vector<HeadTrainingOptions> refused(8);
    refused[0].thresholds.match_threshold = 0.0f;
    refused[1].thresholds.match_threshold = 1.5f;
    refused[2].thresholds.ignore_threshold = -0.1f;
    refused[3].class_gate = 1.5f;
    refused[4].box_weight = -1.0f;
    refused[5].object_weight = not_a_number;
    refused[6].box_weight = std::numeric_limits<float>::infinity();
    refused[7].max_delta = -2.0f;
    refused.emplace_back();
    refused.back().box_gradient = static_cast<BoxGradient>(K_UNKNOWN_ENUMERATOR);

    for (std::size_t index = 0; index < refused.size(); ++index)
    {
        EXPECT_EQ(validate_head_training(refused[index]).code(),
                  StatusCode::InvalidDescription)
            << "options " << index;
    }
}
