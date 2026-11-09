/**************************************************************************/
/* test_assigner_config.cpp                                               */
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
 * @file test_assigner_config.cpp
 * @brief Tests that both assigner presets carry the class count and refuse a count below one.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/free/assigner_config.hpp>

namespace
{
    using SushiAI::Vision::Detect::Free::one_to_many_assigner_config;
    using SushiAI::Vision::Detect::Free::one_to_one_assigner_config;

    /** @brief Holds the strides of a three-level model. */
    const std::vector<float> K_STRIDES{8.0f, 16.0f, 32.0f};
} // namespace

/** @brief Tests that both presets hold the class count they are given. */
TEST(Small_DetectAssignerConfig, BothPresetsCarryTheClassCount)
{
    EXPECT_EQ(one_to_many_assigner_config(3, K_STRIDES).num_classes, 3);
    EXPECT_EQ(one_to_one_assigner_config(3, K_STRIDES).num_classes, 3);
}

/** @brief Tests that both presets take a single stride as stride_val. */
TEST(Small_DetectAssignerConfig, BothPresetsTakeASingleStrideAsStrideVal)
{
    const std::vector<float> one{4.0f};

    EXPECT_EQ(one_to_many_assigner_config(3, one).stride_val, 4.0f);
    EXPECT_EQ(one_to_one_assigner_config(3, one).stride_val, 4.0f);
}

/** @brief Tests that both presets refuse a class count of zero or below. */
TEST(Medium_DetectAssignerConfig, BothPresetsRefuseAClassCountBelowOne)
{
    SA_EXPECT_FATAL(static_cast<void>(one_to_many_assigner_config(0, K_STRIDES)),
                    "positive class count", "got 0");
    SA_EXPECT_FATAL(static_cast<void>(one_to_one_assigner_config(-2, K_STRIDES)),
                    "positive class count", "got -2");
}
