/**************************************************************************/
/* test_yolox_head_contract.cpp                                           */
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
 * @file test_yolox_head_contract.cpp
 * @brief Tests the port names and the objectness prior of the YOLOX head contract.
 * @author Mustafa Garip
 */

#include <cmath>

#include <gtest/gtest.h>

#include <SushiAI/vision/detect/yolox/yolox_head_contract.hpp>
#include <support/tolerance.hpp>

namespace
{
    using SushiAI::Vision::Detect::Yolox::K_YOLOX_BOXES;
    using SushiAI::Vision::Detect::Yolox::K_YOLOX_CLASSES;
    using SushiAI::Vision::Detect::Yolox::K_YOLOX_OBJECTNESS;
    using SushiAI::Vision::Detect::Yolox::yolox_port_name;
    using SushiAI::Vision::Detect::Yolox::yolox_prior_bias;
} // namespace

/** @brief Tests that a port name is "level<level>.<role>". */
TEST(Small_DetectYoloxHeadContract, APortNameIsTheLevelAndTheRole)
{
    EXPECT_EQ(yolox_port_name(0, K_YOLOX_CLASSES), "level0.classes");
    EXPECT_EQ(yolox_port_name(2, K_YOLOX_BOXES), "level2.boxes");
    EXPECT_EQ(yolox_port_name(11, K_YOLOX_OBJECTNESS), "level11.objectness");
}

/**
 * @brief Tests that the prior bias is the logit of 0.01, -log(99), as the reference writes it
 *        -log((1 - 0.01) / 0.01). The bound is two double roundings: the quotient's and the
 *        logarithm's.
 */
TEST(Small_DetectYoloxHeadContract, ThePriorBiasIsTheLogitOfOneHundredth)
{
    const double reference = -std::log((1.0 - 0.01) / 0.01);

    EXPECT_NEAR(yolox_prior_bias(), reference,
                (2.0 + std::fabs(reference)) * TestSupport::unit_roundoff<double>());
    EXPECT_LT(yolox_prior_bias(), 0.0);
}
