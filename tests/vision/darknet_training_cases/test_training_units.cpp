/**************************************************************************/
/* test_training_units.cpp                                                */
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
 * @file test_training_units.cpp
 * @brief Tests the float unit roundoff helper the Darknet training suites state their bounds in.
 * @author Mustafa Garip
 */

#include <cmath>
#include <gtest/gtest.h>
#include <SushiAI/vision/darknet_training_cases/training_units.hpp>

namespace SaTest = SushiAI::Test;

/** @brief Tests that one unit is 2^-24, the roundoff of a float, and that units scale linearly. */
TEST(Small_TrainingUnits, OneUnitIsTwoToTheMinus24)
{
    EXPECT_EQ(SaTest::units(1.0), std::ldexp(1.0, -24));
    EXPECT_EQ(SaTest::units(0.0), 0.0);
    EXPECT_EQ(SaTest::units(8388608.0), 0.5);
}
