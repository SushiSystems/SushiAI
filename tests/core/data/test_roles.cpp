/**************************************************************************/
/* test_roles.cpp                                                         */
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
 * @file test_roles.cpp
 * @brief Tests the spelling of the target roles more than one module reads.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/data/roles.hpp>

using namespace SushiAI;

/** @brief Tests that each role keeps the spelling the datasets and the objectives agree on. */
TEST(Small_DataRoles, EachRoleHasItsSpelling)
{
    EXPECT_EQ(Data::K_ROLE_LABELS, "labels");
    EXPECT_EQ(Data::K_ROLE_BOXES, "boxes");
    EXPECT_EQ(Data::K_ROLE_CLASSES, "classes");
    EXPECT_EQ(Data::K_ROLE_VALID, "valid");
    EXPECT_EQ(Data::K_ROLE_MASK, "mask");
    EXPECT_EQ(Data::K_ROLE_VALUES, "values");
}
