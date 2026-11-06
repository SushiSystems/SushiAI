/**************************************************************************/
/* test_version.cpp                                                       */
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
 * @file test_version.cpp
 * @brief Tests that the runtime version agrees with the compile-time macros.
 * @author Mustafa Garip
 */

#include <cstring>
#include <gtest/gtest.h>
#include <SushiAI/version.hpp>

TEST(Small_Version, MatchesEncodedMacro)
{
    EXPECT_EQ(SUSHIAI_VERSION, SUSHIAI_VERSION_ENCODE(SUSHIAI_VERSION_MAJOR,
                                                       SUSHIAI_VERSION_MINOR,
                                                       SUSHIAI_VERSION_PATCH));
}

TEST(Small_Version, RuntimeStringMatchesCompileTimeMacro)
{
    EXPECT_STREQ(SushiAI::version_string(), SUSHIAI_VERSION_STRING);
}
