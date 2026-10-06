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
 * @file test_version.cpp
 * @brief Tests that the runtime version agrees with the compile-time macros.
 * @author Mustafa Garip
 */

#include <cstring>
#include <gtest/gtest.h>
#include <SushiAI/version.hpp>

TEST(Unit_Version, MatchesEncodedMacro)
{
    EXPECT_EQ(SUSHIAI_VERSION, SUSHIAI_VERSION_ENCODE(SUSHIAI_VERSION_MAJOR,
                                                       SUSHIAI_VERSION_MINOR,
                                                       SUSHIAI_VERSION_PATCH));
}

TEST(Unit_Version, RuntimeStringMatchesCompileTimeMacro)
{
    EXPECT_STREQ(SushiAI::version_string(), SUSHIAI_VERSION_STRING);
}
