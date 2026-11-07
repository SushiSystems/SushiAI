/**************************************************************************/
/* test_incremental.cpp                                                   */
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
 * @file test_incremental.cpp
 * @brief Tests the leaf names of the incremental blocks.
 * @author Mustafa Garip
 */

#include <set>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/language/blocks/incremental.hpp>

using namespace SushiAI::Language::Blocks;

/** @brief Tests that the five leaves an incremental run is fed carry five different names. */
TEST(Small_BlockIncremental, EveryLeafHasItsOwnName)
{
    const std::set<std::string_view> names{K_POSITION_LEAF, K_ATTENTION_MASK_LEAF,
                                           K_CACHE_SLOT_LEAF, K_ROTARY_COSINE_LEAF,
                                           K_ROTARY_SINE_LEAF};

    EXPECT_EQ(names.size(), 5u);
    EXPECT_EQ(names.count(""), 0u);
}
