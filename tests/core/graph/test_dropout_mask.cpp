/**************************************************************************/
/* test_dropout_mask.cpp                                                  */
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
 * @file test_dropout_mask.cpp
 * @brief Tests the SplitMix64 sequence and the site seeds of the dropout generator.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/graph/dropout_mask.hpp>
#include <support/splitmix64.hpp>

using namespace SushiAI;
using TestSupport::reference_splitmix64;

/** @brief Tests the first four numbers of seed 0 against the published SplitMix64 sequence. */
TEST(Small_DropoutMask, TheSequenceOfSeedZeroIsThePublishedSplitMix64)
{
    EXPECT_EQ(Graph::splitmix64_at(0, 0), 0xE220A8397B1DCDAFull);
    EXPECT_EQ(Graph::splitmix64_at(0, 1), 0x6E789E6AA1B965F4ull);
    EXPECT_EQ(Graph::splitmix64_at(0, 2), 0x06C45D188009454Full);
    EXPECT_EQ(Graph::splitmix64_at(0, 3), 0xF88BB8A8724C81ECull);
}

/** @brief Tests that the seed of site s is number s of the trace seed's sequence. */
TEST(Small_DropoutMask, EachSiteOfATraceHasItsOwnSeed)
{
    EXPECT_EQ(Graph::dropout_site_seed(0, 0), 0xE220A8397B1DCDAFull);
    EXPECT_EQ(Graph::dropout_site_seed(0, 1), 0x6E789E6AA1B965F4ull);
    EXPECT_EQ(Graph::dropout_site_seed(7, 0), reference_splitmix64(7, 0));
    EXPECT_EQ(Graph::dropout_site_seed(7, 1), reference_splitmix64(7, 1));
}
