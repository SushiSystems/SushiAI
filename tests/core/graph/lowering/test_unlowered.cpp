/**************************************************************************/
/* test_unlowered.cpp                                                     */
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
 * @file test_unlowered.cpp
 * @brief Tests the list of operations that are recorded beside a graph and have no row.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/graph/lowering/unlowered.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Names two positions of K_UNLOWERED. */
    struct PositionPair
    {
        std::size_t first;  /**< The earlier position. */
        std::size_t second; /**< The later position. */
    };

    /** @brief Lists every pair of positions of the four entries. */
    constexpr PositionPair K_PAIRS[] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

    /** @brief Runs once per pair of positions of K_UNLOWERED. */
    class Small_UnloweredPair : public ::testing::TestWithParam<PositionPair>
    {

    };
} // namespace

/** @brief Tests that the list names the two optimizer updates, the clip and the stats update. */
TEST(Small_Unlowered, TheListNamesTheFourOperationsRecordedBesideTheGraph)
{
    const auto holds = [](OpID op)
    {
        return std::find(Graph::K_UNLOWERED.begin(), Graph::K_UNLOWERED.end(), op)
            != Graph::K_UNLOWERED.end();
    };

    EXPECT_EQ(Graph::K_UNLOWERED.size(), 4u);
    EXPECT_TRUE(holds(Ops::K_ADAMW_UPDATE.id));
    EXPECT_TRUE(holds(Ops::K_SGD_UPDATE.id));
    EXPECT_TRUE(holds(Ops::K_CLIP_SCALE.id));
    EXPECT_TRUE(holds(Ops::K_BATCHNORM_UPDATE_STATS.id));
}

/** @brief Tests that two positions of the list never hold the same operation. */
TEST_P(Small_UnloweredPair, NoOperationIsListedTwice)
{
    const PositionPair pair = GetParam();

    EXPECT_NE(Graph::K_UNLOWERED[pair.first], Graph::K_UNLOWERED[pair.second]);
}

INSTANTIATE_TEST_SUITE_P(, Small_UnloweredPair, ::testing::ValuesIn(K_PAIRS),
                         [](const ::testing::TestParamInfo<PositionPair>& info)
                         {
                             return std::to_string(info.param.first) + "_"
                                  + std::to_string(info.param.second);
                         });
