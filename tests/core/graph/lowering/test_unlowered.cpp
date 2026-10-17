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
 * @file test_unlowered.cpp
 * @brief Tests the list of operations that are recorded beside a graph and have no row.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/graph/lowering/unlowered.hpp>

using namespace SushiAI;

/** @brief Tests that the list names the two optimizer updates, the clip and the stats update. */
TEST(Small_Unlowered, TheListNamesTheFourOperationsRecordedBesideTheGraph)
{
    const auto holds = [](OpID op)
    {
        return std::find(Graph::K_UNLOWERED.begin(), Graph::K_UNLOWERED.end(), op)
            != Graph::K_UNLOWERED.end();
    };

    EXPECT_TRUE(holds(Ops::K_ADAMW_UPDATE.id));
    EXPECT_TRUE(holds(Ops::K_SGD_UPDATE.id));
    EXPECT_TRUE(holds(Ops::K_CLIP_SCALE.id));
    EXPECT_TRUE(holds(Ops::K_BATCHNORM_UPDATE_STATS.id));
}

/** @brief Tests that no operation appears twice, so the four entries are four operations. */
TEST(Small_Unlowered, NoOperationIsListedTwice)
{
    for (std::size_t i = 0; i < Graph::K_UNLOWERED.size(); ++i)
    {
        for (std::size_t j = i + 1; j < Graph::K_UNLOWERED.size(); ++j)
            EXPECT_NE(Graph::K_UNLOWERED[i], Graph::K_UNLOWERED[j]) << i << " and " << j;
    }
}
