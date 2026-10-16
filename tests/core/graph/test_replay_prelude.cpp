/**************************************************************************/
/* test_replay_prelude.cpp                                                */
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
 * @file test_replay_prelude.cpp
 * @brief Tests that a replay prelude holds host work called with the count of finished replays.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/graph/replay_prelude.hpp>

using namespace SushiAI;
using Graph::ReplayPrelude;

/** @brief Tests that a prelude built empty holds no target, which a context refuses. */
TEST(Small_ReplayPrelude, ADefaultPreludeHoldsNoTarget)
{
    const ReplayPrelude prelude;

    EXPECT_FALSE(static_cast<bool>(prelude));
}

/** @brief Tests that a prelude receives the replay count it is called with, 0 first. */
TEST(Small_ReplayPrelude, APreludeReceivesTheCountOfFinishedReplays)
{
    std::vector<std::uint64_t> seen;
    const ReplayPrelude prelude = [&seen](std::uint64_t run)
    {
        seen.push_back(run);
    };

    for (std::uint64_t run = 0; run < 3; ++run)
        prelude(run);

    EXPECT_EQ(seen, (std::vector<std::uint64_t>{0, 1, 2}));
}
