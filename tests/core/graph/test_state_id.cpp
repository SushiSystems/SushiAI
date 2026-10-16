/**************************************************************************/
/* test_state_id.cpp                                                      */
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
 * @file test_state_id.cpp
 * @brief Tests the process-wide state identity counter and self-assignment of a holder.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/graph/state_id.hpp>

using namespace SushiAI;
using Graph::K_NO_STATE;
using Graph::StateID;
using Graph::StateIdentity;

/** @brief Tests that the identity meaning "none" is zero, which the counter never hands out. */
TEST(Small_StateID, NoStateIsZeroAndIsNeverDrawn)
{
    EXPECT_EQ(static_cast<std::uint64_t>(K_NO_STATE), 0u);
    EXPECT_NE(Graph::Detail::draw_state_id(), K_NO_STATE);
}

/** @brief Tests that each draw returns an identity above every earlier one. */
TEST(Small_StateID, EachDrawIsAboveTheOneBefore)
{
    const StateID first = Graph::Detail::draw_state_id();
    const StateID second = Graph::Detail::draw_state_id();
    const StateID third = Graph::Detail::draw_state_id();

    EXPECT_LT(static_cast<std::uint64_t>(first), static_cast<std::uint64_t>(second));
    EXPECT_LT(static_cast<std::uint64_t>(second), static_cast<std::uint64_t>(third));
}

/** @brief Tests that assigning a holder to itself, by copy or by move, keeps its identity. */
TEST(Small_StateID, SelfAssignmentKeepsTheIdentity)
{
    StateIdentity holder;
    const StateID drawn = holder.get();

    StateIdentity& alias = holder;
    holder = alias;
    EXPECT_EQ(holder.peek(), drawn);

    holder = std::move(alias);
    EXPECT_EQ(holder.peek(), drawn);
}
