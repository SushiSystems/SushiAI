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
 * @file test_state_id.cpp
 * @brief Tests the process-wide state identity counter and the identity a holder draws.
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

/** @brief Tests that no drawn identity is the one meaning "none". */
TEST(Small_StateID, NoDrawnIdentityIsNoState)
{
    Graph::StateIDSource ids;

    EXPECT_NE(ids.draw(), K_NO_STATE);
}

/** @brief Tests that each draw returns an identity above every earlier one. */
TEST(Small_StateID, EachDrawIsAboveTheOneBefore)
{
    Graph::StateIDSource ids;
    const StateID first = ids.draw();
    const StateID second = ids.draw();
    const StateID third = ids.draw();

    EXPECT_LT(static_cast<std::uint64_t>(first), static_cast<std::uint64_t>(second));
    EXPECT_LT(static_cast<std::uint64_t>(second), static_cast<std::uint64_t>(third));
}

/** @brief Tests that assigning a holder to itself by copy keeps its identity. */
TEST(Small_StateID, CopySelfAssignmentKeepsTheIdentity)
{
    Graph::StateIDSource ids;
    StateIdentity holder;
    const StateID drawn = holder.get(ids);

    StateIdentity& alias = holder;
    holder = alias;

    EXPECT_EQ(holder.peek(), drawn);
}

/** @brief Tests that assigning a holder to itself by move keeps its identity. */
TEST(Small_StateID, MoveSelfAssignmentKeepsTheIdentity)
{
    Graph::StateIDSource ids;
    StateIdentity holder;
    const StateID drawn = holder.get(ids);

    StateIdentity& alias = holder;
    holder = std::move(alias);

    EXPECT_EQ(holder.peek(), drawn);
}

/** @brief Tests that get() draws an identity once and returns it on every later call. */
TEST(Small_StateID, GetDrawsOneStableIdentity)
{
    Graph::StateIDSource ids;
    StateIdentity holder;

    EXPECT_EQ(holder.peek(), K_NO_STATE);

    const StateID drawn = holder.get(ids);
    EXPECT_NE(drawn, K_NO_STATE);
    EXPECT_EQ(holder.get(ids), drawn);
    EXPECT_EQ(holder.peek(), drawn);
}

/** @brief Tests that two holders never share an identity. */
TEST(Small_StateID, TwoHoldersNeverShareAnIdentity)
{
    Graph::StateIDSource ids;
    StateIdentity first;
    StateIdentity second;

    EXPECT_NE(second.get(ids), first.get(ids));
}

/** @brief Tests that a copy-constructed holder is new state and draws its own identity. */
TEST(Small_StateID, ACopyConstructedHolderDoesNotShareTheSourcesIdentity)
{
    Graph::StateIDSource ids;
    StateIdentity source;
    const StateID drawn = source.get(ids);

    StateIdentity copied(source);

    EXPECT_EQ(copied.peek(), K_NO_STATE);
    EXPECT_NE(copied.get(ids), drawn);
    EXPECT_EQ(source.peek(), drawn);
}

/** @brief Tests that a copy-assigned holder drops its identity and never takes the source's. */
TEST(Small_StateID, ACopyAssignedHolderDoesNotShareTheSourcesIdentity)
{
    Graph::StateIDSource ids;
    StateIdentity source;
    const StateID drawn = source.get(ids);
    StateIdentity assigned;
    const StateID replaced = assigned.get(ids);

    assigned = source;

    EXPECT_EQ(assigned.peek(), K_NO_STATE);
    EXPECT_NE(assigned.get(ids), drawn);
    EXPECT_NE(assigned.get(ids), replaced);
    EXPECT_EQ(source.peek(), drawn);
}

/** @brief Tests that move construction carries the identity to the target and empties the source. */
TEST(Small_StateID, AMoveConstructedHolderCarriesTheIdentityAway)
{
    Graph::StateIDSource ids;
    StateIdentity source;
    const StateID drawn = source.get(ids);

    StateIdentity moved(std::move(source));

    EXPECT_EQ(moved.peek(), drawn);
    EXPECT_EQ(source.peek(), K_NO_STATE);
}

/** @brief Tests that move assignment carries the identity to the target and empties the source. */
TEST(Small_StateID, AMoveAssignedHolderCarriesTheIdentityAway)
{
    Graph::StateIDSource ids;
    StateIdentity source;
    const StateID drawn = source.get(ids);
    StateIdentity assigned;
    (void)assigned.get(ids);

    assigned = std::move(source);

    EXPECT_EQ(assigned.peek(), drawn);
    EXPECT_EQ(source.peek(), K_NO_STATE);
}
