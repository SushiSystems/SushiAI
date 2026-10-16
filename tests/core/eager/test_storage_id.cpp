/**************************************************************************/
/* test_storage_id.cpp                                                    */
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
 * @file test_storage_id.cpp
 * @brief Tests that StorageIDSource hands out each id once and never the empty id.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <unordered_set>
#include <gtest/gtest.h>
#include <SushiAI/eager/storage_id.hpp>

using SushiAI::Eager::K_NO_STORAGE;
using SushiAI::Eager::StorageID;
using SushiAI::Eager::StorageIDSource;

namespace
{
    /** @brief Holds how many ids a case draws from one source. */
    constexpr std::size_t K_DRAWS = 1000;
} // namespace

/** @brief Tests that the empty id is zero, the value a zero-filled table holds. */
TEST(Small_StorageID, TheEmptyIDIsZero)
{
    EXPECT_EQ(K_NO_STORAGE, StorageID{0});
}

/** @brief Tests that a fresh source never returns the empty id. */
TEST(Small_StorageID, AFreshSourceNeverReturnsTheEmptyID)
{
    StorageIDSource source;

    EXPECT_NE(source.next(), K_NO_STORAGE);
}

/** @brief Tests that a thousand draws from one source are all different. */
TEST(Small_StorageID, EveryDrawFromOneSourceIsNew)
{
    StorageIDSource source;
    std::unordered_set<StorageID> seen;

    for (std::size_t i = 0; i < K_DRAWS; ++i)
    {
        const StorageID id = source.next();
        EXPECT_NE(id, K_NO_STORAGE);
        EXPECT_TRUE(seen.insert(id).second) << "id " << id << " was returned twice";
    }
}

/** @brief Tests that two sources count on their own, so ids are unique per source only. */
TEST(Small_StorageID, TwoSourcesCountIndependently)
{
    StorageIDSource first;
    StorageIDSource second;

    const StorageID drawn = first.next();
    (void)first.next();

    EXPECT_EQ(second.next(), drawn);
}
