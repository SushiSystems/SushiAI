/**************************************************************************/
/* test_in_memory_epoch_aware.cpp                                         */
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
 * @file test_in_memory_epoch_aware.cpp
 * @brief Tests that InMemoryEpochAware keeps the promise of the EpochAware interface.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/data/in_memory_epoch_aware.hpp>

using namespace SushiAI;

/** @brief Tests that a fresh adapter has begun no epoch. */
TEST(Small_InMemoryEpochAware, StartsWithNoEpochBegun)
{
    const Data::InMemoryEpochAware adapter;

    EXPECT_EQ(adapter.epoch(), -1);
    EXPECT_EQ(adapter.begin_count(), 0u);
}

/** @brief Tests that begin_epoch through the interface holds the epoch and counts the call. */
TEST(Small_InMemoryEpochAware, BeginEpochThroughTheInterfaceHoldsTheEpoch)
{
    const Data::InMemoryEpochAware adapter;
    const Data::EpochAware& port = adapter;

    port.begin_epoch(3);

    EXPECT_EQ(adapter.epoch(), 3);
    EXPECT_EQ(adapter.begin_count(), 1u);
}

/** @brief Tests that the latest epoch wins and every call is counted, a repeat included. */
TEST(Small_InMemoryEpochAware, TheLatestEpochWinsAndEveryCallIsCounted)
{
    const Data::InMemoryEpochAware adapter;

    adapter.begin_epoch(0);
    adapter.begin_epoch(1);
    adapter.begin_epoch(1);
    adapter.begin_epoch(0);

    EXPECT_EQ(adapter.epoch(), 0);
    EXPECT_EQ(adapter.begin_count(), 4u);
}

/** @brief Tests that a negative epoch stops the process, as the interface states. */
TEST(Medium_InMemoryEpochAwareFatal, ANegativeEpochStopsTheProcess)
{
    const Data::InMemoryEpochAware adapter;

    SA_EXPECT_FATAL(adapter.begin_epoch(-2), "InMemoryEpochAware::begin_epoch", "-2");
}
