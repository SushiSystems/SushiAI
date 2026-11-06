/**************************************************************************/
/* test_in_memory_replica_serve_observer.cpp                              */
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
 * @file test_in_memory_replica_serve_observer.cpp
 * @brief Tests that InMemoryReplicaServeObserver keeps the promise of ReplicaServeObserver.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/distribution/in_memory_replica_serve_observer.hpp>
#include <SushiAI/distribution/replica_position.hpp>

using namespace SushiAI::Distribution;

namespace
{
    /** @brief Builds the position at @p step of an epoch of @p batches batches. */
    ReplicaPosition position_at(std::uint64_t step, std::uint64_t batches)
    {
        return ReplicaPosition::at_step(step, batches);
    }
} // namespace

/** @brief Tests that a fresh observer has heard of no task. */
TEST(Small_InMemoryReplicaServeObserver, StartsEmpty)
{
    const InMemoryReplicaServeObserver observer;

    EXPECT_EQ(observer.served_count(), 0u);
    EXPECT_TRUE(observer.served().empty());
}

/** @brief Tests that tasks told through the interface are kept in order with their position. */
TEST(Small_InMemoryReplicaServeObserver, KeepsEachTaskInOrderThroughTheInterface)
{
    InMemoryReplicaServeObserver observer;
    ReplicaServeObserver& port = observer;

    port.on_served(2, position_at(0, 4));
    port.on_served(0, position_at(5, 4));

    ASSERT_EQ(observer.served_count(), 2u);
    EXPECT_EQ(observer.served()[0].replica, 2u);
    EXPECT_EQ(observer.served()[0].position.step, 0u);
    EXPECT_EQ(observer.served()[1].replica, 0u);
    EXPECT_EQ(observer.served()[1].position.step, 5u);
    EXPECT_EQ(observer.served()[1].position.epoch, 1u);
    EXPECT_EQ(observer.served()[1].position.batch_index, 1u);
}

/** @brief Tests that clear forgets every task, and the observer hears again afterwards. */
TEST(Small_InMemoryReplicaServeObserver, ClearForgetsEveryTask)
{
    InMemoryReplicaServeObserver observer;
    observer.on_served(1, position_at(3, 4));

    observer.clear();
    EXPECT_EQ(observer.served_count(), 0u);

    observer.on_served(1, position_at(4, 4));
    EXPECT_EQ(observer.served_count(), 1u);
}
