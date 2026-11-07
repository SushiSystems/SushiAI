/**************************************************************************/
/* test_in_memory_replica_model.cpp                                       */
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
 * @file test_in_memory_replica_model.cpp
 * @brief Tests that InMemoryReplicaModel holds the counts it was given and the rows it was fed.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/distribution/in_memory_replica_model.hpp>
#include <SushiAI/distribution/replica_model.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using namespace SushiAI::Distribution;

/** @brief Tests that the model holds as many parameters and results as it was given. */
TEST(Small_InMemoryReplicaModel, HoldsTheCountsItWasGiven)
{
    const InMemoryReplicaModel model(3, 2);

    EXPECT_EQ(model.parameters().size(), 3u);
    EXPECT_EQ(model.results().size(), 2u);
}

/** @brief Tests that a model given no result holds none. */
TEST(Small_InMemoryReplicaModel, ARequestForNoResultHoldsNone)
{
    const InMemoryReplicaModel model(1, 0);

    EXPECT_TRUE(model.results().empty());
}

/** @brief Tests that each tensor is described and has no storage. */
TEST(Small_InMemoryReplicaModel, ATensorIsDescribedAndHasNoStorage)
{
    const InMemoryReplicaModel model(1, 1);

    EXPECT_EQ(model.parameters()[0].num_elements, 1);
    EXPECT_FALSE(model.parameters()[0].storage);
}

/** @brief Tests that the rows fed through the interface are kept in order. */
TEST(Small_InMemoryReplicaModel, KeepsEachFeedInOrderThroughTheInterface)
{
    InMemoryReplicaModel model(1, 1);
    ReplicaModel& port = model;
    ReplicaRows first;
    first.micro_batch_index = 0;
    ReplicaRows second;
    second.micro_batch_index = 1;

    SA_EXPECT_OK(port.feed(first));
    SA_EXPECT_OK(port.feed(second));

    ASSERT_EQ(model.fed().size(), 2u);
    EXPECT_EQ(model.fed()[0].micro_batch_index, 0u);
    EXPECT_EQ(model.fed()[1].micro_batch_index, 1u);
}
