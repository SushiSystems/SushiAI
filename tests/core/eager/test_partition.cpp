/**************************************************************************/
/* test_partition.cpp                                                     */
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
 * @file test_partition.cpp
 * @brief Tests the shape a partition of a traced step has.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include "eager_test_support.hpp"
#include <SushiAI/eager/partition.hpp>

using namespace SushiAI;
using Eager::Partition;
using Eager::RegionInfo;
namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Expects every region key of @p partition to be below its optimizer key. */
    void expect_region_keys_below_optimizer(const Partition& partition)
    {
        for (const RegionInfo& region : partition.regions)
            EXPECT_LT(region.key, partition.optimizer_key) << region.name;
    }

    /** @brief Expects the key of every node of @p partition to be below its optimizer key. */
    void expect_node_keys_below_optimizer(const Partition& partition)
    {
        for (std::size_t node = 0; node < partition.region_of_node.size(); ++node)
            EXPECT_LT(partition.region_of_node[node], partition.optimizer_key) << "node " << node;
    }
} // namespace

/** @brief Tests that a traced step's partition assigns every node and lists at least a region. */
TEST(Small_Partition, ThePartitionOfATracedStepCoversEveryNode)
{
    const Support::TracedStep step = Support::encoder_head_loss(Support::Head::Tanh);

    EXPECT_EQ(step.partition.region_of_node.size(), step.graph.node_count());
    EXPECT_FALSE(step.partition.regions.empty());
}

/** @brief Tests that every region key of a traced step is below its optimizer key. */
TEST(Small_Partition, TheOptimizerKeyIsAboveEveryRegionKey)
{
    const Support::TracedStep step = Support::encoder_head_loss(Support::Head::Tanh);

    expect_region_keys_below_optimizer(step.partition);
}

/** @brief Tests that the key of every node of a traced step is below its optimizer key. */
TEST(Small_Partition, TheOptimizerKeyIsAboveEveryNodeKey)
{
    const Support::TracedStep step = Support::encoder_head_loss(Support::Head::Tanh);

    expect_node_keys_below_optimizer(step.partition);
}
