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
 * @file test_partition.cpp
 * @brief Tests the defaults of Partition and the shape a partition of a traced step has.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include "eager_test_support.hpp"
#include <SushiAI/eager/partition.hpp>

using namespace SushiAI;
using Eager::Direction;
using Eager::K_NO_REGION;
using Eager::Partition;
using Eager::RegionInfo;
namespace Support = SushiAI::Eager::Support;

/** @brief Tests that the empty key is zero, the value a zero-filled table holds. */
TEST(Small_Partition, TheEmptyKeyIsZero)
{
    EXPECT_EQ(K_NO_REGION, Eager::RegionKey{0});
}

/** @brief Tests that a default region has no key, no name and holds the forward pass. */
TEST(Small_Partition, ADefaultRegionInfoNamesNoRegion)
{
    const RegionInfo info;

    EXPECT_EQ(info.key, K_NO_REGION);
    EXPECT_TRUE(info.name.empty());
    EXPECT_EQ(info.direction, Direction::Forward);
}

/** @brief Tests that a default partition holds no node, no region and no optimizer key. */
TEST(Small_Partition, ADefaultPartitionIsEmpty)
{
    const Partition partition;

    EXPECT_TRUE(partition.region_of_node.empty());
    EXPECT_TRUE(partition.regions.empty());
    EXPECT_EQ(partition.optimizer_key, K_NO_REGION);
}

/** @brief Tests that a traced step's partition covers every node below its optimizer key. */
TEST(Small_Partition, TheOptimizerKeyIsAboveEveryRegion)
{
    const Support::TracedStep step = Support::encoder_head_loss(Support::Head::Tanh);
    const Partition& partition = step.partition;

    EXPECT_EQ(partition.region_of_node.size(), step.graph.node_count());
    ASSERT_FALSE(partition.regions.empty());
    for (const RegionInfo& region : partition.regions)
        EXPECT_LT(region.key, partition.optimizer_key) << region.name;

    for (std::size_t node = 0; node < partition.region_of_node.size(); ++node)
        EXPECT_LT(partition.region_of_node[node], partition.optimizer_key) << "node " << node;
}
