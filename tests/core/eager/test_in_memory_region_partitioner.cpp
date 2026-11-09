/**************************************************************************/
/* test_in_memory_region_partitioner.cpp                                  */
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
 * @file test_in_memory_region_partitioner.cpp
 * @brief Tests that InMemoryRegionPartitioner keeps the promise of RegionPartitioner.
 * @author Mustafa Garip
 */

#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/eager/in_memory_region_partitioner.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Eager::InMemoryRegionPartitioner;
using Eager::Partition;

namespace
{
    /** @brief Builds a partition of two regions over three nodes, the first a leaf. */
    Partition two_regions()
    {
        Partition partition;
        partition.region_of_node = {Eager::K_NO_REGION, 1, 2};
        partition.regions = {Eager::RegionInfo{1, "head#0", Eager::Direction::Forward},
                             Eager::RegionInfo{2, "head#0", Eager::Direction::Backward}};
        partition.optimizer_key = 3;
        return partition;
    }
} // namespace

/** @brief Tests that the partitioner returns the partition it was given, through the interface. */
TEST(Small_InMemoryRegionPartitioner, ReturnsTheScriptedPartition)
{
    const InMemoryRegionPartitioner partitioner(two_regions());
    const Eager::RegionPartitioner& port = partitioner;
    const Graph::Graph graph;
    const Autograd::Provenance provenance;

    const Partition cut = port.assign(graph, {}, provenance);

    EXPECT_EQ(cut.region_of_node, two_regions().region_of_node);
    ASSERT_EQ(cut.regions.size(), 2u);
    EXPECT_EQ(cut.regions[1].name, "head#0");
    EXPECT_EQ(cut.regions[1].direction, Eager::Direction::Backward);
    EXPECT_EQ(cut.optimizer_key, 3u);
}

/** @brief Tests that the default partition holds no node and no region. */
TEST(Small_InMemoryRegionPartitioner, TheDefaultPartitionIsEmpty)
{
    const InMemoryRegionPartitioner partitioner;
    const Graph::Graph graph;
    const Autograd::Provenance provenance;

    const Partition cut = partitioner.assign(graph, {}, provenance);

    EXPECT_TRUE(cut.region_of_node.empty());
    EXPECT_TRUE(cut.regions.empty());
}

/** @brief Tests that each call is journaled with the segment count and the forward node count. */
TEST(Small_InMemoryRegionPartitioner, JournalsWhatEachCallWasHanded)
{
    const InMemoryRegionPartitioner partitioner;
    const Graph::Graph graph;
    Autograd::Provenance provenance;
    provenance.forward_node_count = 7;
    const std::vector<Eager::Segment> segments(2);

    static_cast<void>(partitioner.assign(
        graph, SushiRuntime::span<const Eager::Segment>(segments.data(), segments.size()),
        provenance));
    static_cast<void>(partitioner.assign(graph, {}, Autograd::Provenance{}));

    ASSERT_EQ(partitioner.calls().size(), 2u);
    EXPECT_EQ(partitioner.calls()[0].segment_count, 2u);
    EXPECT_EQ(partitioner.calls()[0].forward_node_count, 7u);
    EXPECT_EQ(partitioner.calls()[1].segment_count, 0u);
}

/** @brief Tests that a call is journaled with the graph node count and its segments. */
TEST(Small_InMemoryRegionPartitioner, JournalsTheNodeCountAndTheSegmentsOfEachCall)
{
    const InMemoryRegionPartitioner partitioner;
    const Graph::Graph graph;
    std::vector<Eager::Segment> segments(2);
    segments[0].end = 3;
    segments[1].begin = 3;
    segments[1].end = 5;

    static_cast<void>(partitioner.assign(
        graph, SushiRuntime::span<const Eager::Segment>(segments.data(), segments.size()),
        Autograd::Provenance{}));

    ASSERT_EQ(partitioner.calls().size(), 1u);
    EXPECT_EQ(partitioner.calls()[0].node_count, 0u);
    ASSERT_EQ(partitioner.calls()[0].segments.size(), 2u);
    EXPECT_EQ(partitioner.calls()[0].segments[1].begin, 3u);
    EXPECT_EQ(partitioner.calls()[0].segments[1].end, 5u);
}

/** @brief Tests that a partitioner given an inner one returns its cut and journals the call. */
TEST(Small_InMemoryRegionPartitioner, ReturnsTheCutOfTheInnerPartitioner)
{
    const InMemoryRegionPartitioner inner(two_regions());
    // Through the port, so the wrapping constructor is chosen and not the copy constructor.
    const Eager::RegionPartitioner& port = inner;
    const InMemoryRegionPartitioner partitioner(port);
    const Graph::Graph graph;

    const Partition cut = partitioner.assign(graph, {}, Autograd::Provenance{});

    EXPECT_EQ(cut.region_of_node, two_regions().region_of_node);
    EXPECT_EQ(partitioner.calls().size(), 1u);
    EXPECT_EQ(inner.calls().size(), 1u);
}

/** @brief Tests that a scripted refusal throws invalid_argument and the call is journaled. */
TEST(Small_InMemoryRegionPartitioner, AScriptedRefusalThrowsItsMessage)
{
    InMemoryRegionPartitioner partitioner;
    partitioner.script_refusal("this graph has no cut");
    const Graph::Graph graph;

    EXPECT_THROW(static_cast<void>(partitioner.assign(graph, {}, Autograd::Provenance{})),
                 std::invalid_argument);
    EXPECT_EQ(partitioner.calls().size(), 1u);
}
