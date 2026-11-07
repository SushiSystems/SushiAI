/**************************************************************************/
/* test_region_boundary.cpp                                               */
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
 * @file test_region_boundary.cpp
 * @brief Tests which values find_boundaries lists as the imports and exports of a region.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "eager_test_support.hpp"
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Eager::BoundaryMap;
using Eager::Direction;
using Eager::RegionBoundary;
using Graph::ValueID;
namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Reports whether @p values holds @p value. */
    [[nodiscard]] bool holds(const std::vector<ValueID>& values, ValueID value)
    {
        return std::find(values.begin(), values.end(), value) != values.end();
    }

    /** @brief Returns the key of the region that produces @p value, or K_NO_REGION for a leaf. */
    [[nodiscard]] Eager::RegionKey key_of_producer(const Support::TracedStep& step, ValueID value)
    {
        const Graph::NodeID producer = step.graph.value(value).producer;
        return producer == Graph::K_NO_NODE ? Eager::K_NO_REGION
                                            : step.partition.region_of_node[producer];
    }

    /**
     * @brief Expects every import of @p map that a region computes to be an export of a lower key.
     * @return The number of imports that another region computes.
     */
    std::size_t expect_imports_come_from_lower_exports(const Support::TracedStep& step,
                                                       const BoundaryMap& map)
    {
        std::size_t checked = 0;
        for (std::size_t region = 0; region < map.regions.size(); ++region)
        {
            for (ValueID value : map.regions[region].imports)
            {
                const Eager::RegionKey producer = key_of_producer(step, value);
                if (producer == Eager::K_NO_REGION)
                    continue;

                EXPECT_LT(producer, step.partition.regions[region].key) << "value " << value;

                bool exported = false;
                for (std::size_t other = 0; other < map.regions.size(); ++other)
                {
                    if (step.partition.regions[other].key == producer)
                        exported = holds(map.regions[other].exports, value);
                }
                EXPECT_TRUE(exported) << "value " << value;
                ++checked;
            }
        }
        return checked;
    }
} // namespace

/** @brief Holds one traced head variant of encoder_head_loss(). */
struct HeadVariant
{
    Support::Head head;
    bool extra;
};

class Small_RegionBoundaryHeads : public ::testing::TestWithParam<HeadVariant>
{
};

/**
 * @brief Tests that a forward region exports what a later region reads, and that every import
 * is a leaf or an export of a region with a lower key.
 */
TEST_P(Small_RegionBoundaryHeads, AForwardRegionExportsWhatALaterRegionReads)
{
    const Support::TracedStep step =
        Support::encoder_head_loss(GetParam().head, GetParam().extra);
    const BoundaryMap map = Support::boundaries_of(step);

    const std::size_t encoder = Support::region_at(step, "encoder#0", Direction::Forward);
    const std::size_t head = Support::region_at(step, "head#0", Direction::Forward);
    ASSERT_LT(encoder, map.regions.size());
    ASSERT_LT(head, map.regions.size());

    EXPECT_TRUE(holds(map.regions[encoder].exports, step.hidden));
    EXPECT_TRUE(holds(map.regions[head].imports, step.hidden));
    EXPECT_TRUE(holds(map.regions[head].exports, step.logits));
    EXPECT_GT(expect_imports_come_from_lower_exports(step, map), 0u);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_RegionBoundaryHeads,
                         ::testing::Values(HeadVariant{Support::Head::Tanh, false},
                                           HeadVariant{Support::Head::ReLU, true}));

/** @brief Tests that an observed value is exported even when no region reads it. */
TEST(Small_RegionBoundary, AnObservedValueIsExportedWithNoReader)
{
    const Support::TracedStep step = Support::linear_then_relu(false);
    const BoundaryMap observed = Support::boundaries_of(step);

    ASSERT_EQ(observed.regions.size(), 1u);
    EXPECT_EQ(observed.regions[0].exports, std::vector<ValueID>{step.hidden});
}

/** @brief Tests that a region exports nothing when nothing is observed and no region reads it. */
TEST(Small_RegionBoundary, WithNoObservedValueNothingIsExported)
{
    const Support::TracedStep step = Support::linear_then_relu(false);
    const BoundaryMap unobserved =
        Eager::find_boundaries(step.graph, step.partition, SushiRuntime::span<const ValueID>())
            .value();

    ASSERT_EQ(unobserved.regions.size(), 1u);
    EXPECT_TRUE(unobserved.regions[0].exports.empty());
}

TEST(Small_RegionBoundary, AnExportIsNamedByItsPlaceInTheRegion)
{
    const Support::TracedStep step = Support::linear_then_relu(false);
    const BoundaryMap map = Support::boundaries_of(step);
    const RegionBoundary& region = map.regions[0];

    // The region holds MATMUL, BIAS_ADD and RELU in that order; the leaves are in no region.
    ASSERT_EQ(region.export_positions.size(), 1u);
    EXPECT_EQ(region.export_positions[0].node, 2u);
    EXPECT_EQ(region.export_positions[0].result, 0u);
}

TEST(Small_RegionBoundary, ImportsAreListedOnceInReadOrder)
{
    Support::TracedStep step;
    const Graph::GraphBuilder builder(step.graph);
    Eager::ScopeTrace trace(step.graph);

    const ValueID a = builder.input(Shape{4}, "a");
    const ValueID b = builder.input(Shape{4}, "b");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID first, builder.add(b, a));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID second, builder.add(a, b));
    SA_ASSERT_OK_AND_ASSIGN(step.hidden, builder.add(first, second));
    Support::finish_forward(step, trace);

    const BoundaryMap map = Support::boundaries_of(step);
    ASSERT_EQ(map.regions.size(), 1u);
    EXPECT_EQ(map.regions[0].imports, (std::vector<ValueID>{b, a}));
}

TEST(Small_RegionBoundary, AnAccumulateSharesTheBufferOfItsFirstOperand)
{
    Support::TracedStep step;
    const Graph::GraphBuilder builder(step.graph);
    Eager::ScopeTrace trace(step.graph);

    const ValueID x = builder.input(Shape{4}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID first, builder.relu(x));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID second, builder.tanh(x));
    SA_ASSERT_OK_AND_ASSIGN(step.hidden, builder.accumulate(first, second));
    Support::finish_forward(step, trace);

    const BoundaryMap map = Support::boundaries_of(step);
    EXPECT_EQ(map.buffer_of[step.hidden], map.buffer_of[first]);
    EXPECT_NE(map.buffer_of[second], map.buffer_of[first]);

    // The sum is observed, so every value of its buffer is written into shared storage.
    EXPECT_TRUE(holds(map.regions[0].exports, first));
    EXPECT_TRUE(holds(map.regions[0].exports, step.hidden));
    EXPECT_FALSE(holds(map.regions[0].exports, second));
}

TEST(Small_RegionBoundary, AnAccumulateIntoALeafIsAnInvalidDescription)
{
    Support::TracedStep step;
    const Graph::GraphBuilder builder(step.graph);
    Eager::ScopeTrace trace(step.graph);

    const ValueID x = builder.input(Shape{4}, "x");
    SA_ASSERT_OK_AND_ASSIGN(step.hidden, builder.accumulate(x, builder.relu(x)));
    Support::finish_forward(step, trace);

    SA_EXPECT_FAILURE(Support::try_boundaries_of(step), StatusCode::InvalidDescription,
                      "ACCUMULATE", "leaf");
}

/** @brief Tests that find_boundaries stops the process when the partition misses nodes. */
TEST(Medium_RegionBoundaryFatal, APartitionOfAnotherGraphIsFatal)
{
    Support::TracedStep step = Support::linear_then_relu(false);
    step.partition.region_of_node.pop_back();

    SA_EXPECT_FATAL((void)Support::try_boundaries_of(step), "partition assigns");
}
