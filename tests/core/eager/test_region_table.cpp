/**************************************************************************/
/* test_region_table.cpp                                                  */
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
 * @file test_region_table.cpp
 * @brief Tests what RegionTable keeps, records again, builds and drops from step to step.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include "eager_test_support.hpp"
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/region_table.hpp>
#include <SushiAI/eager/storage_id.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Eager::Direction;
using Eager::RegionAction;
using Eager::RegionOrder;
using Eager::RegionTable;
using Eager::StepPlan;
using Eager::StorageID;
namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Plans @p step on @p table with the storage a session would place its leaves in. */
    [[nodiscard]] StepPlan plan(RegionTable& table, const Support::TracedStep& step)
    {
        const Eager::BoundaryMap boundaries = Support::boundaries_of(step);
        const std::vector<StorageID> preset = Support::presets_of(step);
        return table.plan(step.graph, step.partition, boundaries, Support::span_of(preset));
    }

    /** @brief Returns the decision @p planned holds for the region named @p name. */
    [[nodiscard]] RegionOrder order_of(const StepPlan& planned, const Support::TracedStep& step,
                                       const char* name, Direction direction)
    {
        const std::size_t at = Support::region_at(step, name, direction);
        EXPECT_LT(at, planned.regions.size()) << name;
        return at < planned.regions.size() ? planned.regions[at] : RegionOrder{};
    }

    /** @brief Counts the regions of @p planned whose action is @p action. */
    [[nodiscard]] std::size_t count(const StepPlan& planned, RegionAction action)
    {
        std::size_t found = 0;
        for (const RegionOrder& order : planned.regions)
            found += order.action == action ? 1 : 0;

        return found;
    }
} // namespace

TEST(Small_RegionTable, TheFirstStepBuildsEveryRegion)
{
    Eager::StorageIDSource ids;
    RegionTable table(ids);
    const Support::TracedStep step = Support::encoder_head_loss(Support::Head::Tanh);

    const StepPlan planned = plan(table, step);

    ASSERT_EQ(planned.regions.size(), step.partition.regions.size());
    EXPECT_EQ(count(planned, RegionAction::Build), planned.regions.size());
    EXPECT_TRUE(planned.dropped.empty());
    EXPECT_TRUE(planned.released.empty());
    EXPECT_FALSE(planned.created.empty());

    for (const RegionOrder& order : planned.regions)
        EXPECT_EQ(order.previous_key, Eager::K_NO_REGION);
}

TEST(Small_RegionTable, TheSameTraceKeepsEveryRegion)
{
    Eager::StorageIDSource ids;
    RegionTable table(ids);
    const StepPlan first = plan(table, Support::encoder_head_loss(Support::Head::Tanh));

    const Support::TracedStep again = Support::encoder_head_loss(Support::Head::Tanh);
    const StepPlan second = plan(table, again);

    EXPECT_EQ(count(second, RegionAction::Keep), second.regions.size());
    EXPECT_TRUE(second.created.empty());
    EXPECT_TRUE(second.released.empty());
    EXPECT_TRUE(second.dropped.empty());

    // A kept region reads and writes the storage it was recorded with.
    EXPECT_EQ(second.storage_of_value[again.hidden], first.storage_of_value[again.hidden]);
    EXPECT_EQ(second.storage_of_value[again.loss], first.storage_of_value[again.loss]);
}

TEST(Small_RegionTable, AChangedScopeRebuildsItsTwoRegionsAndNoOther)
{
    Eager::StorageIDSource ids;
    RegionTable table(ids);
    (void)plan(table, Support::encoder_head_loss(Support::Head::Tanh));

    const Support::TracedStep flipped = Support::encoder_head_loss(Support::Head::ReLU);
    const StepPlan planned = plan(table, flipped);

    EXPECT_EQ(order_of(planned, flipped, "head#0", Direction::Forward).action,
              RegionAction::Build);
    EXPECT_EQ(order_of(planned, flipped, "head#0", Direction::Backward).action,
              RegionAction::Build);

    // The regions around the head read its results from the storage of the last step.
    EXPECT_EQ(order_of(planned, flipped, "encoder#0", Direction::Forward).action,
              RegionAction::Keep);
    EXPECT_EQ(order_of(planned, flipped, "loss#0", Direction::Forward).action,
              RegionAction::Keep);
    EXPECT_EQ(order_of(planned, flipped, "encoder#0", Direction::Backward).action,
              RegionAction::Keep);
    EXPECT_EQ(count(planned, RegionAction::Build), 2u);
}

TEST(Small_RegionTable, AReaderKeepsTheStorageOfAResultWhoseProducerChanged)
{
    Eager::StorageIDSource ids;
    RegionTable table(ids);
    const Support::TracedStep tanh = Support::encoder_head_loss(Support::Head::Tanh);
    const StepPlan first = plan(table, tanh);

    const Support::TracedStep relu = Support::encoder_head_loss(Support::Head::ReLU);
    const StepPlan second = plan(table, relu);

    EXPECT_EQ(second.storage_of_value[relu.logits], first.storage_of_value[tanh.logits]);
    EXPECT_NE(second.storage_of_value[relu.logits], Eager::K_NO_STORAGE);
}

TEST(Small_RegionTable, AScopeThatAppearsRenumbersARegionWithoutRebuildingIt)
{
    Eager::StorageIDSource ids;
    RegionTable table(ids);
    const Support::TracedStep three = Support::encoder_head_loss(Support::Head::Tanh);
    const StepPlan first = plan(table, three);
    const Eager::RegionKey old_key =
        order_of(first, three, "encoder#0", Direction::Backward).key;

    const Support::TracedStep four = Support::encoder_head_loss(Support::Head::Tanh, true);
    const StepPlan second = plan(table, four);

    // Three segments give the encoder's backward region key 5, and four give it key 7.
    const RegionOrder backward = order_of(second, four, "encoder#0", Direction::Backward);
    EXPECT_EQ(old_key, 5u);
    EXPECT_EQ(backward.key, 7u);
    EXPECT_EQ(backward.previous_key, old_key);
    EXPECT_EQ(backward.action, RegionAction::Record);

    EXPECT_EQ(order_of(second, four, "encoder#0", Direction::Forward).action,
              RegionAction::Keep);
    EXPECT_EQ(order_of(second, four, "extra#0", Direction::Forward).action,
              RegionAction::Build);
    EXPECT_EQ(order_of(second, four, "extra#0", Direction::Forward).previous_key,
              Eager::K_NO_REGION);
}

TEST(Small_RegionTable, AScopeThatDisappearsIsDroppedByItsOldKeys)
{
    Eager::StorageIDSource ids;
    RegionTable table(ids);
    const Support::TracedStep four = Support::encoder_head_loss(Support::Head::Tanh, true);
    const StepPlan first = plan(table, four);
    const Eager::RegionKey forward = order_of(first, four, "extra#0", Direction::Forward).key;
    const Eager::RegionKey backward = order_of(first, four, "extra#0", Direction::Backward).key;

    const StepPlan second = plan(table, Support::encoder_head_loss(Support::Head::Tanh));

    EXPECT_EQ(second.dropped, (std::vector<Eager::RegionKey>{forward, backward}));
    EXPECT_FALSE(second.released.empty());
}

TEST(Small_RegionTable, ALeafWithoutStorageIsFatal)
{
    Eager::StorageIDSource ids;
    RegionTable table(ids);
    const Support::TracedStep step = Support::linear_then_relu(false);
    const Eager::BoundaryMap boundaries = Support::boundaries_of(step);
    const std::vector<StorageID> preset(step.graph.value_count(), Eager::K_NO_STORAGE);

    SA_EXPECT_FATAL(
        (void)table.plan(step.graph, step.partition, boundaries, Support::span_of(preset)),
        "linear#0 forward", "no storage");
}

TEST(Small_RegionTable, ForgetBuildsEveryRegionAgain)
{
    Eager::StorageIDSource ids;
    RegionTable table(ids);
    const StepPlan first = plan(table, Support::encoder_head_loss(Support::Head::Tanh));

    const std::vector<StorageID> released = table.forget();
    EXPECT_EQ(released.size(), first.created.size());
    EXPECT_EQ(table.live_count(), 0u);

    const StepPlan second = plan(table, Support::encoder_head_loss(Support::Head::Tanh));
    EXPECT_EQ(count(second, RegionAction::Build), second.regions.size());
    EXPECT_TRUE(second.dropped.empty());
}
