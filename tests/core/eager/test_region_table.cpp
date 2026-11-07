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

    /** @brief Holds the plan of a step that gained a scope, with what the step before gave. */
    struct Appeared
    {
        Support::TracedStep four; /**< The step with the scope "extra". */
        StepPlan second;          /**< The plan of that step. */
        Eager::RegionKey old_key; /**< The key of the encoder's backward region before. */
    };

    /** @brief Plans a three-segment step, then the same model with a scope "extra" added. */
    [[nodiscard]] Appeared plan_after_a_scope_appears()
    {
        Eager::StorageIDSource ids;
        RegionTable table(ids);
        const Support::TracedStep three = Support::encoder_head_loss(Support::Head::Tanh);
        const StepPlan first = plan(table, three);

        Appeared appeared;
        appeared.old_key = order_of(first, three, "encoder#0", Direction::Backward).key;
        appeared.four = Support::encoder_head_loss(Support::Head::Tanh, true);
        appeared.second = plan(table, appeared.four);
        return appeared;
    }

    /** @brief Holds the plan of a step that lost a scope, with the keys the scope held. */
    struct Vanished
    {
        StepPlan second;                       /**< The plan of the step without the scope. */
        std::vector<Eager::RegionKey> old_keys; /**< The forward and backward keys it held. */
    };

    /** @brief Plans a four-segment step, then the same model without the scope "extra". */
    [[nodiscard]] Vanished plan_after_a_scope_vanishes()
    {
        Eager::StorageIDSource ids;
        RegionTable table(ids);
        const Support::TracedStep four = Support::encoder_head_loss(Support::Head::Tanh, true);
        const StepPlan first = plan(table, four);

        Vanished vanished;
        vanished.old_keys = {order_of(first, four, "extra#0", Direction::Forward).key,
                             order_of(first, four, "extra#0", Direction::Backward).key};
        vanished.second = plan(table, Support::encoder_head_loss(Support::Head::Tanh));
        return vanished;
    }
} // namespace

/** @brief Tests that a first plan builds every region with no previous key and releases nothing. */
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

/** @brief Tests that an identical trace keeps every region and creates, releases and drops nothing. */
TEST(Small_RegionTable, TheSameTraceKeepsEveryRegion)
{
    Eager::StorageIDSource ids;
    RegionTable table(ids);
    (void)plan(table, Support::encoder_head_loss(Support::Head::Tanh));

    const StepPlan second = plan(table, Support::encoder_head_loss(Support::Head::Tanh));

    EXPECT_EQ(count(second, RegionAction::Keep), second.regions.size());
    EXPECT_TRUE(second.created.empty());
    EXPECT_TRUE(second.released.empty());
    EXPECT_TRUE(second.dropped.empty());
}

/** @brief Tests that a kept region reads and writes the storage it was recorded with. */
TEST(Small_RegionTable, AKeptRegionKeepsItsStorage)
{
    Eager::StorageIDSource ids;
    RegionTable table(ids);
    const StepPlan first = plan(table, Support::encoder_head_loss(Support::Head::Tanh));

    const Support::TracedStep again = Support::encoder_head_loss(Support::Head::Tanh);
    const StepPlan second = plan(table, again);

    EXPECT_EQ(second.storage_of_value[again.hidden], first.storage_of_value[again.hidden]);
    EXPECT_EQ(second.storage_of_value[again.loss], first.storage_of_value[again.loss]);
}

/** @brief Tests that a changed scope rebuilds its own two regions and no other. */
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

/** @brief Tests that a reader keeps the storage of a result whose producer changed. */
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

/** @brief Tests that a scope that appears is built with no previous key. */
TEST(Small_RegionTable, AScopeThatAppearsIsBuilt)
{
    const Appeared appeared = plan_after_a_scope_appears();

    const RegionOrder extra =
        order_of(appeared.second, appeared.four, "extra#0", Direction::Forward);
    EXPECT_EQ(extra.action, RegionAction::Build);
    EXPECT_EQ(extra.previous_key, Eager::K_NO_REGION);
}

/** @brief Tests that a region before a scope that appears is kept. */
TEST(Small_RegionTable, AScopeThatAppearsLeavesTheRegionsBeforeItKept)
{
    const Appeared appeared = plan_after_a_scope_appears();

    EXPECT_EQ(order_of(appeared.second, appeared.four, "encoder#0", Direction::Forward).action,
              RegionAction::Keep);
}

/** @brief Tests that a region renumbered by a scope that appears is recorded under its new key. */
TEST(Small_RegionTable, AScopeThatAppearsRenumbersARegionWithoutRebuildingIt)
{
    const Appeared appeared = plan_after_a_scope_appears();

    const RegionOrder backward =
        order_of(appeared.second, appeared.four, "encoder#0", Direction::Backward);
    EXPECT_NE(backward.key, appeared.old_key);
    EXPECT_EQ(backward.previous_key, appeared.old_key);
    EXPECT_EQ(backward.action, RegionAction::Record);
}

/** @brief Tests that the plan of a scope that vanished lists the keys it held as dropped. */
TEST(Small_RegionTable, AScopeThatDisappearsIsDroppedByItsOldKeys)
{
    const Vanished vanished = plan_after_a_scope_vanishes();

    EXPECT_EQ(vanished.second.dropped, vanished.old_keys);
}

/** @brief Tests that the plan of a scope that vanished releases the storage only it used. */
TEST(Small_RegionTable, AScopeThatDisappearsReleasesItsStorage)
{
    const Vanished vanished = plan_after_a_scope_vanishes();

    EXPECT_FALSE(vanished.second.released.empty());
}

/** @brief Tests that a leaf with no storage stops the process, naming the region and the leaf. */
TEST(Medium_RegionTableFatal, ALeafWithoutStorageIsFatal)
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

/** @brief Tests that forget() releases every slot the first plan created and leaves no region. */
TEST(Small_RegionTable, ForgetReleasesEverySlotAndLeavesNoLiveRegion)
{
    Eager::StorageIDSource ids;
    RegionTable table(ids);
    const StepPlan first = plan(table, Support::encoder_head_loss(Support::Head::Tanh));

    const std::vector<StorageID> released = table.forget();

    EXPECT_EQ(released.size(), first.created.size());
    EXPECT_EQ(table.live_count(), 0u);
}

/** @brief Tests that the plan after forget() builds every region and drops nothing. */
TEST(Small_RegionTable, ThePlanAfterForgetBuildsEveryRegion)
{
    Eager::StorageIDSource ids;
    RegionTable table(ids);
    (void)plan(table, Support::encoder_head_loss(Support::Head::Tanh));
    (void)table.forget();

    const StepPlan second = plan(table, Support::encoder_head_loss(Support::Head::Tanh));

    EXPECT_EQ(count(second, RegionAction::Build), second.regions.size());
    EXPECT_TRUE(second.dropped.empty());
}
