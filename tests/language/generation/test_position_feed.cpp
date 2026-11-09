/**************************************************************************/
/* test_position_feed.cpp                                                */
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
 * @file test_position_feed.cpp
 * @brief Tests make_position_feed, which picks a PositionFeed from the leaves a step is fed.
 * @author Mustafa Garip
 */

#include <memory>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include "position_feed_test_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/language/generation/position_feed.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;
using Language::Generation::Detail::PositionFeed;
using Language::Generation::Detail::make_position_feed;

namespace
{
    /** @brief Names the leaves a call gives and the leaves the step is fed. */
    struct RefusalCase
    {
        bool has_position;       /**< Whether the step is fed the table position. */
        bool has_ring;           /**< Whether the step is fed the ring slot and its angles. */
        std::string_view first;  /**< A substring the message holds. */
        std::string_view second; /**< Another substring the message holds. */
    };

    /** @brief Returns the leaves a call gives: the ones that declare a ring, and the plain ones. */
    std::vector<CachedStepLeaves> table_cases()
    {
        return {Support::feed_leaves(),
                CachedStepLeaves{Support::K_FEED_POSITION, "feed_mask", 0.0}};
    }

    /** @brief Returns a step fed both leaves and a step fed neither. */
    std::vector<RefusalCase> refusal_cases()
    {
        return {{true, true, "both", "'feed_position'"},
                {false, false, "neither", "'feed_slot'"}};
    }

    /** @brief Provides the leaves of the calls that pick the table feed. */
    class Small_GenerationPositionFeedTable
        : public Support::HostFeedTest,
          public ::testing::WithParamInterface<CachedStepLeaves>
    {

    };

    /** @brief Provides the steps that are refused. */
    class Small_GenerationPositionFeedRefusal
        : public Support::HostFeedTest,
          public ::testing::WithParamInterface<RefusalCase>
    {

    };

    /** @brief Provides a host engine. */
    class Small_GenerationPositionFeed : public Support::HostFeedTest
    {

    };
} // namespace

/** @brief Tests that a step fed a position gets the feed of one leaf, with or without a ring named. */
TEST_P(Small_GenerationPositionFeedTable, AStepFedAPositionGetsTheTableFeed)
{
    const InMemoryFedLeaves fed = Support::fed_leaves_of(true, false);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PositionFeed> feed,
                            make_position_feed(engine(), fed, GetParam(),
                                               Support::K_FEED_WINDOW));

    ASSERT_EQ(feed->feeds().size(), 1u);
    EXPECT_EQ(feed->feeds()[0].name, Support::K_FEED_POSITION);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationPositionFeedTable,
                         ::testing::ValuesIn(table_cases()));

/** @brief Tests that a step fed a cache slot gets the feed of three leaves. */
TEST_F(Small_GenerationPositionFeed, AStepFedACacheSlotGetsTheRingFeed)
{
    const InMemoryFedLeaves fed = Support::fed_leaves_of(false, true);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PositionFeed> feed,
                            make_position_feed(engine(), fed, Support::feed_leaves(),
                                               Support::K_FEED_WINDOW));

    EXPECT_EQ(feed->feeds().size(), 3u);
}

/** @brief Tests that a step fed both the position and the slot, or neither, is refused. */
TEST_P(Small_GenerationPositionFeedRefusal, AStepFedBothOrNeitherIsRefused)
{
    const RefusalCase& row = GetParam();
    const InMemoryFedLeaves fed = Support::fed_leaves_of(row.has_position, row.has_ring);

    SA_EXPECT_FAILURE(make_position_feed(engine(), fed, Support::feed_leaves(),
                                         Support::K_FEED_WINDOW),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationPositionFeedRefusal,
                         ::testing::ValuesIn(refusal_cases()));
