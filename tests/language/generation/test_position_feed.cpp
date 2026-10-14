/**************************************************************************/
/* test_position_feed.cpp                                                 */
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
 * @file test_position_feed.cpp
 * @brief Tests make_position_feed, which picks a PositionFeed from the leaves a step is fed.
 * @author Mustafa Garip
 */

#include <memory>
#include <gtest/gtest.h>
#include "position_feed_test_support.hpp"
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/language/generation/position_feed.hpp>
#include <SushiAI/train/evaluation_step.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;
using Language::Generation::Detail::PositionFeed;
using Language::Generation::Detail::make_position_feed;

namespace
{
    /** @brief Builds the steps a feed is picked for. */
    class Unit_GenerationPositionFeed : public Support::FedStepTest
    {

    };
} // namespace

/** @brief Tests that a step fed a position gets the feed of one leaf that resets on a slide. */
TEST_F(Unit_GenerationPositionFeed, AStepFedAPositionGetsTheTableFeed)
{
    build_step(true, false);
    ASSERT_NE(step, nullptr);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PositionFeed> feed,
                            make_position_feed(*engine, *step, Support::feed_leaves(),
                                               Support::K_FEED_WINDOW));

    ASSERT_EQ(feed->feeds().size(), 1u);
    EXPECT_EQ(feed->feeds()[0].name, Support::K_FEED_POSITION);
    EXPECT_FALSE(feed->keeps_rows_on_slide());
}

/** @brief Tests that a step fed a cache slot gets the feed of three leaves that keeps its rows. */
TEST_F(Unit_GenerationPositionFeed, AStepFedACacheSlotGetsTheRingFeed)
{
    build_step(false, true);
    ASSERT_NE(step, nullptr);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PositionFeed> feed,
                            make_position_feed(*engine, *step, Support::feed_leaves(),
                                               Support::K_FEED_WINDOW));

    ASSERT_EQ(feed->feeds().size(), 3u);
    EXPECT_EQ(feed->feeds()[0].name, Support::K_FEED_SLOT);
    EXPECT_TRUE(feed->keeps_rows_on_slide());
}

/** @brief Tests that leaves that name no ring pick the table feed of a step fed a position. */
TEST_F(Unit_GenerationPositionFeed, LeavesWithoutARingPickTheTableFeed)
{
    build_step(true, false);
    ASSERT_NE(step, nullptr);
    const CachedStepLeaves plain{Support::K_FEED_POSITION, "feed_mask", 0.0};
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<PositionFeed> feed,
        make_position_feed(*engine, *step, plain, Support::K_FEED_WINDOW));

    EXPECT_EQ(feed->feeds().size(), 1u);
}

/** @brief Tests that a step fed both the position and the slot, or neither, is refused. */
TEST_F(Unit_GenerationPositionFeed, AStepFedBothOrNeitherIsRefused)
{
    build_step(true, true);
    ASSERT_NE(step, nullptr);
    SA_EXPECT_FAILURE(make_position_feed(*engine, *step, Support::feed_leaves(),
                                         Support::K_FEED_WINDOW),
                      StatusCode::InvalidDescription, "CachedGenerator", "both",
                      "'feed_position'", "'feed_slot'");

    step.reset();
    pool.reset();
    state.reset();
    build_step(false, false);
    ASSERT_NE(step, nullptr);
    SA_EXPECT_FAILURE(make_position_feed(*engine, *step, Support::feed_leaves(),
                                         Support::K_FEED_WINDOW),
                      StatusCode::InvalidDescription, "neither", "'feed_position'",
                      "'feed_slot'");
}
