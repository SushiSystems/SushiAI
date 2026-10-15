/**************************************************************************/
/* test_ring_position_feed.cpp                                            */
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
 * @file test_ring_position_feed.cpp
 * @brief Tests RingPositionFeed, which feeds a token's cache slot and the angles of its position.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include "position_feed_test_support.hpp"
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/language/generation/position_feed.hpp>
#include <SushiAI/language/generation/ring_position_feed.hpp>
#include <SushiAI/train/evaluation_step.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;
using Language::Generation::Detail::PositionFeed;
using Language::Generation::Detail::RingPositionFeed;

namespace
{
    /** @brief Holds the index of the slot among the leaves a ring feed lists. */
    constexpr std::size_t K_SLOT = 0;

    /** @brief Holds the index of the cosines among the leaves a ring feed lists. */
    constexpr std::size_t K_COSINE = 1;

    /** @brief Holds the index of the sines among the leaves a ring feed lists. */
    constexpr std::size_t K_SINE = 2;

    /** @brief Builds the steps a ring feed is bound to. */
    class Medium_GenerationRingPositionFeed : public Support::FedStepTest
    {
        protected:
            /** @brief Returns the ring feed of the step over K_FEED_WINDOW rows, or null. */
            std::unique_ptr<PositionFeed> make_feed()
            {
                Result<std::unique_ptr<PositionFeed>> made = RingPositionFeed::make(
                    *engine, *step, Support::feed_leaves().ring, Support::K_FEED_WINDOW);
                EXPECT_TRUE(made.has_value());
                return made.has_value() ? std::move(made).value() : nullptr;
            }
    };
} // namespace

/** @brief Tests that the feed lists the slot, the cosines and the sines, in that order. */
TEST_F(Medium_GenerationRingPositionFeed, TheFeedListsTheSlotThenTheAngles)
{
    build_step(false, true);
    ASSERT_NE(step, nullptr);
    const std::unique_ptr<PositionFeed> feed = make_feed();
    ASSERT_NE(feed, nullptr);

    const SushiRuntime::span<const Train::NamedFeed> fed = feed->feeds();
    ASSERT_EQ(fed.size(), 3u);
    EXPECT_EQ(fed[K_SLOT].name, Support::K_FEED_SLOT);
    EXPECT_EQ(fed[K_COSINE].name, Support::K_FEED_COSINE);
    EXPECT_EQ(fed[K_SINE].name, Support::K_FEED_SINE);
    EXPECT_EQ(fed[K_SLOT].tensor->dtype, DType::Int32);
    EXPECT_EQ(fed[K_SLOT].tensor->num_elements, 1);
    EXPECT_EQ(fed[K_COSINE].tensor->dtype, DType::Float32);
    EXPECT_EQ(fed[K_COSINE].tensor->num_elements, Support::K_FEED_PAIRS);
    EXPECT_EQ(fed[K_SINE].tensor->num_elements, Support::K_FEED_PAIRS);
}

/** @brief Tests that the slot is the position modulo T and the angles those of the position. */
TEST_F(Medium_GenerationRingPositionFeed, PlaceWrapsTheSlotAndKeepsThePositionOfTheAngles)
{
    build_step(false, true);
    ASSERT_NE(step, nullptr);
    const std::unique_ptr<PositionFeed> feed = make_feed();
    ASSERT_NE(feed, nullptr);
    const SushiRuntime::span<const Train::NamedFeed> fed = feed->feeds();
    ASSERT_EQ(fed.size(), 3u);

    // Position 11 over 4 rows is row 3, and write_counted_row writes 11 + pair.
    feed->place(11);
    EXPECT_EQ(fed[K_SLOT].tensor->data_as<std::int32_t>()[0], 3);
    for (std::size_t pair = 0; pair < static_cast<std::size_t>(Support::K_FEED_PAIRS); ++pair)
    {
        const float counted = 11.0f + static_cast<float>(pair);
        EXPECT_EQ(fed[K_COSINE].tensor->data_as<float>()[pair], counted) << "pair " << pair;
        EXPECT_EQ(fed[K_SINE].tensor->data_as<float>()[pair], -counted) << "pair " << pair;
    }

    // Position T is the first to wrap: it writes over the row of position 0.
    feed->place(Support::K_FEED_WINDOW);
    EXPECT_EQ(fed[K_SLOT].tensor->data_as<std::int32_t>()[0], 0);
    EXPECT_EQ(fed[K_COSINE].tensor->data_as<float>()[0],
              static_cast<float>(Support::K_FEED_WINDOW));
    feed->place(2);
    EXPECT_EQ(fed[K_SLOT].tensor->data_as<std::int32_t>()[0], 2);
}

/** @brief Tests that every cached row stays right after a slide. */
TEST_F(Medium_GenerationRingPositionFeed, EveryRowStaysRightAfterASlide)
{
    build_step(false, true);
    ASSERT_NE(step, nullptr);
    const std::unique_ptr<PositionFeed> feed = make_feed();
    ASSERT_NE(feed, nullptr);

    EXPECT_TRUE(feed->keeps_rows_on_slide());
}

/** @brief Tests that a ring without its angle function and a leaf that misfits are refused. */
TEST_F(Medium_GenerationRingPositionFeed, ARingThatDoesNotFitIsRefused)
{
    build_step(true, true);
    ASSERT_NE(step, nullptr);

    RingStepLeaves silent = Support::feed_leaves().ring;
    silent.angles = nullptr;
    SA_EXPECT_FAILURE(RingPositionFeed::make(*engine, *step, silent, Support::K_FEED_WINDOW),
                      StatusCode::InvalidDescription, "no function", "'feed_cosine'",
                      "'feed_sine'");

    RingStepLeaves unnamed = Support::feed_leaves().ring;
    unnamed.sine = "feed_tangent";
    SA_EXPECT_FAILURE(RingPositionFeed::make(*engine, *step, unnamed, Support::K_FEED_WINDOW),
                      StatusCode::InvalidDescription, "'feed_tangent'");

    RingStepLeaves crossed = Support::feed_leaves().ring;
    crossed.slot = Support::K_FEED_COSINE;
    SA_EXPECT_FAILURE(RingPositionFeed::make(*engine, *step, crossed, Support::K_FEED_WINDOW),
                      StatusCode::InvalidDescription, "'feed_cosine'", "int32 leaf of [1, 1]");

    RingStepLeaves narrow = Support::feed_leaves().ring;
    narrow.sine = Support::K_FEED_POSITION;
    SA_EXPECT_FAILURE(RingPositionFeed::make(*engine, *step, narrow, Support::K_FEED_WINDOW),
                      StatusCode::InvalidDescription, "'feed_position'",
                      "float32 leaf of [1, 3]");
}
