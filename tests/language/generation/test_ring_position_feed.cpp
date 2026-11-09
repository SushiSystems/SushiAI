/**************************************************************************/
/* test_ring_position_feed.cpp                                           */
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
 * @file test_ring_position_feed.cpp
 * @brief Tests RingPositionFeed, which feeds a token's cache slot and the angles of its position.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include "position_feed_test_support.hpp"
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/language/generation/position_feed.hpp>
#include <SushiAI/language/generation/ring_position_feed.hpp>

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

    /** @brief Holds the position the angle rows are worked at: 11 over 4 rows is row 3. */
    constexpr std::int64_t K_POSITION = 11;

    /** @brief Names a position and the cache row it is written to. */
    struct SlotCase
    {
        std::int64_t position; /**< The position given to place. */
        std::int32_t slot;     /**< The position modulo T. */
    };

    /** @brief Names a ring that does not fit and what the refusal names. */
    struct RefusalCase
    {
        RingStepLeaves ring;     /**< The ring given to make. */
        std::string_view first;  /**< A substring the message holds. */
        std::string_view second; /**< Another substring the message holds. */
    };

    /** @brief Returns the slot rows: position 11 over 4 rows is row 3, position 2 is row 2. */
    std::vector<SlotCase> slot_cases()
    {
        return {{K_POSITION, 3}, {2, 2}};
    }

    /** @brief Returns the pairs of a row, which write_counted_row fills with position + pair. */
    std::vector<std::size_t> pair_cases()
    {
        return {0u, 1u, 2u};
    }

    /** @brief Returns the ring with the member @p change gave another value. */
    template <typename Change>
    RingStepLeaves ring_with(Change change)
    {
        RingStepLeaves ring = Support::feed_leaves().ring;
        change(ring);
        return ring;
    }

    /** @brief Returns a ring without an angle function, an unnamed sine, a slot of the wrong type and a narrow leaf. */
    std::vector<RefusalCase> refusal_cases()
    {
        return {
            {ring_with([](RingStepLeaves& ring) { ring.angles = nullptr; }), "no function",
             "'feed_cosine'"},
            {ring_with([](RingStepLeaves& ring) { ring.sine = "feed_tangent"; }),
             "'feed_tangent'", ""},
            {ring_with([](RingStepLeaves& ring) { ring.slot = Support::K_FEED_COSINE; }),
             "'feed_cosine'", "int32 leaf of [1, 1]"},
            {ring_with([](RingStepLeaves& ring) { ring.sine = Support::K_FEED_POSITION; }),
             "'feed_position'", "float32 leaf of [1, 3]"}};
    }

    /** @brief Builds the ring feed of a host engine and places positions into it. */
    class Small_GenerationRingPositionFeed : public Support::HostFeedTest
    {
        protected:
            /** @brief Returns the ring feed over K_FEED_WINDOW rows, or null. */
            std::unique_ptr<PositionFeed> make_feed()
            {
                const InMemoryFedLeaves fed = Support::fed_leaves_of(false, true);
                Result<std::unique_ptr<PositionFeed>> made = RingPositionFeed::make(
                    engine(), fed, Support::feed_leaves().ring, Support::K_FEED_WINDOW);
                EXPECT_TRUE(made.has_value());
                return made.has_value() ? std::move(made).value() : nullptr;
            }
    };

    /** @brief Provides the positions and the rows they wrap to. */
    class Small_GenerationRingSlot : public Small_GenerationRingPositionFeed,
                                     public ::testing::WithParamInterface<SlotCase>
    {

    };

    /** @brief Provides the pairs of the angle row. */
    class Small_GenerationRingAngles : public Small_GenerationRingPositionFeed,
                                       public ::testing::WithParamInterface<std::size_t>
    {

    };

    /** @brief Provides the rings that are refused. */
    class Small_GenerationRingRefusal : public Support::HostFeedTest,
                                        public ::testing::WithParamInterface<RefusalCase>
    {

    };
} // namespace

/** @brief Tests that the feed lists the slot, the cosines and the sines, in that order. */
TEST_F(Small_GenerationRingPositionFeed, TheFeedListsTheSlotThenTheAngles)
{
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

/** @brief Tests that the slot is the position modulo T. */
TEST_P(Small_GenerationRingSlot, TheSlotIsThePositionModuloT)
{
    const SlotCase& row = GetParam();
    const std::unique_ptr<PositionFeed> feed = make_feed();
    ASSERT_NE(feed, nullptr);

    feed->place(row.position);

    EXPECT_EQ(feed->feeds()[K_SLOT].tensor->data_as<std::int32_t>()[0], row.slot);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationRingSlot, ::testing::ValuesIn(slot_cases()));

/** @brief Tests that pair i of the angles is those of the position: 11 + i and its negation. */
TEST_P(Small_GenerationRingAngles, TheAnglesAreThoseOfThePosition)
{
    const std::size_t pair = GetParam();
    const std::unique_ptr<PositionFeed> feed = make_feed();
    ASSERT_NE(feed, nullptr);

    feed->place(K_POSITION);

    const float counted = static_cast<float>(K_POSITION) + static_cast<float>(pair);
    EXPECT_EQ(feed->feeds()[K_COSINE].tensor->data_as<float>()[pair], counted);
    EXPECT_EQ(feed->feeds()[K_SINE].tensor->data_as<float>()[pair], -counted);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationRingAngles, ::testing::ValuesIn(pair_cases()));

/** @brief Tests that position T is the first to wrap: it writes over the row of position 0. */
TEST_F(Small_GenerationRingPositionFeed, TheFirstWrapWritesOverTheRowOfPositionZero)
{
    const std::unique_ptr<PositionFeed> feed = make_feed();
    ASSERT_NE(feed, nullptr);

    feed->place(Support::K_FEED_WINDOW);

    EXPECT_EQ(feed->feeds()[K_SLOT].tensor->data_as<std::int32_t>()[0], 0);
    EXPECT_EQ(feed->feeds()[K_COSINE].tensor->data_as<float>()[0],
              static_cast<float>(Support::K_FEED_WINDOW));
}

/** @brief Tests that a ring without its angle function and a leaf that misfits are refused. */
TEST_P(Small_GenerationRingRefusal, ARingThatDoesNotFitIsRefused)
{
    const RefusalCase& row = GetParam();
    const InMemoryFedLeaves fed = Support::fed_leaves_of(true, true);

    SA_EXPECT_FAILURE(RingPositionFeed::make(engine(), fed, row.ring, Support::K_FEED_WINDOW),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationRingRefusal,
                         ::testing::ValuesIn(refusal_cases()));
