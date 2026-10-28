/**************************************************************************/
/* test_recent_partners.cpp                                               */
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
 * @file test_recent_partners.cpp
 * @brief Tests the mosaic partner source drawn from the recent buffer.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/dataset_partners.hpp>
#include <SushiAI/vision/imaging/recent_partners.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Returns the buffer a source would hold after the given indices, oldest first. */
    std::deque<std::size_t> buffer_after(RecentPartners& partners,
                                         std::initializer_list<std::size_t> indices)
    {
        Random random(1u);
        for (const std::size_t index : indices)
            static_cast<void>(partners.choose(index, random));

        return partners.recent();
    }

} // namespace

/** @brief Tests that the buffer length is the smallest of the dataset, eight batches and 1000. */
TEST(Small_ImagingPartners, RecentCapacityIsTheSmallestOfItsThreeBounds)
{
    EXPECT_EQ(recent_capacity(5, 16), 5u);
    EXPECT_EQ(recent_capacity(10000, 16), 128u);
    EXPECT_EQ(recent_capacity(10000, 200), 1000u);
    EXPECT_EQ(recent_capacity(1000, 125), 1000u);
    EXPECT_EQ(require_value(RecentPartners::make(10000, 16))->capacity(), 128u);
    EXPECT_EQ(K_RECENT_BATCHES, 8u);
    EXPECT_EQ(K_RECENT_LIMIT, 1000u);
}

/** @brief Tests that the first sample's partners are the sample itself. */
TEST(Small_ImagingPartners, RecentSourceGivesTheFirstSampleItselfThreeTimes)
{
    const std::unique_ptr<RecentPartners> partners_owner =
        require_value(RecentPartners::make(100, 16));
    RecentPartners& partners = *partners_owner;
    Random random(5u);

    const MosaicPartners chosen = partners.choose(42, random);

    EXPECT_EQ(chosen, (MosaicPartners{42, 42, 42}));
    EXPECT_EQ(partners.recent(), (std::deque<std::size_t>{42}));
}

/** @brief Tests that an addition evicts the oldest index once the buffer reaches its capacity. */
TEST(Small_ImagingPartners, RecentSourceEvictsTheOldestIndexAtCapacity)
{
    const std::unique_ptr<RecentPartners> three_owner = require_value(RecentPartners::make(3, 16));
    RecentPartners& three = *three_owner;
    const std::unique_ptr<RecentPartners> two_owner = require_value(RecentPartners::make(2, 16));
    RecentPartners& two = *two_owner;
    ASSERT_EQ(three.capacity(), 3u);
    ASSERT_EQ(two.capacity(), 2u);

    // The reference pops when 1 < len(buffer) >= capacity, so a full buffer holds capacity - 1.
    EXPECT_EQ(buffer_after(three, {0, 1}), (std::deque<std::size_t>{0, 1}));
    EXPECT_EQ(buffer_after(three, {2}), (std::deque<std::size_t>{1, 2}));
    EXPECT_EQ(buffer_after(three, {0}), (std::deque<std::size_t>{2, 0}));
    EXPECT_EQ(buffer_after(two, {0}), (std::deque<std::size_t>{0}));
    EXPECT_EQ(buffer_after(two, {1}), (std::deque<std::size_t>{1}));
    EXPECT_EQ(buffer_after(two, {0}), (std::deque<std::size_t>{0}));
}

/** @brief Tests that an index already in the buffer is neither added again nor moved. */
TEST(Small_ImagingPartners, RecentSourceDoesNotAddAnIndexItAlreadyHolds)
{
    const std::unique_ptr<RecentPartners> partners_owner =
        require_value(RecentPartners::make(100, 16));
    RecentPartners& partners = *partners_owner;

    EXPECT_EQ(buffer_after(partners, {7, 9, 7, 7, 9}), (std::deque<std::size_t>{7, 9}));
}

/** @brief Tests that a dataset of one image keeps its only index in the buffer. */
TEST(Small_ImagingPartners, RecentSourceOfOneImageNeverEmptiesItsBuffer)
{
    const std::unique_ptr<RecentPartners> partners_owner =
        require_value(RecentPartners::make(1, 16));
    RecentPartners& partners = *partners_owner;
    Random random(3u);

    ASSERT_EQ(partners.capacity(), 1u);
    EXPECT_EQ(partners.choose(0, random), (MosaicPartners{0, 0, 0}));
    EXPECT_EQ(partners.choose(0, random), (MosaicPartners{0, 0, 0}));
    EXPECT_EQ(partners.recent(), (std::deque<std::size_t>{0}));
}

/** @brief Tests that a long run leaves the capacity minus one most recent indices. */
TEST(Small_ImagingPartners, RecentSourceHoldsTheLastIndicesOfALongRun)
{
    const std::unique_ptr<RecentPartners> partners_owner =
        require_value(RecentPartners::make(1000, 2));
    RecentPartners& partners = *partners_owner;
    Random random(8u);
    ASSERT_EQ(partners.capacity(), 16u);

    for (std::size_t index = 0; index < 20; ++index)
        static_cast<void>(partners.choose(index, random));

    ASSERT_EQ(partners.recent().size(), 15u);
    EXPECT_EQ(partners.recent().front(), 5u);
    EXPECT_EQ(partners.recent().back(), 19u);
}

/** @brief Tests that each partner is the buffer entry one index draw names, in draw order. */
TEST(Small_ImagingPartners, RecentSourceDrawsThreeEntriesOfItsBuffer)
{
    const std::unique_ptr<RecentPartners> partners_owner =
        require_value(RecentPartners::make(1000, 16));
    RecentPartners& partners = *partners_owner;
    Random warm(2u);
    for (std::size_t index = 100; index < 140; ++index)
        static_cast<void>(partners.choose(index, warm));

    Random random(2026u);
    Random replica = random;
    const MosaicPartners chosen = partners.choose(500, random);

    const std::deque<std::size_t>& recent = partners.recent();
    ASSERT_EQ(recent.size(), 41u);
    ASSERT_EQ(recent.back(), 500u);
    const std::uint32_t count = static_cast<std::uint32_t>(recent.size());
    EXPECT_EQ(chosen[0], recent[replica.index(count)]);
    EXPECT_EQ(chosen[1], recent[replica.index(count)]);
    EXPECT_EQ(chosen[2], recent[replica.index(count)]);
    EXPECT_TRUE(random == replica) << "the source made a number of draws other than three";
}

/** @brief Tests that the recent source reaches every buffered index over many seeds. */
TEST(Small_ImagingPartners, RecentSourceCoversItsBuffer)
{
    const std::unique_ptr<RecentPartners> partners_owner =
        require_value(RecentPartners::make(100, 16));
    RecentPartners& partners = *partners_owner;
    Random warm(4u);
    for (std::size_t index = 10; index < 15; ++index)
        static_cast<void>(partners.choose(index, warm));

    std::array<bool, 100> seen{};
    for (std::uint64_t seed = 0; seed < 200; ++seed)
    {
        Random random(seed);
        for (const std::size_t partner : partners.choose(12, random))
        {
            ASSERT_GE(partner, 10u) << "seed " << seed;
            ASSERT_LT(partner, 15u) << "seed " << seed;
            seen[partner] = true;
        }
    }

    for (std::size_t index = 10; index < 15; ++index)
        EXPECT_TRUE(seen[index]) << "index " << index << " was never a partner";
}

/** @brief Tests that the recent source refuses a size, a batch or an index out of range. */
TEST(Small_ImagingPartners, RecentSourceRejectsSizesAndIndicesOutOfRange)
{
    const std::unique_ptr<RecentPartners> partners_owner =
        require_value(RecentPartners::make(4, 16));
    RecentPartners& partners = *partners_owner;
    Random random(1u);

    SA_EXPECT_FAILURE(RecentPartners::make(0, 16), StatusCode::InvalidDescription, "RecentPartners",
                      "dataset size of 0");
    SA_EXPECT_FAILURE(RecentPartners::make(10, 0), StatusCode::InvalidDescription, "RecentPartners",
                      "batch size of 0");
    SA_EXPECT_FAILURE(RecentPartners::make(10, -3), StatusCode::InvalidDescription,
                      "RecentPartners", "batch size of -3");
    SA_EXPECT_FATAL(static_cast<void>(partners.choose(4, random)), "RecentPartners::choose",
                    "index 4");
    EXPECT_TRUE(random == Random(1u)) << "a rejected call made a draw";
    EXPECT_TRUE(partners.recent().empty()) << "a rejected call changed the buffer";
}
