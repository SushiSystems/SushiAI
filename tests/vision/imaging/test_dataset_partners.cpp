/**************************************************************************/
/* test_dataset_partners.cpp                                              */
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
 * @file test_dataset_partners.cpp
 * @brief Tests the mosaic partner source drawn from the whole dataset.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <limits>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/vision/imaging/dataset_partners.hpp>
#include <SushiAI/vision/imaging/recent_partners.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

/** @brief Tests that the dataset source returns three index draws in draw order. */
TEST(Small_ImagingPartners, DatasetSourceReturnsThreeIndexDrawsInOrder)
{
    DatasetPartners partners(1000);
    Random random(20261007u);
    Random replica = random;

    const MosaicPartners chosen = partners.choose(0, random);

    EXPECT_EQ(chosen[0], replica.index(1000u));
    EXPECT_EQ(chosen[1], replica.index(1000u));
    EXPECT_EQ(chosen[2], replica.index(1000u));
    EXPECT_TRUE(random == replica) << "the source made a number of draws other than three";
}

/** @brief Tests that the dataset source reaches every image and none outside the dataset. */
TEST(Small_ImagingPartners, DatasetSourceCoversTheDatasetAndNeverLeavesIt)
{
    const std::size_t dataset_size = 7;
    DatasetPartners partners(dataset_size);
    std::array<bool, 7> seen{};

    for (std::uint64_t seed = 0; seed < 300; ++seed)
    {
        Random random(seed);
        for (const std::size_t partner : partners.choose(0, random))
        {
            ASSERT_LT(partner, dataset_size) << "seed " << seed;
            seen[partner] = true;
        }
    }

    for (std::size_t i = 0; i < dataset_size; ++i)
        EXPECT_TRUE(seen[i]) << "index " << i << " was never a partner";
}

/** @brief Tests that earlier calls do not change what the dataset source gives a generator. */
TEST(Small_ImagingPartners, DatasetSourceKeepsNoStateBetweenSamples)
{
    DatasetPartners used(50);
    DatasetPartners fresh(50);
    Random warm(9u);
    Random first(77u);
    Random second(77u);

    static_cast<void>(used.choose(3, warm));
    static_cast<void>(used.choose(8, warm));

    EXPECT_EQ(used.choose(5, first), fresh.choose(5, second));
}

/** @brief Tests that the dataset source refuses a size or an index outside its range. */
TEST(Small_ImagingPartners, DatasetSourceRejectsSizesAndIndicesOutOfRange)
{
    const std::size_t largest = std::numeric_limits<std::uint32_t>::max();
    DatasetPartners partners(4);
    Random random(1u);

    SA_EXPECT_ERROR(DatasetPartners(0), "DatasetPartners", "dataset size of 0");
    SA_EXPECT_ERROR(DatasetPartners(largest + 1), "DatasetPartners", "dataset size");
    EXPECT_NO_THROW(static_cast<void>(DatasetPartners(largest)));
    SA_EXPECT_ERROR(static_cast<void>(partners.choose(4, random)), "DatasetPartners::choose",
                    "index 4");
    EXPECT_TRUE(random == Random(1u)) << "a rejected call made a draw";
}
