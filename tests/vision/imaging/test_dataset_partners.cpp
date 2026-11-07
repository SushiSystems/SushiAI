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
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/dataset_partners.hpp>
#include <SushiAI/vision/imaging/recent_partners.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;
using SushiAI::Test::require_value;

/** @brief Tests that the dataset source returns three index draws in draw order. */
TEST(Small_ImagingPartners, DatasetSourceReturnsThreeIndexDrawsInOrder)
{
    const std::unique_ptr<DatasetPartners> partners_owner =
        require_value(DatasetPartners::make(1000));
    DatasetPartners& partners = *partners_owner;
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
    constexpr std::size_t dataset_size = 7;
    constexpr std::size_t sample_count = 64;
    const std::unique_ptr<DatasetPartners> partners_owner =
        require_value(DatasetPartners::make(dataset_size));
    DatasetPartners& partners = *partners_owner;
    std::array<bool, dataset_size> seen{};
    Random random(20261007u);

    for (std::size_t sample = 0; sample < sample_count; ++sample)
    {
        for (const std::size_t partner : partners.choose(0, random))
        {
            ASSERT_LT(partner, dataset_size) << "sample " << sample;
            seen[partner] = true;
        }
    }

    for (std::size_t i = 0; i < dataset_size; ++i)
        EXPECT_TRUE(seen[i]) << "index " << i << " was never a partner";
}

/** @brief Tests that earlier calls do not change what the dataset source gives a generator. */
TEST(Small_ImagingPartners, DatasetSourceKeepsNoStateBetweenSamples)
{
    const std::unique_ptr<DatasetPartners> used_owner = require_value(DatasetPartners::make(50));
    DatasetPartners& used = *used_owner;
    const std::unique_ptr<DatasetPartners> fresh_owner = require_value(DatasetPartners::make(50));
    DatasetPartners& fresh = *fresh_owner;
    Random warm(9u);
    Random first(77u);
    Random second(77u);

    static_cast<void>(used.choose(3, warm));
    static_cast<void>(used.choose(8, warm));

    EXPECT_EQ(used.choose(5, first), fresh.choose(5, second));
}

/** @brief Tests that the dataset source refuses a size of zero and one above the uint32 range. */
TEST(Small_ImagingPartners, DatasetSourceRejectsSizesOutsideItsRange)
{
    const std::size_t largest = std::numeric_limits<std::uint32_t>::max();

    SA_EXPECT_FAILURE(DatasetPartners::make(0), StatusCode::InvalidDescription, "DatasetPartners",
                      "dataset size of 0");
    SA_EXPECT_FAILURE(DatasetPartners::make(largest + 1), StatusCode::InvalidDescription,
                      "DatasetPartners", "dataset size");
}

/** @brief Tests that the dataset source accepts the largest size the uint32 range holds. */
TEST(Small_ImagingPartners, DatasetSourceAcceptsTheLargestSize)
{
    const std::size_t largest = std::numeric_limits<std::uint32_t>::max();

    SA_EXPECT_OK(DatasetPartners::make(largest));
}

/** @brief Tests that the dataset source stops on a sample index outside the dataset. */
TEST(Medium_ImagingPartners, DatasetSourceStopsOnASampleIndexOutOfRange)
{
    const std::unique_ptr<DatasetPartners> partners_owner = require_value(DatasetPartners::make(4));
    DatasetPartners& partners = *partners_owner;
    Random random(1u);

    SA_EXPECT_FATAL(static_cast<void>(partners.choose(4, random)), "DatasetPartners::choose",
                    "index 4");
}
