/**************************************************************************/
/* test_partner_source.cpp                                                */
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
 * @file test_partner_source.cpp
 * @brief Tests the contract every PartnerSource keeps, asked through the interface.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/dataset_partners.hpp>
#include <SushiAI/vision/imaging/in_memory_partner_source.hpp>
#include <SushiAI/vision/imaging/partner_source.hpp>
#include <SushiAI/vision/imaging/random.hpp>
#include <SushiAI/vision/imaging/recent_partners.hpp>

using namespace SushiAI::Vision::Imaging;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Holds the seed of every generator below. */
    constexpr std::uint64_t K_SEED = 20261016;

    /** @brief Holds the image count of the datasets the real sources are made for. */
    constexpr std::size_t K_DATASET_SIZE = 10;

    /** @brief Makes the in-memory source of the contract. */
    struct InMemoryMaker
    {
        /** @brief Returns a source that replays one scripted entry. */
        static std::unique_ptr<PartnerSource> make()
        {
            return std::make_unique<InMemoryPartnerSource>(
                std::vector<MosaicPartners>{MosaicPartners{0, 0, 0}});
        }
    };

    /** @brief Makes the dataset source of the contract. */
    struct DatasetMaker
    {
        /** @brief Returns a source that draws from a dataset of K_DATASET_SIZE images. */
        static std::unique_ptr<PartnerSource> make()
        {
            return require_value(DatasetPartners::make(K_DATASET_SIZE));
        }
    };

    /** @brief Makes the recent-indices source of the contract. */
    struct RecentMaker
    {
        /** @brief Returns a source over a dataset of K_DATASET_SIZE images and a batch of 4. */
        static std::unique_ptr<PartnerSource> make()
        {
            return require_value(RecentPartners::make(K_DATASET_SIZE, 4));
        }
    };

    /** @brief Runs the PartnerSource contract once per implementation. */
    template <typename Maker>
    class Small_ImagingPartnerSourceContract : public ::testing::Test
    {
    };

    using Sources = ::testing::Types<InMemoryMaker, DatasetMaker, RecentMaker>;

    /** @brief Names each implementation of the contract by its maker. */
    class SourceNames
    {
        public:
            /** @brief Returns the name of the implementation at an index of Sources. */
            template <typename Maker>
            static std::string GetName(int)
            {
                if (std::is_same_v<Maker, InMemoryMaker>)
                    return "InMemory";
                if (std::is_same_v<Maker, DatasetMaker>)
                    return "Dataset";
                return "Recent";
            }
    };

    TYPED_TEST_SUITE(Small_ImagingPartnerSourceContract, Sources, SourceNames);
} // namespace

/** @brief Tests that a source asked through the interface makes exactly three draws. */
TYPED_TEST(Small_ImagingPartnerSourceContract, ASourceAskedThroughTheInterfaceMakesThreeDraws)
{
    const std::unique_ptr<PartnerSource> source = TypeParam::make();
    Random random(K_SEED);
    Random expected(K_SEED);

    static_cast<void>(source->choose(4, random));
    for (std::size_t draw = 0; draw < K_MOSAIC_PARTNERS; ++draw)
        static_cast<void>(expected.next_u32());

    EXPECT_TRUE(random == expected);
}
