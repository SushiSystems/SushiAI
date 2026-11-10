/**************************************************************************/
/* test_in_memory_partner_source.cpp                                      */
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
 * @file test_in_memory_partner_source.cpp
 * @brief Tests that InMemoryPartnerSource keeps the promise of the PartnerSource interface.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/in_memory_partner_source.hpp>
#include <SushiAI/vision/imaging/partner_source.hpp>
#include <SushiAI/vision/imaging/random.hpp>

using namespace SushiAI::Vision::Imaging;

namespace
{
    /** @brief Holds the seed of every generator below. */
    constexpr std::uint64_t K_SEED = 20261106;
} // namespace

/** @brief Tests that the source returns its script in turn, through the interface. */
TEST(Small_InMemoryPartnerSource, ReturnsTheScriptInTurn)
{
    std::unique_ptr<PartnerSource> source = std::make_unique<InMemoryPartnerSource>(
        std::vector<MosaicPartners>{MosaicPartners{1, 2, 3}, MosaicPartners{4, 5, 6}});
    Random random(K_SEED);

    EXPECT_EQ(source->choose(0, random), (MosaicPartners{1, 2, 3}));
    EXPECT_EQ(source->choose(1, random), (MosaicPartners{4, 5, 6}));
    EXPECT_EQ(source->choose(2, random), (MosaicPartners{1, 2, 3}));
}

/** @brief Tests that each call takes exactly three draws, as the interface promises. */
TEST(Small_InMemoryPartnerSource, EachCallTakesExactlyThreeDraws)
{
    InMemoryPartnerSource source({MosaicPartners{0, 0, 0}});
    Random random(K_SEED);
    Random expected(K_SEED);

    static_cast<void>(source.choose(0, random));
    for (std::size_t draw = 0; draw < K_MOSAIC_PARTNERS; ++draw)
        static_cast<void>(expected.next_u32());

    EXPECT_TRUE(random == expected);
}

/** @brief Tests that the journal keeps the sample index of each call. */
TEST(Small_InMemoryPartnerSource, TheJournalKeepsTheSampleIndexOfEachCall)
{
    InMemoryPartnerSource source({MosaicPartners{0, 0, 0}});
    Random random(K_SEED);

    static_cast<void>(source.choose(4, random));
    static_cast<void>(source.choose(9, random));

    const std::vector<std::size_t> expected = {4, 9};
    EXPECT_EQ(source.asked(), expected);
}

/** @brief Tests that a script with no entry stops the process. */
TEST(Medium_InMemoryPartnerSourceFatal, AnEmptyScriptStopsTheProcess)
{
    SA_EXPECT_FATAL(InMemoryPartnerSource(std::vector<MosaicPartners>{}),
                    "InMemoryPartnerSource", "no partners");
}
