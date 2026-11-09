/**************************************************************************/
/* test_seeded_sequence.cpp                                               */
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
 * @file test_seeded_sequence.cpp
 * @brief Tests that the seeded sequence follows SplitMix64, repeats per seed and draws once.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/anchor/seeded_sequence.hpp>
#include <support/splitmix64.hpp>

namespace
{
    using SushiAI::Vision::Detect::Anchor::Detail::SeededSequence;
    using TestSupport::reference_splitmix64;

    /** @brief Holds the seed the cases start from. */
    constexpr std::uint64_t K_SEED = 20261016;

    /** @brief Holds the number of numbers a case compares. */
    constexpr std::size_t K_NUMBERS = 8;

    /** @brief Holds the number of unit draws the interval case makes. */
    constexpr std::size_t K_UNIT_DRAWS = 1000;

    /** @brief Holds 2^53, the divisor of a unit draw. */
    constexpr double K_UNIT_DIVISOR = 9007199254740992.0;

    /** @brief Returns the first K_NUMBERS numbers of the sequence of @p seed. */
    std::array<std::uint64_t, K_NUMBERS> draw_numbers(std::uint64_t seed)
    {
        SeededSequence sequence(seed);
        std::array<std::uint64_t, K_NUMBERS> numbers{};
        for (std::uint64_t& number : numbers)
            number = sequence.next();
        return numbers;
    }

    /** @brief Returns the first K_NUMBERS numbers of the published sequence of @p seed. */
    std::array<std::uint64_t, K_NUMBERS> reference_numbers(std::uint64_t seed)
    {
        std::array<std::uint64_t, K_NUMBERS> numbers{};
        for (std::size_t index = 0; index < K_NUMBERS; ++index)
            numbers[index] = reference_splitmix64(seed, index);
        return numbers;
    }

    /** @brief Returns the first K_UNIT_DRAWS unit draws of the sequence of @p seed. */
    std::array<double, K_UNIT_DRAWS> draw_units(std::uint64_t seed)
    {
        SeededSequence sequence(seed);
        std::array<double, K_UNIT_DRAWS> units{};
        for (double& unit : units)
            unit = sequence.next_unit();
        return units;
    }
} // namespace

/** @brief Tests the first eight numbers of seed 0 against the published SplitMix64 sequence. */
TEST(Small_DetectSeededSequence, SeedZeroMatchesThePublishedNumbers)
{
    const std::array<std::uint64_t, K_NUMBERS> published = {
        0xE220A8397B1DCDAFull, 0x6E789E6AA1B965F4ull, 0x06C45D188009454Full,
        0xF88BB8A8724C81ECull, 0x1B39896A51A8749Bull, 0x53CB9F0C747EA2EAull,
        0x2C829ABE1F4532E1ull, 0xC584133AC916AB3Cull};

    EXPECT_EQ(draw_numbers(0), published);
}

/** @brief Tests that a seed other than 0 follows the reference sequence of that seed. */
TEST(Small_DetectSeededSequence, AnotherSeedMatchesTheReferenceSequence)
{
    EXPECT_EQ(draw_numbers(K_SEED), reference_numbers(K_SEED));
}

/** @brief Tests that two sequences of one seed yield the same numbers in the same order. */
TEST(Small_DetectSeededSequence, TwoSequencesOfOneSeedYieldTheSameNumbers)
{
    EXPECT_EQ(draw_numbers(K_SEED), draw_numbers(K_SEED));
}

/** @brief Tests that neighbouring seeds start with different numbers. */
TEST(Small_DetectSeededSequence, NeighbouringSeedsStartWithDifferentNumbers)
{
    SeededSequence first(K_SEED);
    SeededSequence second(K_SEED + 1);

    EXPECT_NE(first.next(), second.next());
}

/** @brief Tests that a unit draw takes exactly one number of the sequence. */
TEST(Small_DetectSeededSequence, AUnitDrawTakesOneNumber)
{
    SeededSequence drawn(K_SEED);
    SeededSequence counted(K_SEED);

    static_cast<void>(drawn.next_unit());
    static_cast<void>(counted.next());

    EXPECT_EQ(drawn.next(), counted.next());
}

/** @brief Tests that the unit draw is the upper 53 bits of the next number over 2^53. */
TEST(Small_DetectSeededSequence, AUnitDrawIsTheUpperBitsOfTheNextNumber)
{
    SeededSequence sequence(0);

    const double unit = sequence.next_unit();

    // 53 bits over 2^53 is exact in double, so the draw equals the quotient.
    EXPECT_EQ(unit, static_cast<double>(reference_splitmix64(0, 0) >> 11) / K_UNIT_DIVISOR);
}

/** @brief Tests that many unit draws all lie in [0, 1). */
TEST(Small_DetectSeededSequence, EveryUnitDrawLiesInTheHalfOpenUnitInterval)
{
    const std::array<double, K_UNIT_DRAWS> units = draw_units(K_SEED);

    EXPECT_THAT(units, ::testing::Each(::testing::AllOf(::testing::Ge(0.0), ::testing::Lt(1.0))));
}
