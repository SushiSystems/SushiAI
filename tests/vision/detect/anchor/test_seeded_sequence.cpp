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
 * @brief Tests that the seeded sequence repeats per seed, differs between seeds and draws once.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/anchor/seeded_sequence.hpp>

namespace
{
    using SushiAI::Vision::Detect::Anchor::Detail::SeededSequence;

    /** @brief Holds the seed the cases start from. */
    constexpr std::uint64_t K_SEED = 20261016;

    /** @brief Holds the number of draws a case compares. */
    constexpr int K_DRAWS = 1000;
} // namespace

/** @brief Tests that two sequences of one seed yield the same numbers in the same order. */
TEST(Small_DetectSeededSequence, TwoSequencesOfOneSeedYieldTheSameNumbers)
{
    SeededSequence first(K_SEED);
    SeededSequence second(K_SEED);

    for (int draw = 0; draw < K_DRAWS; ++draw)
        EXPECT_EQ(first.next(), second.next()) << "draw " << draw;
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

/** @brief Tests that many unit draws all lie in [0, 1). */
TEST(Small_DetectSeededSequence, EveryUnitDrawLiesInTheHalfOpenUnitInterval)
{
    SeededSequence sequence(K_SEED);

    for (int draw = 0; draw < K_DRAWS; ++draw)
    {
        const double unit = sequence.next_unit();
        EXPECT_GE(unit, 0.0) << "draw " << draw;
        EXPECT_LT(unit, 1.0) << "draw " << draw;
    }
}
