/**************************************************************************/
/* test_block_rotary_positions.cpp                                        */
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
 * @file test_block_rotary_positions.cpp
 * @brief Tests the angles of rotary positions and the row of cosines and sines of a position.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/language/blocks/rotary_positions.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using Language::Blocks::rotary_angle;
using Language::Blocks::write_rotary_row;

namespace
{
    /** @brief Bounds a double against a hand-worked value, absolutely. */
    constexpr double K_TOLERANCE = 1e-12;

    /** @brief Bounds a float32 cosine or sine against the host's double, absolutely. */
    constexpr float K_ROW_TOLERANCE = 1e-6f;

    /** @brief Writes the row of @p position over @p pairs pairs into the two vectors. */
    void row_of(std::int64_t position, std::size_t pairs, std::vector<float>& cosines,
                std::vector<float>& sines)
    {
        cosines.assign(pairs, 0.0f);
        sines.assign(pairs, 0.0f);
        write_rotary_row(position, SushiRuntime::span<float>(cosines.data(), pairs),
                         SushiRuntime::span<float>(sines.data(), pairs));
    }
} // namespace

/** @brief Tests that pair i of n turns by position * 10000^(-i / n). */
TEST(Unit_BlockRotaryPositions, AnAngleIsThePositionTimesTheFrequencyOfThePair)
{
    EXPECT_DOUBLE_EQ(Language::Blocks::K_ROTARY_BASE, 10000.0);
    EXPECT_NEAR(rotary_angle(0, 1, 4), 0.0, K_TOLERANCE);
    EXPECT_NEAR(rotary_angle(7, 0, 4), 7.0, K_TOLERANCE);
    EXPECT_NEAR(rotary_angle(3, 1, 2), 0.03, K_TOLERANCE);
    EXPECT_NEAR(rotary_angle(5, 1, 4), 0.5, K_TOLERANCE);
    EXPECT_NEAR(rotary_angle(5, 2, 4), 0.05, K_TOLERANCE);
    EXPECT_NEAR(rotary_angle(5, 3, 4), 0.005, K_TOLERANCE);
}

/** @brief Tests that position 0 writes a cosine of 1 and a sine of 0 for every pair. */
TEST(Unit_BlockRotaryPositions, PositionZeroTurnsNothing)
{
    std::vector<float> cosines;
    std::vector<float> sines;
    row_of(0, 4, cosines, sines);

    for (std::size_t pair = 0; pair < cosines.size(); ++pair)
    {
        EXPECT_EQ(cosines[pair], 1.0f) << "pair " << pair;
        EXPECT_EQ(sines[pair], 0.0f) << "pair " << pair;
    }
}

/** @brief Tests a row against the cosine and the sine of each angle. */
TEST(Unit_BlockRotaryPositions, ARowHoldsTheCosineAndTheSineOfEachAngle)
{
    std::vector<float> cosines;
    std::vector<float> sines;
    row_of(11, 4, cosines, sines);

    for (std::size_t pair = 0; pair < cosines.size(); ++pair)
    {
        const double angle = rotary_angle(11, static_cast<std::int64_t>(pair), 4);
        EXPECT_NEAR(cosines[pair], static_cast<float>(std::cos(angle)), K_ROW_TOLERANCE);
        EXPECT_NEAR(sines[pair], static_cast<float>(std::sin(angle)), K_ROW_TOLERANCE);
    }

    // Pair 0 turns by 11 radians.
    EXPECT_NEAR(cosines[0], 0.0044257f, K_ROW_TOLERANCE);
    EXPECT_NEAR(sines[0], -0.9999902f, K_ROW_TOLERANCE);
}

/** @brief Tests that a position past the int32 range still writes a row of unit length. */
TEST(Unit_BlockRotaryPositions, ALargePositionWritesAPointOfTheUnitCircle)
{
    std::vector<float> cosines;
    std::vector<float> sines;
    row_of(std::int64_t{1} << 40, 4, cosines, sines);

    for (std::size_t pair = 0; pair < cosines.size(); ++pair)
    {
        EXPECT_NEAR(cosines[pair] * cosines[pair] + sines[pair] * sines[pair], 1.0f,
                    K_ROW_TOLERANCE)
            << "pair " << pair;
    }
}

/** @brief Tests that a row of no pair and two rows of different lengths are fatal. */
TEST(Unit_BlockRotaryPositionsDeathTest, ARowWithoutMatchingSpansIsFatal)
{
    std::vector<float> two(2, 0.0f);
    std::vector<float> three(3, 0.0f);

    SA_EXPECT_FATAL(write_rotary_row(1, SushiRuntime::span<float>(),
                                     SushiRuntime::span<float>()),
                    "0 cosines");
    SA_EXPECT_FATAL(write_rotary_row(1, SushiRuntime::span<float>(two.data(), two.size()),
                                     SushiRuntime::span<float>(three.data(), three.size())),
                    "2 cosines", "3 sines");
}
