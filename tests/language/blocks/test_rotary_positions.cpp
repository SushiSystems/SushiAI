/**************************************************************************/
/* test_rotary_positions.cpp                                              */
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
 * @file test_rotary_positions.cpp
 * @brief Tests the angles of rotary positions and the row of cosines and sines of a position.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/blocks/rotary_positions.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Language::Blocks::check_rotary_row_spans;
using Language::Blocks::rotary_angle;
using Language::Blocks::write_rotary_row;

namespace
{
    /**
     * @brief Bounds a double against a hand-worked value, absolutely: 16384 times the double unit
     *        roundoff, the smallest power of two not tighter than 1e-12.
     */
    constexpr double K_TOLERANCE = 16384.0 * TestSupport::unit_roundoff<double>();

    /**
     * @brief Bounds a float32 cosine or sine against a hand-worked value, absolutely: 32 times the
     *        float unit roundoff, the smallest power of two not tighter than 1e-6f.
     */
    constexpr float K_ROW_TOLERANCE =
        static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());

    /** @brief Holds the cosines and the sines of one row. */
    struct Row
    {
        std::vector<float> cosines; /**< One cosine per pair. */
        std::vector<float> sines;   /**< One sine per pair. */
    };

    /** @brief Returns the row of @p position over @p pairs pairs. */
    Row row_of(std::int64_t position, std::size_t pairs)
    {
        Row row{std::vector<float>(pairs, 0.0f), std::vector<float>(pairs, 0.0f)};
        write_rotary_row(position, SushiRuntime::span<float>(row.cosines.data(), pairs),
                         SushiRuntime::span<float>(row.sines.data(), pairs));
        return row;
    }

    /** @brief Names the angle of one pair, worked by hand from position * 10000^(-pair / pairs). */
    struct AngleCase
    {
        std::int64_t position; /**< The position the pair turns at. */
        std::int64_t pair;     /**< The pair, in [0, pairs). */
        std::int64_t pairs;    /**< The pairs of the group. */
        double angle;          /**< The angle in radians; 10000^(-1/4) is 0.1 and 10000^(-1/2) 0.01. */
    };

    /** @brief Returns the hand-worked angles. */
    std::vector<AngleCase> angle_cases()
    {
        return {{0, 1, 4, 0.0},   {7, 0, 4, 7.0},   {3, 1, 2, 0.03},
                {5, 1, 4, 0.5},   {5, 2, 4, 0.05},  {5, 3, 4, 0.005}};
    }

    /** @brief Provides the rows of the angle cases. */
    class Small_BlockRotaryAngle : public ::testing::TestWithParam<AngleCase>
    {

    };

    /** @brief Provides the pair counts of the position-zero and unit-circle cases. */
    class Small_BlockRotaryRow : public ::testing::TestWithParam<std::size_t>
    {

    };

    /** @brief Provides the pair of the unit-circle case. */
    class Small_BlockRotaryUnitCircle : public ::testing::TestWithParam<std::size_t>
    {

    };

    /** @brief Names the lengths of two spans and whether the check accepts them. */
    struct SpanCase
    {
        std::size_t cosines; /**< The length of the cosine span. */
        std::size_t sines;   /**< The length of the sine span. */
    };

    /** @brief Provides the lengths the check refuses. */
    class Small_BlockRotarySpanRefusal : public ::testing::TestWithParam<SpanCase>
    {

    };

    /** @brief Runs the process-stopping case in a child process. */
    class Medium_BlockRotaryPositionsDeath : public ::testing::Test
    {

    };
} // namespace

/** @brief Tests that pair i of n turns by position * 10000^(-i / n). */
TEST_P(Small_BlockRotaryAngle, AnAngleIsThePositionTimesTheFrequencyOfThePair)
{
    const AngleCase& row = GetParam();

    EXPECT_NEAR(rotary_angle(row.position, row.pair, row.pairs), row.angle, K_TOLERANCE);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_BlockRotaryAngle, ::testing::ValuesIn(angle_cases()));

/** @brief Tests that position 0 writes a cosine of 1 and a sine of 0 for every pair. */
TEST_P(Small_BlockRotaryRow, PositionZeroTurnsNothing)
{
    const std::size_t pairs = GetParam();

    const Row row = row_of(0, pairs);

    EXPECT_EQ(row.cosines, std::vector<float>(pairs, 1.0f));
    EXPECT_EQ(row.sines, std::vector<float>(pairs, 0.0f));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_BlockRotaryRow, ::testing::Values(1u, 2u, 4u));

/**
 * @brief Tests the first pair of position 11 against the values worked by hand.
 *
 * Pair 0 turns by 11 radians: cos 11 = 0.0044257 and sin 11 = -0.9999902.
 */
TEST(Small_BlockRotaryPositions, ARowHoldsTheCosineAndTheSineOfEachAngle)
{
    const Row row = row_of(11, 4);

    EXPECT_NEAR(row.cosines[0], 0.0044257f, K_ROW_TOLERANCE);
    EXPECT_NEAR(row.sines[0], -0.9999902f, K_ROW_TOLERANCE);
}

/** @brief Tests that a position past the int32 range still writes a row of unit length. */
TEST_P(Small_BlockRotaryUnitCircle, ALargePositionWritesAPointOfTheUnitCircle)
{
    const std::size_t pair = GetParam();

    const Row row = row_of(std::int64_t{1} << 40, 4);

    EXPECT_NEAR(row.cosines[pair] * row.cosines[pair] + row.sines[pair] * row.sines[pair], 1.0f,
                K_ROW_TOLERANCE);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_BlockRotaryUnitCircle,
                         ::testing::Values(0u, 1u, 2u, 3u));

/** @brief Tests that spans of one nonzero length are accepted. */
TEST(Small_BlockRotaryPositions, SpansOfOneNonzeroLengthAreAccepted)
{
    SA_EXPECT_OK(check_rotary_row_spans(3, 3));
}

/** @brief Tests that an empty cosine span and spans of different lengths are refused. */
TEST_P(Small_BlockRotarySpanRefusal, SpansThatDoNotMatchAreRefused)
{
    const SpanCase& row = GetParam();

    SA_EXPECT_FAILURE(check_rotary_row_spans(row.cosines, row.sines),
                      StatusCode::InvalidDescription, std::to_string(row.cosines) + " cosines",
                      std::to_string(row.sines) + " sines");
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_BlockRotarySpanRefusal,
                         ::testing::Values(SpanCase{0, 0}, SpanCase{2, 3}));

/** @brief Tests that write_rotary_row stops the process on the refusal of the span check. */
TEST_F(Medium_BlockRotaryPositionsDeath, ARowWithoutMatchingSpansIsFatal)
{
    std::vector<float> two(2, 0.0f);
    std::vector<float> three(3, 0.0f);

    SA_EXPECT_FATAL(write_rotary_row(1, SushiRuntime::span<float>(two.data(), two.size()),
                                     SushiRuntime::span<float>(three.data(), three.size())),
                    "2 cosines", "3 sines");
}
