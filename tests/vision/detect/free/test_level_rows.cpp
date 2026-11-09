/**************************************************************************/
/* test_level_rows.cpp                                                    */
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
 * @file test_level_rows.cpp
 * @brief Tests that joining a level leaves other rows alone and that splitting undoes a join.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/free/level_rows.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Vision::Detect::Free::join_level;
    using SushiAI::Vision::Detect::Free::LevelPlacement;
    using SushiAI::Vision::Detect::Free::split_level;

    /** @brief Holds the value a joined row starts at, which no join of these cases writes. */
    constexpr float K_UNTOUCHED = -7.0f;

    /** @brief Expects @p actual to hold @p expected, value for value, exactly. */
    void expect_values(const std::vector<float>& actual, const std::vector<float>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_EQ(actual[i], expected[i]) << "value " << i;
    }
} // namespace

/** @brief Tests that a join writes the level's rows and keeps every other row as it was. */
TEST(Small_DetectLevelRows, AJoinKeepsTheRowsOfOtherLevels)
{
    // One image, four anchors of one value; the level holds anchors 1 and 2.
    const LevelPlacement placement{1, 4, 1, 1, 2};
    const std::vector<float> level{5.0f, 6.0f};
    std::vector<float> joined(4, K_UNTOUCHED);

    SA_EXPECT_OK(join_level(level, placement, joined));

    expect_values(joined, {K_UNTOUCHED, 5.0f, 6.0f, K_UNTOUCHED});
}

/** @brief Tests that splitting a joined array returns the level that was joined into it. */
TEST(Small_DetectLevelRows, ASplitAfterAJoinReturnsTheLevel)
{
    // Two images, six anchors of three values; the level holds the last two anchors.
    const LevelPlacement placement{2, 6, 3, 4, 2};
    const std::vector<float> level{0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
                                   6.5f, 7.5f, 8.5f, 9.5f, 10.5f, 11.5f};
    std::vector<float> joined(2 * 6 * 3, K_UNTOUCHED);

    SA_EXPECT_OK(join_level(level, placement, joined));
    std::vector<float> returned(level.size(), K_UNTOUCHED);
    SA_EXPECT_OK(split_level(joined, placement, returned));

    expect_values(returned, level);
}

/** @brief Tests that a split reads the rows of the level at its anchor offset in every image. */
TEST(Small_DetectLevelRows, ASplitReadsTheLevelAtItsAnchorOffset)
{
    // Two images, five anchors of two values; the level holds anchors 1 to 3.
    const LevelPlacement placement{2, 5, 2, 1, 3};
    const std::vector<float> joined{0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                                    8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f, 19.0f};
    std::vector<float> level(12, K_UNTOUCHED);

    SA_EXPECT_OK(split_level(joined, placement, level));

    expect_values(level, {2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                          12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f});
}

/** @brief Tests that a level reaching past the last anchor is refused. */
TEST(Small_DetectLevelRows, AJoinRefusesALevelPastTheLastAnchor)
{
    const LevelPlacement placement{1, 5, 2, 4, 2};
    const std::vector<float> level(4, 0.0f);
    std::vector<float> joined(10, 0.0f);

    SA_EXPECT_FAILURE(join_level(level, placement, joined), StatusCode::InvalidDescription,
                      "2 anchors", "offset 4", "5 anchors");
}

/** @brief Tests that splitting into a level array of the wrong size is refused with both sizes. */
TEST(Small_DetectLevelRows, ASplitRefusesALevelArrayOfTheWrongSize)
{
    // Two images of three anchors and two values need 12 level values out of 20 joined ones.
    const LevelPlacement placement{2, 5, 2, 1, 3};
    const std::vector<float> joined(20, 0.0f);
    std::vector<float> level(10);

    SA_EXPECT_FAILURE(split_level(joined, placement, level), StatusCode::InvalidDescription,
                      "12 level values", "20 joined values", "got 10 and 20");
}
