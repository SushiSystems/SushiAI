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
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/free/level_rows.hpp>

namespace
{
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
    std::vector<float> level(2 * 2 * 3);
    for (std::size_t i = 0; i < level.size(); ++i)
        level[i] = static_cast<float>(i) + 0.5f;
    std::vector<float> joined(2 * 6 * 3, K_UNTOUCHED);

    SA_EXPECT_OK(join_level(level, placement, joined));
    std::vector<float> returned(level.size(), K_UNTOUCHED);
    SA_EXPECT_OK(split_level(joined, placement, returned));

    expect_values(returned, level);
}
