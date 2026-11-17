/**************************************************************************/
/* test_trailing_spec.cpp                                                 */
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
 * @file test_trailing_spec.cpp
 * @brief Tests that TrailingSpec says when it is plain and when two specs name the same work.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiAI/eager/storage_id.hpp>
#include <SushiAI/eager/trailing_spec.hpp>
#include <SushiAI/graph/in_memory_step_recorder.hpp>

using namespace SushiAI;
using Eager::TrailingBlend;
using Eager::TrailingCopy;
using Eager::TrailingSpec;
using Graph::InMemoryStepRecorder;

namespace
{
    /** @brief Stands for the recorder of the reference spec; only its address is read. */
    InMemoryStepRecorder g_first_recorder;

    /** @brief Stands for another caller's recorder; only its address is read. */
    InMemoryStepRecorder g_second_recorder;

    /** @brief Returns the spec every difference row is measured against. */
    TrailingSpec reference_spec()
    {
        TrailingSpec spec;
        spec.inner = &g_first_recorder;
        spec.copies = {TrailingCopy{1, 2}, TrailingCopy{3, 4}};
        spec.blends = {TrailingBlend{5, 6, 0.5, 2.0}};
        return spec;
    }

    /** @brief Pairs a name with the one change that makes a spec differ from the reference. */
    struct DifferenceRow
    {
        std::string name;              /**< The row's name. */
        void (*change)(TrailingSpec&); /**< The change applied to a copy of the reference. */
    };

    /** @brief Runs once per change a spec can show against the reference. */
    class Small_TrailingSpecDifference : public ::testing::TestWithParam<DifferenceRow>
    {

    };
} // namespace

/** @brief Tests that a spec with no copy and no blend is plain, whatever the caller's recorder. */
TEST(Small_TrailingSpec, ASpecWithNoCopyAndNoBlendIsPlainWhateverTheCallersRecorder)
{
    TrailingSpec bare;
    TrailingSpec recorded;
    recorded.inner = &g_first_recorder;

    EXPECT_TRUE(bare.is_plain());
    EXPECT_TRUE(recorded.is_plain());
}

/** @brief Tests that one copy is enough to make a spec not plain. */
TEST(Small_TrailingSpec, ASpecWithACopyIsNotPlain)
{
    TrailingSpec spec;
    spec.copies.push_back(TrailingCopy{1, 2});

    EXPECT_FALSE(spec.is_plain());
}

/** @brief Tests that one blend is enough to make a spec not plain. */
TEST(Small_TrailingSpec, ASpecWithABlendIsNotPlain)
{
    TrailingSpec spec;
    spec.blends.push_back(TrailingBlend{1, 2, 0.1, 1.0});

    EXPECT_FALSE(spec.is_plain());
}

/** @brief Tests that two specs built the same way compare equal. */
TEST(Small_TrailingSpec, TwoSpecsBuiltTheSameWayAreEqual)
{
    EXPECT_TRUE(reference_spec() == reference_spec());
}

/** @brief Tests that a blend names its target, source, momentum and scale in its equality. */
TEST(Small_TrailingSpec, TwoBlendsAreEqualOnlyWhenEveryFieldIsEqual)
{
    const TrailingBlend blend{1, 2, 0.5, 2.0};

    EXPECT_TRUE(blend == (TrailingBlend{1, 2, 0.5, 2.0}));
    EXPECT_FALSE(blend == (TrailingBlend{9, 2, 0.5, 2.0}));
    EXPECT_FALSE(blend == (TrailingBlend{1, 9, 0.5, 2.0}));
    EXPECT_FALSE(blend == (TrailingBlend{1, 2, 0.25, 2.0}));
    EXPECT_FALSE(blend == (TrailingBlend{1, 2, 0.5, 1.0}));
}

/** @brief Tests that a copy is equal to another only with the same source and destination. */
TEST(Small_TrailingSpec, TwoCopiesAreEqualOnlyWhenBothIdsAreEqual)
{
    const TrailingCopy copy{1, 2};

    EXPECT_TRUE(copy == (TrailingCopy{1, 2}));
    EXPECT_FALSE(copy == (TrailingCopy{2, 2}));
    EXPECT_FALSE(copy == (TrailingCopy{1, 1}));
}

/** @brief Tests that any one change of the work makes a spec differ from the reference. */
TEST_P(Small_TrailingSpecDifference, AnyChangedPartOfTheWorkMakesTwoSpecsDiffer)
{
    TrailingSpec changed = reference_spec();
    GetParam().change(changed);

    EXPECT_FALSE(changed == reference_spec());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_TrailingSpecDifference,
    ::testing::Values(
        DifferenceRow{"AnotherCallersRecorder",
                      [](TrailingSpec& spec) { spec.inner = &g_second_recorder; }},
        DifferenceRow{"NoCallersRecorder", [](TrailingSpec& spec) { spec.inner = nullptr; }},
        DifferenceRow{"ACopyLess", [](TrailingSpec& spec) { spec.copies.pop_back(); }},
        DifferenceRow{"ACopyMore",
                      [](TrailingSpec& spec) { spec.copies.push_back(TrailingCopy{7, 8}); }},
        DifferenceRow{"TheCopiesInAnotherOrder",
                      [](TrailingSpec& spec) { std::swap(spec.copies[0], spec.copies[1]); }},
        DifferenceRow{"AnotherBlendTarget",
                      [](TrailingSpec& spec) { spec.blends[0].target = 9; }},
        DifferenceRow{"ABlendLess", [](TrailingSpec& spec) { spec.blends.clear(); }}),
    [](const ::testing::TestParamInfo<DifferenceRow>& info) { return info.param.name; });
