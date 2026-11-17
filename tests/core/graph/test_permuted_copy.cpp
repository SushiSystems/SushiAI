/**************************************************************************/
/* test_permuted_copy.cpp                                                 */
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
 * @file test_permuted_copy.cpp
 * @brief Tests the axis permutation checks and the copy plan of permuted_copy.hpp.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/permuted_copy.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>

using namespace SushiAI;
using Graph::CopyAxis;
using Graph::PermutedCopy;

namespace
{
    /** @brief Wraps @p values as the span the header takes. */
    SushiRuntime::span<const std::int64_t> as_span(const std::vector<std::int64_t>& values)
    {
        return SushiRuntime::span<const std::int64_t>(values.data(), values.size());
    }

    /** @brief Expects @p axis to walk @p extent elements @p stride apart. */
    void expect_axis(const CopyAxis& axis, std::int64_t extent, std::int64_t stride,
                     std::size_t position)
    {
        EXPECT_EQ(axis.extent, extent) << "axis " << position;
        EXPECT_EQ(axis.stride, stride) << "axis " << position;
    }

    /** @brief Holds one list and whether it is a permutation of its own length. */
    struct PermutationRow
    {
        std::string name;                   /**< The row's name. */
        std::vector<std::int64_t> axes;     /**< The list handed to is_axis_permutation. */
        bool valid;                         /**< Whether it is a permutation. */
    };

    /** @brief Runs once per list the check judges. */
    class Small_AxisPermutationRows : public ::testing::TestWithParam<PermutationRow>
    {

    };

    /** @brief Holds one operand, one permutation and the four axes the plan must hold. */
    struct PlanRow
    {
        std::string name;                      /**< The row's name. */
        Shape source;                          /**< The operand shape, batch axis first. */
        std::vector<std::int64_t> permutation; /**< The permutation of the non-batch axes. */
        std::array<std::array<std::int64_t, 2>, Graph::K_COPY_AXES> axes; /**< Extent, stride. */
    };

    /** @brief Runs once per plan the header derives. */
    class Small_PermutedCopyPlanRows : public ::testing::TestWithParam<PlanRow>
    {

    };
} // namespace

/** @brief Tests which lists is_axis_permutation accepts. */
TEST_P(Small_AxisPermutationRows, OnlyAListThatHoldsEachAxisOnceIsAPermutation)
{
    const PermutationRow& row = GetParam();

    EXPECT_EQ(Graph::is_axis_permutation(as_span(row.axes)), row.valid);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_AxisPermutationRows,
    ::testing::Values(
        PermutationRow{"Empty", {}, true}, PermutationRow{"Identity", {0, 1, 2}, true},
        PermutationRow{"Swap", {1, 0}, true}, PermutationRow{"Rotation", {2, 0, 1}, true},
        PermutationRow{"RepeatedAxis", {0, 0}, false},
        PermutationRow{"AxisBeyondTheLength", {0, 2}, false},
        PermutationRow{"NegativeAxis", {-1, 0}, false}),
    [](const ::testing::TestParamInfo<PermutationRow>& info) { return info.param.name; });

/** @brief Tests that a list as long as the rank limit is refused, since the batch axis is left out. */
TEST(Small_AxisPermutation, AListAsLongAsTheRankLimitIsRefused)
{
    std::vector<std::int64_t> axes;
    for (std::int64_t axis = 0; axis < K_MAX_RANK; ++axis)
        axes.push_back(axis);

    EXPECT_FALSE(Graph::is_axis_permutation(as_span(axes)));

    axes.pop_back();

    EXPECT_TRUE(Graph::is_axis_permutation(as_span(axes)));
}

/** @brief Tests that applying a permutation and then its inverse puts every axis back. */
TEST(Small_AxisPermutation, TheInverseUndoesThePermutation)
{
    const std::vector<std::vector<std::int64_t>> permutations = {
        {1, 0}, {2, 0, 1}, {1, 2, 0}, {3, 1, 0, 2}};

    for (const std::vector<std::int64_t>& permutation : permutations)
    {
        const auto inverse = Graph::inverse_axis_permutation(as_span(permutation));

        for (std::size_t axis = 0; axis < permutation.size(); ++axis)
        {
            EXPECT_EQ(permutation[static_cast<std::size_t>(inverse[axis])],
                      static_cast<std::int64_t>(axis));
        }
    }
}

/** @brief Tests that the inverse of a rotation of three axes is the rotation the other way. */
TEST(Small_AxisPermutation, TheInverseOfARotationIsTheRotationTheOtherWay)
{
    const std::vector<std::int64_t> rotation = {2, 0, 1};

    const auto inverse = Graph::inverse_axis_permutation(as_span(rotation));

    EXPECT_EQ(inverse[0], 1);
    EXPECT_EQ(inverse[1], 2);
    EXPECT_EQ(inverse[2], 0);
}

/** @brief Tests that permutation_of reads slot k of the node as entry k. */
TEST(Small_AxisPermutation, ANodeHoldsThePermutationInItsFirstSlots)
{
    Graph::Node node;
    node.attributes.values[0] = 2.0;
    node.attributes.values[1] = 0.0;
    node.attributes.values[2] = 1.0;

    const auto permutation = Graph::permutation_of(node, 3);

    EXPECT_EQ(permutation[0], 2);
    EXPECT_EQ(permutation[1], 0);
    EXPECT_EQ(permutation[2], 1);
}

/** @brief Tests the axes plan_permuted_copy derives for hand-worked operands. */
TEST_P(Small_PermutedCopyPlanRows, TheAxesAreTheRunsTheResultOrderWalks)
{
    const PlanRow& row = GetParam();

    const PermutedCopy plan = Graph::plan_permuted_copy(row.source, as_span(row.permutation));

    ASSERT_TRUE(plan.fits);
    for (std::size_t position = 0; position < Graph::K_COPY_AXES; ++position)
        expect_axis(plan.axes[position], row.axes[position][0], row.axes[position][1], position);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_PermutedCopyPlanRows,
    ::testing::Values(
        // Source [2, 3, 4] has the strides 12, 4 and 1; the swap walks batch, stride 1, stride 4.
        PlanRow{"SwapOfTwoAxes", Shape{2, 3, 4}, {1, 0},
                {{{1, 24}, {2, 12}, {4, 1}, {3, 4}}}},
        // The identity joins the batch and both axes into one run of 24 elements.
        PlanRow{"IdentityJoinsEveryAxis", Shape{2, 3, 4}, {0, 1},
                {{{1, 24}, {1, 24}, {1, 24}, {24, 1}}}},
        // An axis of extent 1 is dropped, so the batch and the axis of 5 join: 10 elements.
        PlanRow{"UnitAxisIsDropped", Shape{2, 1, 5}, {1, 0}, {{{1, 10}, {1, 10}, {1, 10}, {10, 1}}}},
        // No axis has an extent above 1: the copy is one element.
        PlanRow{"OneElement", Shape{1, 1, 1}, {1, 0}, {{{1, 1}, {1, 1}, {1, 1}, {1, 1}}}}),
    [](const ::testing::TestParamInfo<PlanRow>& info) { return info.param.name; });

/** @brief Tests that a permutation that cannot join its axes into four runs does not fit. */
TEST(Small_PermutedCopyPlan, AReversalOfFiveAxesDoesNotFitInFourRuns)
{
    // Batch 2 and the axes 3, 4, 5, 6, 7 reversed leave six runs, none of which join.
    const Shape source{2, 3, 4, 5, 6, 7};
    const std::vector<std::int64_t> reversal = {4, 3, 2, 1, 0};

    const PermutedCopy plan = Graph::plan_permuted_copy(source, as_span(reversal));

    EXPECT_FALSE(plan.fits);
}
