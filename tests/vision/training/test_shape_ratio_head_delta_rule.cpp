/**************************************************************************/
/* test_shape_ratio_head_delta_rule.cpp                                   */
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
 * @file test_shape_ratio_head_delta_rule.cpp
 * @brief Tests the shape-ratio head delta rule against its assignment, loss and delta.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_assignment.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_delta.hpp>
#include <SushiAI/vision/detect/anchor/target_assignment.hpp>
#include <SushiAI/vision/training/anchor_head.hpp>
#include <SushiAI/vision/training/shape_ratio_head_delta_rule.hpp>

#include "head_delta_rule_fixture.hpp"
#include <support/tolerance.hpp>

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::HeadDeltaRuleTest;
    using Vision::Detect::Anchor::AssignmentInput;
    using Vision::Detect::Anchor::build_objectness_targets;
    using Vision::Detect::Anchor::compute_shape_ratio_delta;
    using Vision::Detect::Anchor::derive_shape_ratio_factors;
    using Vision::Detect::Anchor::HeadCoding;
    using Vision::Detect::Anchor::HeadCost;
    using Vision::Detect::Anchor::HeadDelta;
    using Vision::Detect::Anchor::measure_shape_ratio_loss;
    using Vision::Detect::Anchor::NetworkSize;
    using Vision::Detect::Anchor::ShapeRatioAssignment;
    using Vision::Detect::Anchor::ShapeRatioDeltaInput;
    using Vision::Detect::Anchor::ShapeRatioLoss;
    using Vision::Detect::Anchor::ShapeRatioOptions;
    using Vision::Detect::Anchor::SizeForm;
    using Vision::Detect::Anchor::TargetAssignment;
    using Vision::Training::AnchorHead;
    using Vision::Training::ShapeRatioHeadDeltaRule;

    /**
     * @brief Holds the absolute the tolerance of a sum of doubles taken in two orders: 16384 times
     *        the double unit roundoff, the smallest power of two not tighter than the earlier
     *        1e-12.
     */
    constexpr double K_SUM_TOLERANCE = 16384.0 * TestSupport::unit_roundoff<double>();

    /** @brief Holds what the pieces of the shape-ratio path give one level of the fixture. */
    struct LevelParts
    {
        HeadDelta alone;            /**< The delta of the level, computed without the rule. */
        double total = 0.0;         /**< The level's loss total. */
        double box = 0.0;           /**< The level's box share, scaled to the batch. */
    };

    /**
     * @brief Runs the assignment, the loss and the delta on level @p level of @p fixture, one
     *        after the other, as the rule does for that level.
     */
    void compute_level_parts(RuleBatch& fixture, std::size_t level, LevelParts& parts)
    {
        const AnchorHead& head = fixture.heads()[level];
        AssignmentInput given;
        given.layout = &fixture.layout();
        given.head_index = static_cast<std::int64_t>(level);
        given.shape = head.shape;
        given.classes = K_CLASSES;
        given.truth = &fixture.truth();
        given.network = NetworkSize{K_NETWORK_SIDE, K_NETWORK_SIDE};
        std::vector<TargetAssignment> assignments;
        SA_EXPECT_OK(ShapeRatioAssignment(ShapeRatioOptions()).assign(given, assignments));

        ShapeRatioDeltaInput input;
        input.head = fixture.output(level);
        input.shape = head.shape;
        input.options = head.options;
        input.layout = &fixture.layout();
        input.head_index = static_cast<std::int64_t>(level);
        input.training = &head.training;
        input.network = NetworkSize{K_NETWORK_SIDE, K_NETWORK_SIDE};
        input.assignments = SushiRuntime::span<const TargetAssignment>(assignments.data(),
                                                                       assignments.size());
        std::vector<float> targets;
        SA_EXPECT_OK(build_objectness_targets(input, targets));
        SA_ASSERT_OK_AND_ASSIGN(
            const ShapeRatioLoss loss,
            measure_shape_ratio_loss(
                input, SushiRuntime::span<const float>(targets.data(), targets.size())));
        SA_EXPECT_OK(compute_shape_ratio_delta(input, parts.alone));

        parts.total = loss.total;
        parts.box = loss.box * derive_shape_ratio_factors(input).box *
                    static_cast<double>(head.shape.batch);
    }

    /** @brief Provides the level of each head of the batch. */
    class Small_ShapeRatioHeadDeltaRuleLevel : public ::testing::TestWithParam<std::size_t>
    {

    };
} // namespace

/** @brief Tests that heads of logits with squared sigmoid sides are accepted. */
TEST(Small_ShapeRatioHeadDeltaRule, CheckAcceptsLogitHeadsOfSquaredSigmoidSides)
{
    const RuleBatch fixture(SizeForm::SquaredSigmoid);

    SA_EXPECT_OK(
        ShapeRatioHeadDeltaRule(ShapeRatioOptions()).check(fixture.layout(), fixture.head_span()));
}

/** @brief Tests that an activated head is refused by name. */
TEST(Small_ShapeRatioHeadDeltaRule, CheckRefusesAnActivatedHead)
{
    RuleBatch fixture(SizeForm::SquaredSigmoid);
    fixture.heads()[1].options.coding = HeadCoding::Activated;
    fixture.heads()[1].options.size = SizeForm::Exponential;

    SA_EXPECT_FAILURE(
        ShapeRatioHeadDeltaRule(ShapeRatioOptions()).check(fixture.layout(), fixture.head_span()),
        StatusCode::InvalidDescription, "heads of logits", "head 'head.1'");
}

/** @brief Tests that a head of exponential sides is refused by name. */
TEST(Small_ShapeRatioHeadDeltaRule, CheckRefusesAHeadOfExponentialSides)
{
    RuleBatch fixture(SizeForm::SquaredSigmoid);
    fixture.heads()[0].options.size = SizeForm::Exponential;

    SA_EXPECT_FAILURE(
        ShapeRatioHeadDeltaRule(ShapeRatioOptions()).check(fixture.layout(), fixture.head_span()),
        StatusCode::InvalidDescription, "squared sigmoids", "head 'head.0'");
}

/** @brief Tests that a layout of more levels than the objectness balance weighs is refused. */
TEST(Small_ShapeRatioHeadDeltaRule, CheckRefusesALayoutOfMoreLevelsThanTheBalanceWeighs)
{
    Vision::Detect::Anchor::AnchorLayout layout;
    layout.anchors = {{0.25f, 0.25f}};
    layout.head_anchors.assign(6, std::vector<std::int64_t>{0});

    SA_EXPECT_FAILURE(ShapeRatioHeadDeltaRule(ShapeRatioOptions())
                          .check(layout, SushiRuntime::span<const AnchorHead>()),
                      StatusCode::InvalidDescription, "at most 5 levels", "layout has 6");
}

/** @brief Tests that the rule gives a level the delta its assignment, loss and delta give. */
TEST_P(Small_ShapeRatioHeadDeltaRuleLevel, ComputeFollowsAssignmentThenLossThenDeltaForTheLevel)
{
    const std::size_t level = GetParam();
    RuleBatch fixture(SizeForm::SquaredSigmoid);
    std::vector<HeadDelta> deltas = make_deltas();
    SA_EXPECT_OK(ShapeRatioHeadDeltaRule(ShapeRatioOptions())
                     .compute(fixture.batch(), view_deltas(deltas)));
    LevelParts parts;
    compute_level_parts(fixture, level, parts);

    EXPECT_EQ(deltas[level].delta, parts.alone.delta);
    EXPECT_EQ(deltas[level].assigned_row, parts.alone.assigned_row);
    EXPECT_EQ(deltas[level].assigned_class, parts.alone.assigned_class);
    EXPECT_EQ(deltas[level].assignments, parts.alone.assignments);
}

INSTANTIATE_TEST_SUITE_P(Levels, Small_ShapeRatioHeadDeltaRuleLevel, ::testing::Values(0u, 1u));

/** @brief Tests that the rule assigns at least one box over the two levels. */
TEST(Small_ShapeRatioHeadDeltaRule, ComputeAssignsTheTruth)
{
    RuleBatch fixture(SizeForm::SquaredSigmoid);
    std::vector<HeadDelta> deltas = make_deltas();

    SA_EXPECT_OK(ShapeRatioHeadDeltaRule(ShapeRatioOptions())
                     .compute(fixture.batch(), view_deltas(deltas)));

    EXPECT_GT(deltas[0].assignments + deltas[1].assignments, 0);
}

/** @brief Tests that the cost shares are the sums of the level losses and the box factors. */
TEST(Small_ShapeRatioHeadDeltaRule, TheCostSharesAreTheSumsOfTheLevelLosses)
{
    RuleBatch fixture(SizeForm::SquaredSigmoid);
    std::vector<HeadDelta> deltas = make_deltas();
    SA_ASSERT_OK_AND_ASSIGN(
        const HeadCost cost,
        ShapeRatioHeadDeltaRule(ShapeRatioOptions()).compute(fixture.batch(),
                                                             view_deltas(deltas)));
    LevelParts fine;
    LevelParts coarse;
    compute_level_parts(fixture, 0, fine);
    compute_level_parts(fixture, 1, coarse);

    EXPECT_NEAR(cost.total, fine.total + coarse.total, K_SUM_TOLERANCE);
    EXPECT_NEAR(cost.box, fine.box + coarse.box, K_SUM_TOLERANCE);
    EXPECT_NEAR(cost.non_box, (fine.total - fine.box) + (coarse.total - coarse.box),
                K_SUM_TOLERANCE);
}

/** @brief Tests that the stored cost equals the total, which is above zero. */
TEST(Small_ShapeRatioHeadDeltaRule, TheStoredCostEqualsTheTotal)
{
    RuleBatch fixture(SizeForm::SquaredSigmoid);
    std::vector<HeadDelta> deltas = make_deltas();

    SA_ASSERT_OK_AND_ASSIGN(
        const HeadCost cost,
        ShapeRatioHeadDeltaRule(ShapeRatioOptions()).compute(fixture.batch(),
                                                             view_deltas(deltas)));

    EXPECT_GT(cost.total, 0.0);
    EXPECT_EQ(cost.stored, cost.total);
}

/** @brief Tests that the box and the non-box share add up to the total. */
TEST(Small_ShapeRatioHeadDeltaRule, TheSharesAddUpToTheTotal)
{
    RuleBatch fixture(SizeForm::SquaredSigmoid);
    std::vector<HeadDelta> deltas = make_deltas();

    SA_ASSERT_OK_AND_ASSIGN(
        const HeadCost cost,
        ShapeRatioHeadDeltaRule(ShapeRatioOptions()).compute(fixture.batch(),
                                                             view_deltas(deltas)));

    EXPECT_NEAR(cost.box + cost.non_box, cost.total, K_SUM_TOLERANCE * cost.total);
}

/** @brief Tests that a truth of another batch than the heads' is refused naming both. */
TEST(Small_ShapeRatioHeadDeltaRule, ComputeRefusesATruthOfAnotherBatch)
{
    RuleBatch fixture(SizeForm::SquaredSigmoid);
    fixture.truth().batch = 2;
    std::vector<HeadDelta> deltas = make_deltas();

    SA_EXPECT_FAILURE(
        ShapeRatioHeadDeltaRule(ShapeRatioOptions()).compute(fixture.batch(),
                                                             view_deltas(deltas)),
        StatusCode::InvalidDescription, "1 images", "batch of 2");
}
