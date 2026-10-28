/**************************************************************************/
/* test_best_shape_head_delta_rule.cpp                                    */
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
 * @file test_best_shape_head_delta_rule.cpp
 * @brief Tests the best-shape head delta rule against compute_head_delta and measure_head_cost.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/training/anchor_head.hpp>
#include <SushiAI/vision/training/best_shape_head_delta_rule.hpp>

#include "head_delta_rule_fixture.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::HeadDeltaRuleTest;
    using Vision::Detect::Anchor::compute_head_delta;
    using Vision::Detect::Anchor::HeadCost;
    using Vision::Detect::Anchor::HeadDelta;
    using Vision::Detect::Anchor::measure_head_cost;
    using Vision::Detect::Anchor::NetworkSize;
    using Vision::Detect::Anchor::SizeForm;
    using Vision::Training::AnchorHead;
    using Vision::Training::BestShapeHeadDeltaRule;
} // namespace

/** @brief Tests that heads decoding their sides as exponentials are accepted. */
TEST(Small_BestShapeHeadDeltaRule, CheckAcceptsHeadsThatDecodeExponentials)
{
    const RuleBatch fixture(SizeForm::Exponential);

    SA_EXPECT_OK(BestShapeHeadDeltaRule().check(fixture.layout(), fixture.head_span()));
}

/** @brief Tests that a head of squared sigmoid sides is refused by name. */
TEST(Small_BestShapeHeadDeltaRule, CheckRefusesAHeadOfSquaredSigmoidSides)
{
    RuleBatch fixture(SizeForm::Exponential);
    fixture.heads()[1].options.size = SizeForm::SquaredSigmoid;

    SA_EXPECT_FAILURE(BestShapeHeadDeltaRule().check(fixture.layout(), fixture.head_span()),
                      StatusCode::InvalidDescription, "exponentials", "head 'head.1'");
}

/** @brief Tests that the rule gives each head the delta compute_head_delta gives it alone. */
TEST(Small_BestShapeHeadDeltaRule, ComputeWritesTheDeltaOfEachHead)
{
    RuleBatch fixture(SizeForm::Exponential);
    std::vector<HeadDelta> deltas = make_deltas();

    SA_ASSERT_OK_AND_ASSIGN(const HeadCost cost,
                            BestShapeHeadDeltaRule().compute(fixture.batch(),
                                                             view_deltas(deltas)));

    ASSERT_EQ(deltas.size(), 2u);
    for (std::size_t level = 0; level < 2; ++level)
    {
        SCOPED_TRACE(level);
        const AnchorHead& head = fixture.heads()[level];
        HeadDelta alone;
        SA_EXPECT_OK(compute_head_delta(fixture.output(level), head.shape, head.options,
                                        fixture.layout(), static_cast<std::int64_t>(level),
                                        head.training, fixture.truth(),
                                        NetworkSize{K_NETWORK_SIDE, K_NETWORK_SIDE}, alone));

        EXPECT_EQ(deltas[level].delta.size(),
                  static_cast<std::size_t>(K_SIDES[level] * K_SIDES[level] * K_CHANNELS));
        EXPECT_EQ(deltas[level].delta, alone.delta);
        EXPECT_EQ(deltas[level].assigned_row, alone.assigned_row);
        EXPECT_EQ(deltas[level].assigned_class, alone.assigned_class);
        EXPECT_EQ(deltas[level].assignments, alone.assignments);
    }
    EXPECT_GT(deltas[0].assignments + deltas[1].assignments, 0);
    EXPECT_GT(cost.total, 0.0);
}

/** @brief Tests that the cost is the sum over the heads of measure_head_cost. */
TEST(Small_BestShapeHeadDeltaRule, ComputeSumsTheCostOfEachHead)
{
    RuleBatch fixture(SizeForm::Exponential);
    std::vector<HeadDelta> deltas = make_deltas();

    SA_ASSERT_OK_AND_ASSIGN(const HeadCost cost,
                            BestShapeHeadDeltaRule().compute(fixture.batch(),
                                                             view_deltas(deltas)));

    HeadCost expected;
    for (std::size_t level = 0; level < 2; ++level)
    {
        const AnchorHead& head = fixture.heads()[level];
        const HeadCost one =
            measure_head_cost(deltas[level], head.shape, head.options, head.training);
        expected.total += one.total;
        expected.box += one.box;
        expected.non_box += one.non_box;
        expected.stored += one.stored;
    }
    EXPECT_DOUBLE_EQ(cost.total, expected.total);
    EXPECT_DOUBLE_EQ(cost.box, expected.box);
    EXPECT_DOUBLE_EQ(cost.non_box, expected.non_box);
    EXPECT_DOUBLE_EQ(cost.stored, expected.stored);
}

/** @brief Tests that deltas a caller filled before are replaced, not extended. */
TEST(Small_BestShapeHeadDeltaRule, ComputeReplacesTheDeltasItWasGiven)
{
    RuleBatch fixture(SizeForm::Exponential);
    std::vector<HeadDelta> deltas = make_deltas();
    deltas[0].delta.assign(7, 99.0f);
    deltas[1].assigned_row.assign(5, 99);

    SA_ASSERT_OK_AND_ASSIGN(const HeadCost cost,
                            BestShapeHeadDeltaRule().compute(fixture.batch(),
                                                             view_deltas(deltas)));

    EXPECT_GT(cost.total, 0.0);
    EXPECT_EQ(deltas[0].delta.size(),
              static_cast<std::size_t>(K_SIDES[0] * K_SIDES[0] * K_CHANNELS));
    EXPECT_EQ(deltas[1].assigned_row.size(),
              static_cast<std::size_t>(K_SIDES[1] * K_SIDES[1] * 2));
}

/** @brief Tests that a truth of another batch than the heads' is refused naming both. */
TEST(Small_BestShapeHeadDeltaRule, ComputeRefusesATruthOfAnotherBatch)
{
    RuleBatch fixture(SizeForm::Exponential);
    fixture.truth().batch = 2;
    std::vector<HeadDelta> deltas = make_deltas();

    SA_EXPECT_FAILURE(BestShapeHeadDeltaRule().compute(fixture.batch(), view_deltas(deltas)),
                      StatusCode::InvalidDescription, "1 images", "batch of 2");
}
