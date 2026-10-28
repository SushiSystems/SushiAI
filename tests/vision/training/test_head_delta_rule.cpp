/**************************************************************************/
/* test_head_delta_rule.cpp                                               */
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
 * @file test_head_delta_rule.cpp
 * @brief Tests the HeadDeltaRule interface and its batch through both rules.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_assignment.hpp>
#include <SushiAI/vision/training/best_shape_head_delta_rule.hpp>
#include <SushiAI/vision/training/head_delta_rule.hpp>
#include <SushiAI/vision/training/shape_ratio_head_delta_rule.hpp>

#include "head_delta_rule_fixture.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::HeadDeltaRuleTest;
    using Vision::Detect::Anchor::HeadCost;
    using Vision::Detect::Anchor::HeadDelta;
    using Vision::Detect::Anchor::ShapeRatioOptions;
    using Vision::Detect::Anchor::SizeForm;
    using Vision::Training::BestShapeHeadDeltaRule;
    using Vision::Training::HeadDeltaBatch;
    using Vision::Training::HeadDeltaRule;
    using Vision::Training::ShapeRatioHeadDeltaRule;

    /**
     * @brief Checks that @p rule, used only as a HeadDeltaRule, accepts the heads of @p fixture
     *        and gives each head a delta of its `[B, H, W, A * (5 + C)]` size.
     */
    void expect_delta_shapes(const HeadDeltaRule& rule, RuleBatch& fixture)
    {
        SA_EXPECT_OK(rule.check(fixture.layout(), fixture.head_span()));

        std::vector<HeadDelta> deltas = make_deltas();
        SA_ASSERT_OK_AND_ASSIGN(const HeadCost cost,
                                rule.compute(fixture.batch(), view_deltas(deltas)));

        EXPECT_GT(cost.total, 0.0);
        ASSERT_EQ(deltas.size(), 2u);
        EXPECT_EQ(deltas[0].delta.size(),
                  static_cast<std::size_t>(K_SIDES[0] * K_SIDES[0] * K_CHANNELS));
        EXPECT_EQ(deltas[1].delta.size(),
                  static_cast<std::size_t>(K_SIDES[1] * K_SIDES[1] * K_CHANNELS));
    }
} // namespace

/** @brief Tests that a default batch names no layout, no truth, no head and no output. */
TEST(Small_HeadDeltaRule, ADefaultBatchHoldsNothing)
{
    const HeadDeltaBatch batch;

    EXPECT_EQ(batch.layout, nullptr);
    EXPECT_EQ(batch.truth, nullptr);
    EXPECT_TRUE(batch.heads.empty());
    EXPECT_TRUE(batch.outputs.empty());
    EXPECT_EQ(batch.network.height, 0);
    EXPECT_EQ(batch.network.width, 0);
}

/** @brief Tests that a fixture's batch views the heads, outputs, layout and truth it holds. */
TEST(Small_HeadDeltaRule, ABatchViewsTheHeadsOutputsLayoutAndTruthItIsGiven)
{
    RuleBatch fixture(SizeForm::Exponential);

    const HeadDeltaBatch batch = fixture.batch();

    EXPECT_EQ(batch.layout, &fixture.layout());
    EXPECT_EQ(batch.truth, &fixture.truth());
    ASSERT_EQ(batch.heads.size(), 2u);
    EXPECT_EQ(batch.heads[1].output, "head.1");
    ASSERT_EQ(batch.outputs.size(), 2u);
    EXPECT_EQ(batch.outputs[0].data(), fixture.output(0).data());
    EXPECT_EQ(batch.network.height, K_NETWORK_SIDE);
    EXPECT_EQ(batch.network.width, K_NETWORK_SIDE);
}

/** @brief Tests that the best-shape rule checks and computes through the interface. */
TEST(Small_HeadDeltaRule, TheBestShapeRuleServesTheInterface)
{
    RuleBatch fixture(SizeForm::Exponential);
    const std::unique_ptr<HeadDeltaRule> rule = std::make_unique<BestShapeHeadDeltaRule>();

    expect_delta_shapes(*rule, fixture);
}

/** @brief Tests that the shape-ratio rule checks and computes through the interface. */
TEST(Small_HeadDeltaRule, TheShapeRatioRuleServesTheInterface)
{
    RuleBatch fixture(SizeForm::SquaredSigmoid);
    const std::unique_ptr<HeadDeltaRule> rule =
        std::make_unique<ShapeRatioHeadDeltaRule>(ShapeRatioOptions());

    expect_delta_shapes(*rule, fixture);
}

/** @brief Tests that a rule refuses through check the heads it cannot train. */
TEST(Small_HeadDeltaRule, EachRuleRefusesThroughCheckTheHeadsOfTheOther)
{
    const RuleBatch exponential(SizeForm::Exponential);
    const RuleBatch squared(SizeForm::SquaredSigmoid);
    const std::unique_ptr<HeadDeltaRule> best = std::make_unique<BestShapeHeadDeltaRule>();
    const std::unique_ptr<HeadDeltaRule> ratio =
        std::make_unique<ShapeRatioHeadDeltaRule>(ShapeRatioOptions());

    EXPECT_FALSE(best->check(squared.layout(), squared.head_span()).is_ok());
    EXPECT_FALSE(ratio->check(exponential.layout(), exponential.head_span()).is_ok());
}
