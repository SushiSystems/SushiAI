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
 * @file test_head_delta_rule.cpp
 * @brief Tests the HeadDeltaRule interface and its batch through both rules.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
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

    /** @brief Names a rule, the size form its heads decode and how to build it. */
    struct RuleRow
    {
        const char* name;                                  /**< Names the row in the test name. */
        SizeForm form;                                     /**< The sides the rule trains. */
        std::unique_ptr<HeadDeltaRule> (*make)();          /**< Builds the rule. */
    };

    /** @brief Builds the best-shape rule. */
    std::unique_ptr<HeadDeltaRule> make_best_shape()
    {
        return std::make_unique<BestShapeHeadDeltaRule>();
    }

    /** @brief Builds the shape-ratio rule with its default options. */
    std::unique_ptr<HeadDeltaRule> make_shape_ratio()
    {
        return std::make_unique<ShapeRatioHeadDeltaRule>(ShapeRatioOptions());
    }

    /** @brief Provides the rules that serve the HeadDeltaRule interface. */
    class Small_HeadDeltaRuleContract : public ::testing::TestWithParam<RuleRow>
    {

    };

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

/** @brief Tests that a rule checks and computes through the interface, whichever rule it is. */
TEST_P(Small_HeadDeltaRuleContract, TheRuleServesTheInterface)
{
    RuleBatch fixture(GetParam().form);
    const std::unique_ptr<HeadDeltaRule> rule = GetParam().make();

    expect_delta_shapes(*rule, fixture);
}

INSTANTIATE_TEST_SUITE_P(Rules, Small_HeadDeltaRuleContract,
                         ::testing::Values(RuleRow{"BestShape", SizeForm::Exponential,
                                                   &make_best_shape},
                                           RuleRow{"ShapeRatio", SizeForm::SquaredSigmoid,
                                                   &make_shape_ratio}),
                         [](const ::testing::TestParamInfo<RuleRow>& info)
                         { return std::string(info.param.name); });

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
