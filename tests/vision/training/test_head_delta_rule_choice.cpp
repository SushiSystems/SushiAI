/**************************************************************************/
/* test_head_delta_rule_choice.cpp                                        */
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
 * @file test_head_delta_rule_choice.cpp
 * @brief Tests that the assignment an anchor task selects builds the matching head delta rule.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/anchor/assignment_section.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_assignment.hpp>
#include <SushiAI/vision/training/best_shape_head_delta_rule.hpp>
#include <SushiAI/vision/training/head_delta_rule.hpp>
#include <SushiAI/vision/training/head_delta_rule_choice.hpp>
#include <SushiAI/vision/training/shape_ratio_head_delta_rule.hpp>

#include "head_delta_rule_fixture.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::HeadDeltaRuleTest;
    using Vision::Detect::Anchor::AssignmentChoice;
    using Vision::Detect::Anchor::AssignmentRuleKind;
    using Vision::Detect::Anchor::HeadCost;
    using Vision::Detect::Anchor::HeadDelta;
    using Vision::Detect::Anchor::SizeForm;
    using Vision::Training::BestShapeHeadDeltaRule;
    using Vision::Training::choose_head_delta_rule;
    using Vision::Training::HeadDeltaRule;
    using Vision::Training::ShapeRatioHeadDeltaRule;

    /** @brief Returns the choice of the shape-ratio rule with a neighbour offset of @p offset. */
    AssignmentChoice shape_ratio_choice(float offset)
    {
        AssignmentChoice choice;
        choice.rule = AssignmentRuleKind::ShapeRatio;
        choice.shape_ratio.neighbour_offset = offset;
        return choice;
    }

    /** @brief Returns the slots @p rule assigns over the squared sigmoid fixture. */
    std::int64_t count_assignments(const HeadDeltaRule& rule)
    {
        RuleBatch fixture(SizeForm::SquaredSigmoid);
        std::vector<HeadDelta> deltas = make_deltas();
        const Result<HeadCost> cost = rule.compute(fixture.batch(), view_deltas(deltas));

        EXPECT_TRUE(cost.has_value());
        return deltas[0].assignments + deltas[1].assignments;
    }
} // namespace

/** @brief Tests that the default choice builds the best-shape rule. */
TEST(Small_HeadDeltaRuleChoice, BestShapeBuildsTheBestShapeRule)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::shared_ptr<const HeadDeltaRule> rule,
                            choose_head_delta_rule(AssignmentChoice()));

    ASSERT_NE(rule, nullptr);
    EXPECT_NE(dynamic_cast<const BestShapeHeadDeltaRule*>(rule.get()), nullptr);
}

/** @brief Tests that the shape-ratio choice builds the shape-ratio rule. */
TEST(Small_HeadDeltaRuleChoice, ShapeRatioBuildsTheShapeRatioRule)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::shared_ptr<const HeadDeltaRule> rule,
                            choose_head_delta_rule(shape_ratio_choice(0.5f)));

    ASSERT_NE(rule, nullptr);
    EXPECT_NE(dynamic_cast<const ShapeRatioHeadDeltaRule*>(rule.get()), nullptr);
}

/** @brief Tests that the settings of a shape-ratio choice reach the rule it builds. */
TEST(Small_HeadDeltaRuleChoice, TheShapeRatioSettingsReachTheRule)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::shared_ptr<const HeadDeltaRule> with_neighbours,
                            choose_head_delta_rule(shape_ratio_choice(0.5f)));
    SA_ASSERT_OK_AND_ASSIGN(const std::shared_ptr<const HeadDeltaRule> centre_only,
                            choose_head_delta_rule(shape_ratio_choice(0.0f)));

    // The box at (20, 20) is 1.25 cells in, so an offset of 0.5 takes its left and top cells.
    EXPECT_GT(count_assignments(*with_neighbours), count_assignments(*centre_only));
}

/** @brief Tests that shape-ratio settings the assignment refuses are refused naming the key. */
TEST(Small_HeadDeltaRuleChoice, ShapeRatioSettingsThatAreRefusedNameTheirKey)
{
    AssignmentChoice choice = shape_ratio_choice(0.5f);
    choice.shape_ratio.ratio_limit = 1.0f;

    SA_EXPECT_FAILURE(choose_head_delta_rule(choice), StatusCode::InvalidDescription,
                      "'ratio_limit'");
}

/** @brief Tests that the settings of the shape-ratio rule are not read for the best-shape one. */
TEST(Small_HeadDeltaRuleChoice, BestShapeIgnoresTheShapeRatioSettings)
{
    AssignmentChoice choice;
    choice.shape_ratio.ratio_limit = 1.0f;

    SA_ASSERT_OK_AND_ASSIGN(const std::shared_ptr<const HeadDeltaRule> rule,
                            choose_head_delta_rule(choice));

    EXPECT_NE(dynamic_cast<const BestShapeHeadDeltaRule*>(rule.get()), nullptr);
}
