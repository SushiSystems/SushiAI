/**************************************************************************/
/* test_anchor_head.cpp                                                   */
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
 * @file test_anchor_head.cpp
 * @brief Tests AnchorHead, what an anchor objective knows of one head.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/training/anchor_head.hpp>
#include <SushiAI/vision/training/best_shape_head_delta_rule.hpp>

#include "head_delta_rule_fixture.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::HeadDeltaRuleTest;
    using Vision::Detect::Anchor::BoxLoss;
    using Vision::Detect::Anchor::HeadCoding;
    using Vision::Detect::Anchor::ObjectnessTarget;
    using Vision::Detect::Anchor::SizeForm;
    using Vision::Training::AnchorHead;
    using Vision::Training::BestShapeHeadDeltaRule;
} // namespace

/** @brief Tests that a default head names no output, has no extent and the default options. */
TEST(Small_AnchorHead, ADefaultHeadIsEmpty)
{
    const AnchorHead head;

    EXPECT_TRUE(head.output.empty());
    EXPECT_EQ(head.shape.batch, 0);
    EXPECT_EQ(head.shape.height, 0);
    EXPECT_EQ(head.shape.width, 0);
    EXPECT_EQ(head.options.num_classes, 0);
    EXPECT_EQ(head.options.coding, HeadCoding::Logit);
    EXPECT_EQ(head.options.scale_xy, 1.0f);
    EXPECT_EQ(head.options.size, SizeForm::Exponential);
    EXPECT_EQ(head.training.box_loss, BoxLoss::SquaredError);
    EXPECT_EQ(head.training.objectness_target, ObjectnessTarget::One);
}

/** @brief Tests that a copy of a head holds every part and changes apart from its source. */
TEST(Small_AnchorHead, ACopyHoldsEveryPartAndChangesApart)
{
    AnchorHead head;
    head.output = "head.0";
    head.shape.batch = 2;
    head.shape.height = 4;
    head.shape.width = 3;
    head.options.num_classes = 7;
    head.training.box_weight = 0.5f;

    AnchorHead copy = head;
    copy.output = "head.1";
    copy.training.box_weight = 2.0f;

    EXPECT_EQ(head.output, "head.0");
    EXPECT_EQ(head.training.box_weight, 0.5f);
    EXPECT_EQ(copy.shape.batch, 2);
    EXPECT_EQ(copy.shape.height, 4);
    EXPECT_EQ(copy.shape.width, 3);
    EXPECT_EQ(copy.options.num_classes, 7);
}

/** @brief Tests that a rule names a head by its output in a refusal, so the field is read. */
TEST(Small_AnchorHead, ARuleNamesAHeadByItsOutput)
{
    RuleBatch fixture(SizeForm::Exponential);
    fixture.heads()[0].output = "stem.detect";
    fixture.heads()[0].options.size = SizeForm::SquaredSigmoid;

    SA_EXPECT_FAILURE(BestShapeHeadDeltaRule().check(fixture.layout(), fixture.head_span()),
                      StatusCode::InvalidDescription, "head 'stem.detect'");
}
