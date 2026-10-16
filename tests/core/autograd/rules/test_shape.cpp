/**************************************************************************/
/* test_shape.cpp                                                         */
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
 * @file test_shape.cpp
 * @brief Tests that the shape rule table pairs reshape and layout conversion with their rules.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/rules/shape.hpp>
#include <SushiAI/core/ops.hpp>

using SushiAI::Autograd::K_SHAPE_RULES;
namespace Autograd = SushiAI::Autograd;
namespace Ops = SushiAI::Ops;

/** @brief Tests that the table pairs reshape and layout conversion with their own rules. */
TEST(Small_ShapeRules, TheTablePairsEachStructuralOperationWithItsRule)
{
    ASSERT_EQ(K_SHAPE_RULES.size(), 2u);
    EXPECT_EQ(K_SHAPE_RULES[0].op, Ops::K_RESHAPE.id);
    EXPECT_EQ(K_SHAPE_RULES[0].rule, &Autograd::reshape_backward);
    EXPECT_EQ(K_SHAPE_RULES[1].op, Ops::K_LAYOUT_CONVERT.id);
    EXPECT_EQ(K_SHAPE_RULES[1].rule, &Autograd::layout_convert_backward);
}

/** @brief Tests that both operations are declared differentiable and have a rule in the library. */
TEST(Small_ShapeRules, BothOperationsAreDifferentiableAndHaveARule)
{
    EXPECT_TRUE(Ops::K_RESHAPE.differentiable);
    EXPECT_TRUE(Ops::K_LAYOUT_CONVERT.differentiable);
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_RESHAPE.id));
    EXPECT_TRUE(Autograd::has_backward_rule(Ops::K_LAYOUT_CONVERT.id));
}
