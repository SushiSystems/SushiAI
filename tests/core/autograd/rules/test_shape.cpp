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
 * @file test_shape.cpp
 * @brief Tests that the shape rule table pairs reshape and layout conversion with their rules.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/rules/shape.hpp>
#include <SushiAI/ops/ops.hpp>

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
