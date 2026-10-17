/**************************************************************************/
/* test_elementwise.cpp                                                   */
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
 * @file test_elementwise.cpp
 * @brief Tests that the elementwise rule table pairs each operation with its own rule.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/rules/elementwise.hpp>
#include <SushiAI/ops/ops.hpp>

using SushiAI::Autograd::BackwardRule;
using SushiAI::Autograd::K_ELEMENTWISE_RULES;
namespace Autograd = SushiAI::Autograd;
namespace Ops = SushiAI::Ops;

namespace
{
    /** @brief Holds one expected row: the operation and the rule declared for it. */
    struct ExpectedRow
    {
        Ops::OpName op;    /**< The forward operation. */
        BackwardRule rule; /**< The rule elementwise.hpp declares for it. */
    };

    /** @brief Holds the rows the table lists, in table order. */
    const ExpectedRow K_EXPECTED[] = {
        {Ops::K_ADD, &Autograd::add_backward},
        {Ops::K_SUB, &Autograd::sub_backward},
        {Ops::K_MUL, &Autograd::mul_backward},
        {Ops::K_SCALE, &Autograd::scale_backward},
        {Ops::K_EXP, &Autograd::exp_backward},
        {Ops::K_ACCUMULATE, &Autograd::accumulate_backward},
        {Ops::K_DIV, &Autograd::div_backward},
        {Ops::K_MINIMUM, &Autograd::minimum_backward},
        {Ops::K_MAXIMUM, &Autograd::maximum_backward},
        {Ops::K_ABS, &Autograd::abs_backward},
        {Ops::K_ATAN, &Autograd::atan_backward},
        {Ops::K_SELECT, &Autograd::select_backward},
    };
} // namespace

/** @brief Tests that every row pairs its operation with the rule declared for it. */
TEST(Small_ElementwiseRules, EveryRowPairsItsOperationWithItsOwnRule)
{
    ASSERT_EQ(K_ELEMENTWISE_RULES.size(), sizeof(K_EXPECTED) / sizeof(K_EXPECTED[0]));
    for (std::size_t i = 0; i < K_ELEMENTWISE_RULES.size(); ++i)
    {
        EXPECT_EQ(K_ELEMENTWISE_RULES[i].op, K_EXPECTED[i].op.id) << K_EXPECTED[i].op.name;
        EXPECT_EQ(K_ELEMENTWISE_RULES[i].rule, K_EXPECTED[i].rule) << K_EXPECTED[i].op.name;
    }
}

/** @brief Tests that every operation is declared differentiable and has a rule in the library. */
TEST(Small_ElementwiseRules, EveryOperationIsDifferentiableAndHasARule)
{
    for (const ExpectedRow& row : K_EXPECTED)
    {
        EXPECT_TRUE(row.op.differentiable) << row.op.name;
        EXPECT_TRUE(Autograd::has_backward_rule(row.op.id)) << row.op.name;
    }
}

/** @brief Tests that no operation appears in two rows. */
TEST(Small_ElementwiseRules, NoOperationAppearsTwice)
{
    for (std::size_t i = 0; i < K_ELEMENTWISE_RULES.size(); ++i)
    {
        for (std::size_t j = i + 1; j < K_ELEMENTWISE_RULES.size(); ++j)
            EXPECT_NE(K_ELEMENTWISE_RULES[i].op, K_ELEMENTWISE_RULES[j].op) << i << ", " << j;
    }
}
