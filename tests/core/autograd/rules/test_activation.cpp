/**************************************************************************/
/* test_activation.cpp                                                    */
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
 * @file test_activation.cpp
 * @brief Tests that the activation rule table sends each activation to activation_backward.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/rules/activation.hpp>
#include <SushiAI/ops/ops.hpp>

using SushiAI::Autograd::K_ACTIVATION_RULES;
namespace Autograd = SushiAI::Autograd;
namespace Ops = SushiAI::Ops;

namespace
{
    /** @brief Holds the operations the table lists, in table order. */
    const Ops::OpName K_EXPECTED[] = {
        Ops::K_RELU, Ops::K_GELU, Ops::K_SIGMOID, Ops::K_TANH,
        Ops::K_LEAKY_RELU, Ops::K_SILU, Ops::K_SOFTPLUS,
    };
} // namespace

/** @brief Tests that the table lists the seven activations, each declared differentiable. */
TEST(Small_ActivationRules, TheTableListsTheSevenActivations)
{
    ASSERT_EQ(K_ACTIVATION_RULES.size(), sizeof(K_EXPECTED) / sizeof(K_EXPECTED[0]));
    for (std::size_t i = 0; i < K_ACTIVATION_RULES.size(); ++i)
    {
        EXPECT_EQ(K_ACTIVATION_RULES[i].op, K_EXPECTED[i].id) << "row " << i;
        EXPECT_TRUE(K_EXPECTED[i].differentiable) << K_EXPECTED[i].name;
    }
}

/** @brief Tests that one rule serves every row and the library reports a rule for each. */
TEST(Small_ActivationRules, OneRuleServesEveryActivation)
{
    for (const Autograd::RuleEntry& entry : K_ACTIVATION_RULES)
    {
        EXPECT_EQ(entry.rule, &Autograd::activation_backward) << Ops::name_of(entry.op);
        EXPECT_TRUE(Autograd::has_backward_rule(entry.op)) << Ops::name_of(entry.op);
    }
}
