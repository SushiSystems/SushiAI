/**************************************************************************/
/* test_loss.cpp                                                          */
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
 * @file test_loss.cpp
 * @brief Tests the loss rule table and which rule starts from the implicit scalar seed.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/rules/loss.hpp>
#include <SushiAI/core/ops.hpp>

using SushiAI::Autograd::K_LOSS_RULES;
using SushiAI::Autograd::starts_from_implicit_seed;
namespace Autograd = SushiAI::Autograd;
namespace Ops = SushiAI::Ops;

/** @brief Tests that the table pairs log-softmax and the NLL loss with their own rules. */
TEST(Small_LossRules, TheTablePairsEachLossOperationWithItsRule)
{
    ASSERT_EQ(K_LOSS_RULES.size(), 2u);
    EXPECT_EQ(K_LOSS_RULES[0].op, Ops::K_LOG_SOFTMAX.id);
    EXPECT_EQ(K_LOSS_RULES[0].rule, &Autograd::log_softmax_backward);
    EXPECT_EQ(K_LOSS_RULES[1].op, Ops::K_NLL_LOSS.id);
    EXPECT_EQ(K_LOSS_RULES[1].rule, &Autograd::nll_loss_backward);

    for (const Autograd::RuleEntry& entry : K_LOSS_RULES)
        EXPECT_TRUE(Autograd::has_backward_rule(entry.op)) << Ops::name_of(entry.op);
}

/** @brief Tests that only the NLL loss seeds itself; its neighbour and other roots do not. */
TEST(Small_LossRules, OnlyTheNLLLossStartsFromTheImplicitSeed)
{
    static_assert(starts_from_implicit_seed(Ops::K_NLL_LOSS.id), "the check is constexpr");

    EXPECT_TRUE(starts_from_implicit_seed(Ops::K_NLL_LOSS.id));
    EXPECT_FALSE(starts_from_implicit_seed(Ops::K_LOG_SOFTMAX.id));
    EXPECT_FALSE(starts_from_implicit_seed(Ops::K_ADD.id));
    EXPECT_FALSE(starts_from_implicit_seed(Ops::K_SCALE.id));
}
