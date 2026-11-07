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
 * @file test_loss.cpp
 * @brief Tests the loss rule table and which rule starts from the implicit scalar seed.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/rules/loss.hpp>
#include <SushiAI/ops/ops.hpp>

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
}

/** @brief Holds one row of the loss rule table as the test parameter. */
class Small_LossRuleRow : public ::testing::TestWithParam<Autograd::RuleEntry>
{
};

/** @brief Tests that the library reports a backward rule for the operation of the row. */
TEST_P(Small_LossRuleRow, EachLossRowHasABackwardRule)
{
    EXPECT_TRUE(Autograd::has_backward_rule(GetParam().op)) << Ops::name_of(GetParam().op);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_LossRuleRow,
                         ::testing::ValuesIn(K_LOSS_RULES.begin(), K_LOSS_RULES.end()),
                         [](const ::testing::TestParamInfo<Autograd::RuleEntry>& info)
                         {
                             std::string name(Ops::name_of(info.param.op));
                             std::replace(name.begin(), name.end(), '.', '_');
                             return name;
                         });

/** @brief Tests that only the NLL loss seeds itself; its neighbour and other roots do not. */
TEST(Small_LossRules, OnlyTheNLLLossStartsFromTheImplicitSeed)
{
    static_assert(starts_from_implicit_seed(Ops::K_NLL_LOSS.id), "the check is constexpr");

    EXPECT_TRUE(starts_from_implicit_seed(Ops::K_NLL_LOSS.id));
    EXPECT_FALSE(starts_from_implicit_seed(Ops::K_LOG_SOFTMAX.id));
    EXPECT_FALSE(starts_from_implicit_seed(Ops::K_ADD.id));
    EXPECT_FALSE(starts_from_implicit_seed(Ops::K_SCALE.id));
}
