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
 * @file test_activation.cpp
 * @brief Tests that the activation rule table sends each activation to activation_backward.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
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

/** @brief Holds one row of the activation rule table as the test parameter. */
class Small_ActivationRuleRow : public ::testing::TestWithParam<Autograd::RuleEntry>
{
};

/** @brief Tests that the row's rule is activation_backward and the library reports it. */
TEST_P(Small_ActivationRuleRow, OneRuleServesTheActivation)
{
    const Autograd::RuleEntry& entry = GetParam();

    EXPECT_EQ(entry.rule, &Autograd::activation_backward);
    EXPECT_TRUE(Autograd::has_backward_rule(entry.op));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ActivationRuleRow,
                         ::testing::ValuesIn(K_ACTIVATION_RULES.begin(), K_ACTIVATION_RULES.end()),
                         [](const ::testing::TestParamInfo<Autograd::RuleEntry>& info)
                         {
                             std::string name(Ops::name_of(info.param.op));
                             for (char& letter : name)
                             {
                                 if (letter == '.')
                                     letter = '_';
                             }
                             return name;
                         });
