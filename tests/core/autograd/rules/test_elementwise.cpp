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
 * @file test_elementwise.cpp
 * @brief Tests that the elementwise rule table pairs each operation with its own rule.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <vector>
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

namespace
{
    /** @brief Holds one expected row of the elementwise rule table as the test parameter. */
    class Small_ElementwiseRuleRow : public ::testing::TestWithParam<ExpectedRow>
    {
    };

    /** @brief Finds the rows of the table that hold @p op. */
    std::vector<Autograd::RuleEntry> rows_holding(SushiAI::OpID op)
    {
        std::vector<Autograd::RuleEntry> rows;
        std::copy_if(K_ELEMENTWISE_RULES.begin(), K_ELEMENTWISE_RULES.end(),
                     std::back_inserter(rows),
                     [op](const Autograd::RuleEntry& entry) { return entry.op == op; });
        return rows;
    }
} // namespace

/** @brief Tests that the table holds the operation of the row once, with the rule declared for it. */
TEST_P(Small_ElementwiseRuleRow, EveryRowPairsItsOperationWithItsOwnRule)
{
    const ExpectedRow& expected = GetParam();
    const std::vector<Autograd::RuleEntry> rows = rows_holding(expected.op.id);

    ASSERT_EQ(rows.size(), 1u) << expected.op.name;
    EXPECT_EQ(rows[0].rule, expected.rule) << expected.op.name;
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ElementwiseRuleRow,
                         ::testing::ValuesIn(std::begin(K_EXPECTED), std::end(K_EXPECTED)),
                         [](const ::testing::TestParamInfo<ExpectedRow>& info)
                         {
                             std::string name(info.param.op.name);
                             std::replace(name.begin(), name.end(), '.', '_');
                             return name;
                         });

/** @brief Tests that the table has a row for each expected operation and no other. */
TEST(Small_ElementwiseRules, TheTableHoldsOneRowPerExpectedOperation)
{
    EXPECT_EQ(K_ELEMENTWISE_RULES.size(), sizeof(K_EXPECTED) / sizeof(K_EXPECTED[0]));
}
