/****************************************************************/
/* test_rule_table.cpp                                          */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#include <gtest/gtest.h>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/rules/all.hpp>

namespace SushiAI
{
    namespace Test
    {
        TEST(Unit_RuleTable, EveryDomainsRulesReachTheCombinedTable)
        {
            // all.hpp concatenates; a domain left out of the concat compiles
            // fine and simply never differentiates. The count is the cheapest
            // thing that notices.
            constexpr std::size_t domains =
                Autograd::ELEMENTWISE_RULES.size() + Autograd::LINALG_RULES.size()
                + Autograd::ACTIVATION_RULES.size() + Autograd::LOSS_RULES.size()
                + Autograd::SHAPE_RULES.size() + Autograd::SPATIAL_RULES.size()
                + Autograd::NORM_RULES.size();

            EXPECT_EQ(Autograd::RULES.size(), domains);
        }

        TEST(Unit_RuleTable, NoOperationHasTwoRules)
        {
            // Two entries for one identity is not a compile error and the
            // first one silently wins - so which gradient runs would depend on
            // the order two domains happened to be concatenated in.
            for (std::size_t i = 0; i < Autograd::RULES.size(); ++i)
                for (std::size_t j = i + 1; j < Autograd::RULES.size(); ++j)
                    EXPECT_NE(Autograd::RULES[i].op, Autograd::RULES[j].op)
                        << "'" << Ops::name_of(Autograd::RULES[i].op)
                        << "' has two backward rules";
        }

        TEST(Unit_RuleTable, TheRulesThatExistedBeforeTheSplitStillDo)
        {
            // The split is a move, not a rewrite. These fourteen were the
            // table before it; every one must still resolve.
            for (OpID op : {Ops::MATMUL.id, Ops::BIAS_ADD.id, Ops::ADD.id,
                                 Ops::SUB.id, Ops::MUL.id, Ops::SCALE.id, Ops::EXP.id,
                                 Ops::RELU.id, Ops::GELU.id, Ops::SIGMOID.id,
                                 Ops::TANH.id, Ops::SUM_ROWS.id, Ops::ACCUMULATE.id,
                                 Ops::NLL_LOSS.id})
                EXPECT_TRUE(Autograd::has_backward_rule(op))
                    << "'" << Ops::name_of(op) << "' lost its rule in the split";
        }
    }
}
