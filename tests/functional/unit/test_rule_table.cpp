/****************************************************************/
/* test_rule_table.cpp                                          */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_rule_table.cpp
 * @brief Tests that every domain's backward rules reach the combined table exactly once.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/rules/all.hpp>

namespace SushiAI
{
    namespace Test
    {
        TEST(Unit_RuleTable, EveryDomainsRulesReachTheCombinedTable)
        {
            // A domain left out of all.hpp still compiles; see tests/functional/unit/README.md.
            constexpr std::size_t domains =
                Autograd::ELEMENTWISE_RULES.size() + Autograd::LINALG_RULES.size()
                + Autograd::ACTIVATION_RULES.size() + Autograd::LOSS_RULES.size()
                + Autograd::SHAPE_RULES.size() + Autograd::SPATIAL_RULES.size()
                + Autograd::NORM_RULES.size();

            EXPECT_EQ(Autograd::RULES.size(), domains);
        }

        TEST(Unit_RuleTable, NoOperationHasTwoRules)
        {
            // The first of two entries would win silently; see tests/functional/unit/README.md.
            for (std::size_t i = 0; i < Autograd::RULES.size(); ++i)
                for (std::size_t j = i + 1; j < Autograd::RULES.size(); ++j)
                    EXPECT_NE(Autograd::RULES[i].op, Autograd::RULES[j].op)
                        << "'" << Ops::name_of(Autograd::RULES[i].op)
                        << "' has two backward rules";
        }

        TEST(Unit_RuleTable, TheRulesThatExistedBeforeTheSplitStillDo)
        {
            // These fourteen rules predate the split; see tests/functional/unit/README.md.
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
