/**************************************************************************/
/* test_rule_table.cpp                                                    */
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
            // A domain left out of all.hpp still compiles; see tests/unit/README.md.
            constexpr std::size_t K_DOMAINS =
                Autograd::K_ELEMENTWISE_RULES.size() + Autograd::K_LINALG_RULES.size()
                + Autograd::K_ACTIVATION_RULES.size() + Autograd::K_LOSS_RULES.size()
                + Autograd::K_SHAPE_RULES.size() + Autograd::K_SPATIAL_RULES.size()
                + Autograd::K_NORM_RULES.size();

            EXPECT_EQ(Autograd::K_RULES.size(), K_DOMAINS);
        }

        TEST(Unit_RuleTable, NoOperationHasTwoRules)
        {
            // The first of two entries would win silently; see tests/unit/README.md.
            for (std::size_t i = 0; i < Autograd::K_RULES.size(); ++i)
                for (std::size_t j = i + 1; j < Autograd::K_RULES.size(); ++j)
                    EXPECT_NE(Autograd::K_RULES[i].op, Autograd::K_RULES[j].op)
                        << "'" << Ops::name_of(Autograd::K_RULES[i].op)
                        << "' has two backward rules";
        }

        TEST(Unit_RuleTable, TheRulesThatExistedBeforeTheSplitStillDo)
        {
            // These fourteen rules predate the split; see tests/unit/README.md.
            for (OpID op : {Ops::K_MATMUL.id, Ops::K_BIAS_ADD.id, Ops::K_ADD.id,
                                 Ops::K_SUB.id, Ops::K_MUL.id, Ops::K_SCALE.id, Ops::K_EXP.id,
                                 Ops::K_RELU.id, Ops::K_GELU.id, Ops::K_SIGMOID.id,
                                 Ops::K_TANH.id, Ops::K_SUM_ROWS.id, Ops::K_ACCUMULATE.id,
                                 Ops::K_NLL_LOSS.id})
                EXPECT_TRUE(Autograd::has_backward_rule(op))
                    << "'" << Ops::name_of(op) << "' lost its rule in the split";
        }

        /** @brief Tests that no identity declared a gradient barrier has a backward rule. */
        TEST(Unit_RuleTable, NoGradientBarrierHasARule)
        {
            std::size_t barriers = 0;
            for (const Ops::OpName& entry : Ops::K_ALL)
            {
                if (!entry.blocks_gradient)
                    continue;

                ++barriers;
                EXPECT_FALSE(Autograd::has_rule(entry.id)) << entry.name;
                EXPECT_FALSE(Autograd::has_backward_rule(entry.id)) << entry.name;
            }

            EXPECT_EQ(barriers, 4u) << "the loop above checked a different set than it names";
        }

        /** @brief Tests that the comparison is a gradient barrier and has no rule row. */
        TEST(Unit_RuleTable, TheComparisonIsABarrierWithoutARule)
        {
            EXPECT_FALSE(Ops::K_LESS_EQUAL.differentiable);
            EXPECT_TRUE(Ops::K_LESS_EQUAL.blocks_gradient);
            EXPECT_FALSE(Autograd::has_rule(Ops::K_LESS_EQUAL.id));
        }

        /** @brief Tests that each ACTIVATIONS row has a rule for its forward identity alone. */
        TEST(Unit_RuleTable, EveryActivationRowHasARuleForItsForwardIdentityOnly)
        {
            for (const Ops::Activation& entry : Ops::K_ACTIVATIONS)
            {
                EXPECT_TRUE(Autograd::has_rule(entry.forward))
                    << "'" << Ops::name_of(entry.forward) << "' has no backward rule";
                EXPECT_FALSE(Autograd::has_rule(entry.backward))
                    << "'" << Ops::name_of(entry.backward) << "' is a gradient with a rule";
            }

            EXPECT_EQ(Autograd::K_ACTIVATION_RULES.size(), Ops::K_ACTIVATIONS.size());
        }

        /** @brief Tests that the normalisation domain rules BATCHNORM2D and not the fold. */
        TEST(Unit_RuleTable, BatchNormFoldHasNoRuleAndBatchNormKeepsIts)
        {
            EXPECT_TRUE(Autograd::has_rule(Ops::K_BATCHNORM2D.id));
            EXPECT_FALSE(Autograd::has_rule(Ops::K_BATCHNORM_FOLD.id));
            EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_BATCHNORM_FOLD.id));
            EXPECT_EQ(Autograd::K_NORM_RULES.size(), 2u);
        }
    }
}
