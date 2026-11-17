/**************************************************************************/
/* test_all.cpp                                                           */
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
 * @file test_all.cpp
 * @brief Tests the combined backward rule table: its coverage, its rows and the operations that
 *        have no rule.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/rules/all.hpp>
#include <SushiAI/ops/ops.hpp>

namespace SushiAI
{
    namespace Test
    {
        namespace
        {
            /** @brief Spells an operation's name as a test-parameter name, dots as underscores. */
            std::string case_name(OpID op)
            {
                std::string name(Ops::name_of(op));
                std::replace(name.begin(), name.end(), '.', '_');
                return name;
            }

            /** @brief Expects @p op to be declared differentiable and to have a rule. */
            void expect_differentiable_with_rule(const Ops::OpName& op)
            {
                EXPECT_TRUE(op.differentiable) << op.name;
                EXPECT_TRUE(Autograd::has_rule(op.id)) << op.name;
            }

            /** @brief Holds one row of the combined rule table as the test parameter. */
            class Small_RuleTableEntry : public ::testing::TestWithParam<Autograd::RuleEntry>
            {
            };

            /** @brief Holds one operation that had a rule before the table was split by domain. */
            class Small_RuleTableLegacyOperation : public ::testing::TestWithParam<Ops::OpName>
            {
            };

            /** @brief Holds one operation declared a gradient barrier. */
            class Small_RuleTableBarrier : public ::testing::TestWithParam<Ops::OpName>
            {
            };

            /** @brief Holds one row of the activation table of the operation set. */
            class Small_RuleTableActivation : public ::testing::TestWithParam<Ops::Activation>
            {
            };
        } // namespace

        /** @brief Tests that the combined table holds the rows of the nine domain tables. */
        TEST(Small_RuleTable, EveryDomainsRulesReachTheCombinedTable)
        {
            // A domain left out of all.hpp still compiles; see tests/core/autograd/README.md.
            constexpr std::size_t K_DOMAINS =
                Autograd::K_ELEMENTWISE_RULES.size() + Autograd::K_LINALG_RULES.size()
                + Autograd::K_ACTIVATION_RULES.size() + Autograd::K_LOSS_RULES.size()
                + Autograd::K_SHAPE_RULES.size() + Autograd::K_SPATIAL_RULES.size()
                + Autograd::K_NORM_RULES.size() + Autograd::K_RANDOM_RULES.size()
                + Autograd::K_BROADCAST_RULES.size();

            EXPECT_EQ(Autograd::K_RULES.size(), K_DOMAINS);
        }

        /** @brief Tests that the combined table holds the operation of the row exactly once. */
        TEST_P(Small_RuleTableEntry, NoOperationHasTwoRules)
        {
            // The first of two entries would win silently; see tests/core/autograd/README.md.
            const OpID op = GetParam().op;
            const auto holders = std::count_if(
                Autograd::K_RULES.begin(), Autograd::K_RULES.end(),
                [op](const Autograd::RuleEntry& entry) { return entry.op == op; });

            EXPECT_EQ(holders, 1) << "'" << Ops::name_of(op) << "' has " << holders
                                  << " backward rules";
        }

        /** @brief Tests that the operation of the row is declared differentiable and has a rule. */
        TEST_P(Small_RuleTableEntry, EveryOperationIsDifferentiableAndHasARule)
        {
            const OpID op = GetParam().op;
            const auto declared = std::find_if(
                Ops::K_ALL.begin(), Ops::K_ALL.end(),
                [op](const Ops::OpName& entry) { return entry.id == op; });

            ASSERT_NE(declared, Ops::K_ALL.end()) << Ops::name_of(op);
            EXPECT_TRUE(declared->differentiable) << declared->name;
            EXPECT_TRUE(Autograd::has_backward_rule(op)) << declared->name;
        }

        INSTANTIATE_TEST_SUITE_P(
            Rows, Small_RuleTableEntry,
            ::testing::ValuesIn(Autograd::K_RULES.begin(), Autograd::K_RULES.end()),
            [](const ::testing::TestParamInfo<Autograd::RuleEntry>& info)
            {
                return case_name(info.param.op);
            });

        /** @brief Tests that the operation of the row has a rule in the library. */
        TEST_P(Small_RuleTableLegacyOperation, TheRulesThatExistedBeforeTheSplitStillDo)
        {
            // These fourteen rules predate the split; see tests/core/autograd/README.md.
            EXPECT_TRUE(Autograd::has_backward_rule(GetParam().id))
                << "'" << GetParam().name << "' lost its rule in the split";
        }

        INSTANTIATE_TEST_SUITE_P(
            Rows, Small_RuleTableLegacyOperation,
            ::testing::Values(Ops::K_MATMUL, Ops::K_BIAS_ADD, Ops::K_ADD, Ops::K_SUB, Ops::K_MUL,
                              Ops::K_SCALE, Ops::K_EXP, Ops::K_RELU, Ops::K_GELU, Ops::K_SIGMOID,
                              Ops::K_TANH, Ops::K_SUM_ROWS, Ops::K_ACCUMULATE, Ops::K_NLL_LOSS),
            [](const ::testing::TestParamInfo<Ops::OpName>& info)
            {
                return case_name(info.param.id);
            });

        /** @brief Tests that the gradient barrier of the row is not differentiable, has no rule. */
        TEST_P(Small_RuleTableBarrier, NoGradientBarrierHasARule)
        {
            const Ops::OpName& barrier = GetParam();

            EXPECT_TRUE(barrier.blocks_gradient) << barrier.name;
            EXPECT_FALSE(barrier.differentiable) << barrier.name;
            EXPECT_FALSE(Autograd::has_rule(barrier.id)) << barrier.name;
            EXPECT_FALSE(Autograd::has_backward_rule(barrier.id)) << barrier.name;
        }

        INSTANTIATE_TEST_SUITE_P(
            Rows, Small_RuleTableBarrier,
            ::testing::Values(Ops::K_ZERO, Ops::K_CONSTANT, Ops::K_LESS_EQUAL,
                              Ops::K_STOP_GRADIENT, Ops::K_ARGMAX_ROWS),
            [](const ::testing::TestParamInfo<Ops::OpName>& info)
            {
                return case_name(info.param.id);
            });

        /** @brief Tests that the barrier rows above name every barrier the operation set declares. */
        TEST(Small_RuleTable, TheBarrierRowsNameEveryDeclaredBarrier)
        {
            const auto barriers = std::count_if(
                Ops::K_ALL.begin(), Ops::K_ALL.end(),
                [](const Ops::OpName& entry) { return entry.blocks_gradient; });

            EXPECT_EQ(barriers, 5);
        }

        /** @brief Tests that the activation of the row has a rule for its forward identity alone. */
        TEST_P(Small_RuleTableActivation, EveryActivationRowHasARuleForItsForwardIdentityOnly)
        {
            const Ops::Activation& entry = GetParam();

            EXPECT_TRUE(Autograd::has_rule(entry.forward))
                << "'" << Ops::name_of(entry.forward) << "' has no backward rule";
            EXPECT_FALSE(Autograd::has_rule(entry.backward))
                << "'" << Ops::name_of(entry.backward) << "' is a gradient with a rule";
        }

        INSTANTIATE_TEST_SUITE_P(
            Rows, Small_RuleTableActivation,
            ::testing::ValuesIn(Ops::K_ACTIVATIONS.begin(), Ops::K_ACTIVATIONS.end()),
            [](const ::testing::TestParamInfo<Ops::Activation>& info)
            {
                return case_name(info.param.forward);
            });

        /** @brief Tests that the activation rule table has one row per activation. */
        TEST(Small_RuleTable, TheActivationRuleTableHasOneRowPerActivation)
        {
            EXPECT_EQ(Autograd::K_ACTIVATION_RULES.size(), Ops::K_ACTIVATIONS.size());
        }

        /** @brief Tests that the normalisation domain rules BATCHNORM2D and not the fold. */
        TEST(Small_RuleTable, BatchNormFoldHasNoRuleAndBatchNormKeepsIts)
        {
            EXPECT_TRUE(Autograd::has_rule(Ops::K_BATCHNORM2D.id));
            EXPECT_FALSE(Autograd::has_rule(Ops::K_BATCHNORM_FOLD.id));
            EXPECT_FALSE(Autograd::has_backward_rule(Ops::K_BATCHNORM_FOLD.id));
            EXPECT_EQ(Autograd::K_NORM_RULES.size(), 3u);
        }

        /** @brief Tests that LOG_SOFTMAX has the second loss rule and its gradient none. */
        TEST(Small_RuleTable, LogSoftmaxHasItsRuleAndItsGradientHasNone)
        {
            EXPECT_EQ(Autograd::K_LOSS_RULES.size(), 2u);
            EXPECT_TRUE(Ops::K_LOG_SOFTMAX.differentiable);
            EXPECT_TRUE(Autograd::has_rule(Ops::K_LOG_SOFTMAX.id));
            EXPECT_FALSE(Ops::K_LOG_SOFTMAX_BACKWARD.differentiable);
            EXPECT_FALSE(Autograd::has_rule(Ops::K_LOG_SOFTMAX_BACKWARD.id));
        }

        /** @brief Tests that the depthwise forward has a rule in the spatial table. */
        TEST(Small_RuleTable, TheDepthwiseForwardHasItsRuleAndItsGradientsHaveNone)
        {
            EXPECT_EQ(Autograd::K_SPATIAL_RULES.size(), 10u);
            EXPECT_TRUE(Autograd::has_rule(Ops::K_DEPTHWISE_CONV2D.id));

            // The two gradient identities are not differentiable, so they need no row.
            EXPECT_FALSE(Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT.differentiable);
            EXPECT_FALSE(Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT.differentiable);
            EXPECT_FALSE(Autograd::has_rule(Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT.id));
            EXPECT_FALSE(Autograd::has_rule(Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT.id));
        }

        /** @brief Tests that the two static proofs of the flags hold at run time too. */
        TEST(Small_RuleTable, TheStaticProofsOfTheFlagsHoldAtRunTimeToo)
        {
            const SushiRuntime::span<const Ops::OpName> table(Ops::K_ALL.data(), Ops::K_ALL.size());

            EXPECT_TRUE(Autograd::every_differentiable_op_has_a_rule(table));
            EXPECT_TRUE(Autograd::no_gradient_barrier_has_a_rule(table));
        }

        /** @brief Tests that the root, logarithm and power have rules and the argmax has none. */
        TEST(Small_RuleTable, TheRootTheLogarithmAndThePowerHaveRulesAndTheArgmaxHasNone)
        {
            EXPECT_TRUE(Ops::K_SQRT.differentiable);
            EXPECT_TRUE(Ops::K_LOG.differentiable);
            EXPECT_TRUE(Ops::K_POW.differentiable);
            EXPECT_TRUE(Autograd::has_rule(Ops::K_SQRT.id));
            EXPECT_TRUE(Autograd::has_rule(Ops::K_LOG.id));
            EXPECT_TRUE(Autograd::has_rule(Ops::K_POW.id));
            EXPECT_TRUE(Ops::K_ARGMAX_ROWS.blocks_gradient);
            EXPECT_FALSE(Autograd::has_rule(Ops::K_ARGMAX_ROWS.id));
        }

        /** @brief Tests that the bilinear forward is ruled and its gradient is not. */
        TEST(Small_RuleTable, TheBilinearForwardHasItsRuleAndItsGradientHasNone)
        {
            EXPECT_TRUE(Ops::K_UPSAMPLE2D_BILINEAR.differentiable);
            EXPECT_TRUE(Autograd::has_rule(Ops::K_UPSAMPLE2D_BILINEAR.id));
            EXPECT_FALSE(Ops::K_UPSAMPLE2D_BILINEAR_BACKWARD.differentiable);
            EXPECT_FALSE(Autograd::has_rule(Ops::K_UPSAMPLE2D_BILINEAR_BACKWARD.id));
        }

        /** @brief Tests that the broadcasts, the sum along and the along rows have rules. */
        TEST(Small_RuleTable, TheBroadcastsTheSumAlongAndTheAlongRowsHaveRules)
        {
            EXPECT_EQ(Autograd::K_BROADCAST_RULES.size(), 5u);
            expect_differentiable_with_rule(Ops::K_BROADCAST_ADD);
            expect_differentiable_with_rule(Ops::K_BROADCAST_SUB);
            expect_differentiable_with_rule(Ops::K_BROADCAST_MUL);
            expect_differentiable_with_rule(Ops::K_BROADCAST_DIV);
            expect_differentiable_with_rule(Ops::K_SUM_ALONG);
            expect_differentiable_with_rule(Ops::K_TAKE_ALONG);
            expect_differentiable_with_rule(Ops::K_PUT_ALONG);
        }
    }
}
