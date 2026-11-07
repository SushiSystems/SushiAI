/**************************************************************************/
/* test_ops.cpp                                                           */
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
 * @file test_ops.cpp
 * @brief Tests that the operation table is complete and that the reverse lookup resolves names.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/ops/ops.hpp>
#include <support/test_names.hpp>

namespace Ops = SushiAI::Ops;

namespace
{
    using SushiAI::Test::parameter_name;

    /** @brief Names a row after the operation it holds. */
    std::string operation_row_name(const ::testing::TestParamInfo<Ops::OpName>& info)
    {
        return parameter_name(info.param.name);
    }

    /** @brief Counts the distinct identity values among the rows of Ops::K_ALL. */
    std::size_t count_distinct_identities()
    {
        std::set<std::uint32_t> seen;
        for (const Ops::OpName& entry : Ops::K_ALL)
            seen.insert(entry.id.value);
        return seen.size();
    }

    /** @brief Counts the rows of Ops::K_ALL that carry the blocks_gradient flag. */
    std::size_t count_flagged_gradient_barriers()
    {
        std::size_t flagged = 0;
        for (const Ops::OpName& entry : Ops::K_ALL)
            flagged += entry.blocks_gradient ? 1u : 0u;
        return flagged;
    }

    /** @brief Counts the rows of Ops::K_ALL that is_activation_backward accepts. */
    std::size_t count_accepted_activation_backwards()
    {
        std::size_t accepted = 0;
        for (const Ops::OpName& entry : Ops::K_ALL)
            accepted += Ops::is_activation_backward(entry.id) ? 1u : 0u;
        return accepted;
    }

    /** @brief Returns whether a row of Ops::K_ALL carries the name @p name. */
    bool table_holds_name(std::string_view name)
    {
        for (const Ops::OpName& entry : Ops::K_ALL)
        {
            if (entry.name == name)
                return true;
        }
        return false;
    }

    /** @brief Lists the names every identity declared in ops.hpp carries. */
    const std::vector<std::string_view>& declared_names()
    {
        static const std::vector<std::string_view> names = {
            std::string_view("sushiai.leaf"),
            std::string_view("sushiai.accumulate"),
            std::string_view("sushiai.cast"),
            std::string_view("sushiai.matmul"),
            std::string_view("sushiai.bias_add"),
            std::string_view("sushiai.add"),
            std::string_view("sushiai.sub"),
            std::string_view("sushiai.mul"),
            std::string_view("sushiai.scale"),
            std::string_view("sushiai.dynamic_scale"),
            std::string_view("sushiai.exp"),
            std::string_view("sushiai.zero"),
            std::string_view("sushiai.relu"),
            std::string_view("sushiai.gelu"),
            std::string_view("sushiai.sigmoid"),
            std::string_view("sushiai.tanh"),
            std::string_view("sushiai.relu_backward"),
            std::string_view("sushiai.gelu_backward"),
            std::string_view("sushiai.sigmoid_backward"),
            std::string_view("sushiai.tanh_backward"),
            std::string_view("sushiai.log_softmax"),
            std::string_view("sushiai.nll_loss"),
            std::string_view("sushiai.sum_rows"),
            std::string_view("sushiai.optim.adamw"),
            std::string_view("sushiai.optim.sgd"),
            std::string_view("sushiai.optim.clip_scale"),
            std::string_view("sushiai.fused.gemm_bias"),
            std::string_view("sushiai.fused.gemm_bias_relu"),
            std::string_view("sushiai.fused.gemm_bias_sigmoid"),
            std::string_view("sushiai.fused.gemm_bias_tanh"),
            std::string_view("sushiai.fused.gemm_bias_gelu"),
            std::string_view("sushiai.fused.exp_sub_scale"),
            std::string_view("sushiai.leaky_relu"),
            std::string_view("sushiai.leaky_relu_backward"),
            std::string_view("sushiai.reshape"),
            std::string_view("sushiai.layout_convert"),
            std::string_view("sushiai.im2col"),
            std::string_view("sushiai.col2im"),
            std::string_view("sushiai.depthwise_conv2d"),
            std::string_view("sushiai.depthwise_conv2d_backward_input"),
            std::string_view("sushiai.depthwise_conv2d_backward_weight"),
            std::string_view("sushiai.maxpool2d"),
            std::string_view("sushiai.maxpool2d_backward"),
            std::string_view("sushiai.avgpool2d"),
            std::string_view("sushiai.avgpool2d_backward"),
            std::string_view("sushiai.spatial_mean"),
            std::string_view("sushiai.spatial_mean_backward"),
            std::string_view("sushiai.upsample2d"),
            std::string_view("sushiai.upsample2d_backward"),
            std::string_view("sushiai.concat_channels"),
            std::string_view("sushiai.slice_channels"),
            std::string_view("sushiai.batchnorm2d"),
            std::string_view("sushiai.batchnorm2d_backward"),
            std::string_view("sushiai.channel_affine"),
            std::string_view("sushiai.batchnorm_update_stats"),
            std::string_view("sushiai.div"),
            std::string_view("sushiai.minimum"),
            std::string_view("sushiai.maximum"),
            std::string_view("sushiai.abs"),
            std::string_view("sushiai.atan"),
            std::string_view("sushiai.softplus"),
            std::string_view("sushiai.less_equal"),
            std::string_view("sushiai.select"),
            std::string_view("sushiai.silu"),
            std::string_view("sushiai.silu_backward"),
            std::string_view("sushiai.softplus_backward"),
            std::string_view("sushiai.stop_gradient"),
            std::string_view("sushiai.constant"),
            std::string_view("sushiai.batchnorm_fold"),
            std::string_view("sushiai.gather_rows"),
            std::string_view("sushiai.scatter_add_rows"),
            std::string_view("sushiai.causal_mask"),
            std::string_view("sushiai.layer_norm"),
            std::string_view("sushiai.layer_norm_backward"),
            std::string_view("sushiai.dropout"),
            std::string_view("sushiai.log_softmax_backward"),
        };
        return names;
    }

    /** @brief Lists the identities declared in ops.hpp that Ops::K_ALL must resolve by name. */
    std::vector<Ops::OpName> declared_identities()
    {
        return {
            Ops::K_RESHAPE, Ops::K_LAYOUT_CONVERT,
            Ops::K_IM2COL, Ops::K_COL2IM,
            Ops::K_DEPTHWISE_CONV2D, Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT,
            Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT,
            Ops::K_MAXPOOL2D, Ops::K_MAXPOOL2D_BACKWARD,
            Ops::K_AVGPOOL2D, Ops::K_AVGPOOL2D_BACKWARD,
            Ops::K_SPATIAL_MEAN, Ops::K_SPATIAL_MEAN_BACKWARD,
            Ops::K_UPSAMPLE2D, Ops::K_UPSAMPLE2D_BACKWARD,
            Ops::K_CONCAT_CHANNELS, Ops::K_SLICE_CHANNELS,
            Ops::K_BATCHNORM2D, Ops::K_BATCHNORM2D_BACKWARD,
            Ops::K_CHANNEL_AFFINE,
            Ops::K_BATCHNORM_UPDATE_STATS,
            Ops::K_LEAKY_RELU, Ops::K_LEAKY_RELU_BACKWARD,
            Ops::K_DIV, Ops::K_MINIMUM, Ops::K_MAXIMUM,
            Ops::K_ABS, Ops::K_ATAN,
            Ops::K_LESS_EQUAL, Ops::K_SELECT,
            Ops::K_SILU, Ops::K_SILU_BACKWARD,
            Ops::K_SOFTPLUS, Ops::K_SOFTPLUS_BACKWARD,
        };
    }

    /** @brief Lists the identities that block the gradient. */
    std::vector<Ops::OpName> gradient_barriers()
    {
        return {Ops::K_ZERO, Ops::K_CONSTANT, Ops::K_LESS_EQUAL, Ops::K_STOP_GRADIENT};
    }

    /** @brief Lists the identities that pass a gradient or throw on one, so are no barrier. */
    std::vector<Ops::OpName> gradient_passers()
    {
        return {Ops::K_LEAF, Ops::K_CAST, Ops::K_DYNAMIC_SCALE, Ops::K_LOG_SOFTMAX,
                Ops::K_LOG_SOFTMAX_BACKWARD, Ops::K_RELU_BACKWARD, Ops::K_SILU_BACKWARD,
                Ops::K_SOFTPLUS_BACKWARD, Ops::K_MAXPOOL2D_BACKWARD,
                Ops::K_BATCHNORM2D_BACKWARD, Ops::K_BATCHNORM_UPDATE_STATS,
                Ops::K_BATCHNORM_FOLD, Ops::K_GEMM_BIAS, Ops::K_EXP_SUB_SCALE,
                Ops::K_ADAMW_UPDATE};
    }

    /** @brief Names an operation and the differentiable flag its row must carry. */
    struct DifferentiableCase
    {
        /** @brief Holds the operation. */
        Ops::OpName operation;

        /** @brief Holds the flag the row of the operation carries. */
        bool differentiable;
    };

    /** @brief Names an activation's forward and gradient operations. */
    struct InputActivationCase
    {
        /** @brief Holds the forward operation. */
        Ops::OpName forward;

        /** @brief Holds the gradient operation of the forward one. */
        Ops::OpName backward;
    };

    /** @brief Runs each row of Ops::K_ALL. */
    class Small_OpsEntry : public ::testing::TestWithParam<Ops::OpName>
    {

    };

    /** @brief Runs each identity declared in ops.hpp. */
    class Small_OpsDeclaredIdentity : public ::testing::TestWithParam<Ops::OpName>
    {

    };

    /** @brief Runs each name declared in ops.hpp. */
    class Small_OpsDeclaredName : public ::testing::TestWithParam<std::string_view>
    {

    };

    /** @brief Runs each identity that blocks the gradient. */
    class Small_OpsGradientBarrier : public ::testing::TestWithParam<Ops::OpName>
    {

    };

    /** @brief Runs each identity that passes a gradient or throws on one. */
    class Small_OpsGradientPasser : public ::testing::TestWithParam<Ops::OpName>
    {

    };

    /** @brief Runs each identity whose differentiable flag a row states. */
    class Small_OpsDifferentiableFlag : public ::testing::TestWithParam<DifferentiableCase>
    {

    };

    /** @brief Runs each activation that differentiates at its input. */
    class Small_OpsInputActivation : public ::testing::TestWithParam<InputActivationCase>
    {

    };

    /** @brief Runs each row of Ops::K_ACTIVATIONS. */
    class Small_OpsActivationRow : public ::testing::TestWithParam<Ops::Activation>
    {

    };
} // namespace

/** @brief Tests that no two rows of Ops::K_ALL share an identity value. */
TEST(Small_Ops, EveryIdentityHashesDistinctly)
{
    EXPECT_EQ(count_distinct_identities(), Ops::K_ALL.size());
}

/** @brief Tests that the name of a row starts with the library prefix. */
TEST_P(Small_OpsEntry, EveryNameCarriesTheLibraryPrefix)
{
    // rfind(prefix, 0) matches at position 0 only (C++17); see tests/core/ops/README.md.
    constexpr std::string_view K_PREFIX = "sushiai.";

    EXPECT_EQ(GetParam().name.rfind(K_PREFIX, 0), 0u);
}

/** @brief Tests that name_of returns the name of a row for its identity. */
TEST_P(Small_OpsEntry, NameOfRoundTripsEveryIdentity)
{
    EXPECT_EQ(Ops::name_of(GetParam().id), GetParam().name);
}

/** @brief Tests that is_gradient_barrier returns the blocks_gradient flag of a row. */
TEST_P(Small_OpsEntry, IsGradientBarrierEqualsTheRowFlag)
{
    EXPECT_EQ(Ops::is_gradient_barrier(GetParam().id), GetParam().blocks_gradient);
}

/** @brief Tests that a row is not flagged both differentiable and a gradient barrier. */
TEST_P(Small_OpsEntry, NoGradientBarrierIsFlaggedDifferentiable)
{
    EXPECT_FALSE(GetParam().differentiable && GetParam().blocks_gradient);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_OpsEntry, ::testing::ValuesIn(Ops::K_ALL),
                         operation_row_name);

/** @brief Tests that name_of reports sushiai.unknown for an identity no row holds. */
TEST(Small_Ops, NameOfReportsUnknownForAForeignIdentity)
{
    const SushiAI::OpID foreign(0xDEADBEEFu);
    EXPECT_EQ(Ops::name_of(foreign), "sushiai.unknown");
}

/** @brief Tests that Ops::K_ALL holds as many rows as ops.hpp declares names. */
TEST(Small_Ops, TheTableHoldsAsManyRowsAsDeclaredNames)
{
    EXPECT_EQ(Ops::K_ALL.size(), declared_names().size());
}

/** @brief Tests that a declared name has a row in Ops::K_ALL. */
TEST_P(Small_OpsDeclaredName, TheTableContainsEveryDeclaredIdentity)
{
    // Membership in ALL is checked by name; see tests/core/ops/README.md.
    EXPECT_TRUE(table_holds_name(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_OpsDeclaredName, ::testing::ValuesIn(declared_names()),
                         [](const ::testing::TestParamInfo<std::string_view>& info)
                         {
                             return parameter_name(info.param);
                         });

/** @brief Tests that name_of resolves a declared identity, so Ops::K_ALL lists it. */
TEST_P(Small_OpsDeclaredIdentity, EveryDeclaredIdentityIsListedInAll)
{
    // ALL is what the collision proof and name_of() read; see tests/core/ops/README.md.
    EXPECT_NE(Ops::name_of(GetParam().id), "sushiai.unknown");
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_OpsDeclaredIdentity,
                         ::testing::ValuesIn(declared_identities()), operation_row_name);

/** @brief Tests that no CONV2D identity exists, since convolution is IM2COL plus MATMUL. */
TEST(Small_Ops, ThereIsNoConv2dIdentity)
{
    // Convolution is IM2COL + MATMUL in the IR (CV-1); see tests/core/ops/README.md.
    EXPECT_EQ(Ops::name_of(Ops::op("sushiai.conv2d").id), "sushiai.unknown");
}

/** @brief Tests that leaky ReLU differentiates at its output and survives fusion. */
TEST(Small_Ops, LeakyReLUDifferentiatesAtItsOutput)
{
    // For alpha > 0 the output's sign equals the input's; see tests/core/ops/README.md.
    const Ops::Activation* entry = Ops::activation_of(Ops::K_LEAKY_RELU.id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->backward, Ops::K_LEAKY_RELU_BACKWARD.id);
    EXPECT_EQ(entry->gradient_source, Ops::GradientSource::Output);
    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::K_LEAKY_RELU.id));
}

/** @brief Tests that the row of an operation carries the differentiable flag the case states. */
TEST_P(Small_OpsDifferentiableFlag, AnIdentityCarriesItsDifferentiableFlag)
{
    EXPECT_EQ(GetParam().operation.differentiable, GetParam().differentiable);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_OpsDifferentiableFlag,
    ::testing::Values(DifferentiableCase{Ops::K_DIV, true}, DifferentiableCase{Ops::K_MINIMUM, true},
                      DifferentiableCase{Ops::K_MAXIMUM, true}, DifferentiableCase{Ops::K_ABS, true},
                      DifferentiableCase{Ops::K_ATAN, true},
                      DifferentiableCase{Ops::K_SOFTPLUS, true},
                      DifferentiableCase{Ops::K_SELECT, true}, DifferentiableCase{Ops::K_SILU, true},
                      DifferentiableCase{Ops::K_LESS_EQUAL, false},
                      DifferentiableCase{Ops::K_SILU_BACKWARD, false},
                      DifferentiableCase{Ops::K_SOFTPLUS_BACKWARD, false}),
    [](const ::testing::TestParamInfo<DifferentiableCase>& info)
    {
        return parameter_name(info.param.operation.name);
    });

/** @brief Tests that is_gradient_barrier is true for an identity that passes no gradient. */
TEST_P(Small_OpsGradientBarrier, ABarrierIdentityIsAGradientBarrier)
{
    EXPECT_TRUE(Ops::is_gradient_barrier(GetParam().id));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_OpsGradientBarrier, ::testing::ValuesIn(gradient_barriers()),
                         operation_row_name);

/** @brief Tests that exactly four rows of Ops::K_ALL carry the blocks_gradient flag. */
TEST(Small_Ops, ExactlyFourIdentitiesBlockTheGradient)
{
    // A fifth barrier fails this count until gradient_barriers() names it.
    EXPECT_EQ(count_flagged_gradient_barriers(), gradient_barriers().size());
}

/** @brief Tests that an identity that passes a gradient or throws on one is not a barrier. */
TEST_P(Small_OpsGradientPasser, AGradientPasserIsNotAGradientBarrier)
{
    EXPECT_FALSE(Ops::is_gradient_barrier(GetParam().id));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_OpsGradientPasser, ::testing::ValuesIn(gradient_passers()),
                         operation_row_name);

/** @brief Tests that an identity no row holds is not a gradient barrier. */
TEST(Small_Ops, AForeignIdentityIsNotAGradientBarrier)
{
    EXPECT_FALSE(Ops::is_gradient_barrier(SushiAI::OpID(0xDEADBEEFu)));
}

/** @brief Tests that an activation's row differentiates at the forward input and does not fold. */
TEST_P(Small_OpsInputActivation, AnActivationDifferentiatesAtItsInputAndDoesNotSurviveFusion)
{
    const InputActivationCase& row = GetParam();

    const Ops::Activation* entry = Ops::activation_of(row.forward.id);

    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->backward, row.backward.id);
    EXPECT_EQ(entry->gradient_source, Ops::GradientSource::Input);
    EXPECT_FALSE(Ops::activation_survives_fusion(row.forward.id));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_OpsInputActivation,
    ::testing::Values(InputActivationCase{Ops::K_SILU, Ops::K_SILU_BACKWARD},
                      InputActivationCase{Ops::K_SOFTPLUS, Ops::K_SOFTPLUS_BACKWARD}),
    [](const ::testing::TestParamInfo<InputActivationCase>& info)
    {
        return parameter_name(info.param.forward.name);
    });

/** @brief Tests that is_activation_backward accepts the row's gradient and rejects its forward. */
TEST_P(Small_OpsActivationRow, IsActivationBackwardAcceptsTheGradientAndNotTheForward)
{
    EXPECT_TRUE(Ops::is_activation_backward(GetParam().backward));
    EXPECT_FALSE(Ops::is_activation_backward(GetParam().forward));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_OpsActivationRow, ::testing::ValuesIn(Ops::K_ACTIVATIONS),
                         [](const ::testing::TestParamInfo<Ops::Activation>& info)
                         {
                             return parameter_name(Ops::name_of(info.param.forward));
                         });

/** @brief Tests that is_activation_backward accepts as many identities as the table has rows. */
TEST(Small_Ops, IsActivationBackwardAcceptsAsManyIdentitiesAsTheTableHasRows)
{
    EXPECT_EQ(count_accepted_activation_backwards(), Ops::K_ACTIVATIONS.size());
}

/** @brief Tests that BATCHNORM_FOLD is neither differentiable nor a gradient barrier. */
TEST(Small_Ops, BatchNormFoldIsNeitherDifferentiableNorAGradientBarrier)
{
    EXPECT_FALSE(Ops::K_BATCHNORM_FOLD.differentiable);
    EXPECT_FALSE(Ops::K_BATCHNORM_FOLD.blocks_gradient);
}
