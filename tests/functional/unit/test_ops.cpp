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
 * @file test_ops.cpp
 * @brief Tests that the operation table is complete and that the reverse lookup resolves names.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <set>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/core/ops.hpp>

namespace Ops = SushiAI::Ops;

TEST(Unit_Ops, EveryIdentityHashesDistinctly)
{
    std::set<std::uint32_t> seen;
    for (const Ops::OpName& entry : Ops::K_ALL)
        EXPECT_TRUE(seen.insert(entry.id.value).second)
            << "duplicate OpID for " << entry.name;

    EXPECT_EQ(seen.size(), Ops::K_ALL.size());
}

TEST(Unit_Ops, EveryNameCarriesTheLibraryPrefix)
{
    // rfind(prefix, 0) matches at position 0 only (C++17); see tests/functional/unit/README.md.
    constexpr std::string_view K_PREFIX = "sushiai.";
    for (const Ops::OpName& entry : Ops::K_ALL)
        EXPECT_EQ(entry.name.rfind(K_PREFIX, 0), 0u) << entry.name;
}

TEST(Unit_Ops, NameOfRoundTripsEveryIdentity)
{
    for (const Ops::OpName& entry : Ops::K_ALL)
        EXPECT_EQ(Ops::name_of(entry.id), entry.name);
}

TEST(Unit_Ops, NameOfReportsUnknownForAForeignIdentity)
{
    const SushiAI::OpID foreign(0xDEADBEEFu);
    EXPECT_EQ(Ops::name_of(foreign), "sushiai.unknown");
}

TEST(Unit_Ops, TheTableContainsEveryDeclaredIdentity)
{
    // Membership in ALL is checked by name; see tests/functional/unit/README.md.
    const std::array expected = {
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
    };

    EXPECT_EQ(Ops::K_ALL.size(), expected.size());

    for (std::string_view name : expected)
    {
        bool found = false;
        for (const Ops::OpName& entry : Ops::K_ALL)
            if (entry.name == name) { found = true; break; }
        EXPECT_TRUE(found) << name << " is missing from Ops::ALL";
    }
}

TEST(Unit_Ops, EverySpatialIdentityIsListedInAll)
{
    // ALL is what the collision proof and name_of() read; see tests/functional/unit/README.md.
    const std::array<SushiAI::OpID, 20> declared = {
        Ops::K_RESHAPE.id, Ops::K_LAYOUT_CONVERT.id,
        Ops::K_IM2COL.id, Ops::K_COL2IM.id,
        Ops::K_MAXPOOL2D.id, Ops::K_MAXPOOL2D_BACKWARD.id,
        Ops::K_AVGPOOL2D.id, Ops::K_AVGPOOL2D_BACKWARD.id,
        Ops::K_SPATIAL_MEAN.id, Ops::K_SPATIAL_MEAN_BACKWARD.id,
        Ops::K_UPSAMPLE2D.id, Ops::K_UPSAMPLE2D_BACKWARD.id,
        Ops::K_CONCAT_CHANNELS.id, Ops::K_SLICE_CHANNELS.id,
        Ops::K_BATCHNORM2D.id, Ops::K_BATCHNORM2D_BACKWARD.id,
        Ops::K_CHANNEL_AFFINE.id,
        Ops::K_BATCHNORM_UPDATE_STATS.id,
        Ops::K_LEAKY_RELU.id, Ops::K_LEAKY_RELU_BACKWARD.id,
    };

    for (SushiAI::OpID id : declared)
        EXPECT_NE(Ops::name_of(id), "sushiai.unknown")
            << "an identity was declared but not added to Ops::ALL";
}

TEST(Unit_Ops, ThereIsNoConv2dIdentity)
{
    // Convolution is IM2COL + MATMUL in the IR (CV-1); see tests/functional/unit/README.md.
    EXPECT_EQ(Ops::name_of(Ops::op("sushiai.conv2d").id), "sushiai.unknown");
}

TEST(Unit_Ops, LeakyReluDifferentiatesAtItsOutput)
{
    // For alpha > 0 the output's sign equals the input's; see tests/functional/unit/README.md.
    const Ops::Activation* entry = Ops::activation_of(Ops::K_LEAKY_RELU.id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->backward, Ops::K_LEAKY_RELU_BACKWARD.id);
    EXPECT_EQ(entry->gradient_source, Ops::GradientSource::Output);
    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::K_LEAKY_RELU.id));
}

/** @brief Tests that every elementwise, selection and SiLU identity resolves through ALL. */
TEST(Unit_Ops, EveryElementwiseAndSelectionIdentityIsListedInAll)
{
    const std::array<SushiAI::OpID, 11> declared = {
        Ops::K_DIV.id, Ops::K_MINIMUM.id, Ops::K_MAXIMUM.id,
        Ops::K_ABS.id, Ops::K_ATAN.id,
        Ops::K_LESS_EQUAL.id, Ops::K_SELECT.id,
        Ops::K_SILU.id, Ops::K_SILU_BACKWARD.id,
        Ops::K_SOFTPLUS.id, Ops::K_SOFTPLUS_BACKWARD.id,
    };

    for (SushiAI::OpID id : declared)
        EXPECT_NE(Ops::name_of(id), "sushiai.unknown")
            << "an identity was declared but not added to Ops::ALL";
}

/** @brief Tests that the eight forward identities are flagged differentiable and three are not. */
TEST(Unit_Ops, TheComparisonAndTheActivationGradientsAreNotFlaggedDifferentiable)
{
    for (const Ops::OpName& entry : {Ops::K_DIV, Ops::K_MINIMUM, Ops::K_MAXIMUM, Ops::K_ABS, Ops::K_ATAN,
                                     Ops::K_SOFTPLUS, Ops::K_SELECT, Ops::K_SILU})
        EXPECT_TRUE(entry.differentiable) << entry.name;

    EXPECT_FALSE(Ops::K_LESS_EQUAL.differentiable);
    EXPECT_FALSE(Ops::K_SILU_BACKWARD.differentiable);
    EXPECT_FALSE(Ops::K_SOFTPLUS_BACKWARD.differentiable);
}

/** @brief Tests that exactly ZERO, CONSTANT, LESS_EQUAL and STOP_GRADIENT are gradient barriers. */
TEST(Unit_Ops, TheGradientBarriersAreTheFourIdentitiesThatPassNoGradient)
{
    const std::array<SushiAI::OpID, 4> barriers = {
        Ops::K_ZERO.id, Ops::K_CONSTANT.id, Ops::K_LESS_EQUAL.id, Ops::K_STOP_GRADIENT.id,
    };

    for (SushiAI::OpID id : barriers)
        EXPECT_TRUE(Ops::is_gradient_barrier(id)) << Ops::name_of(id);

    std::size_t flagged = 0;
    for (const Ops::OpName& entry : Ops::K_ALL)
    {
        EXPECT_EQ(Ops::is_gradient_barrier(entry.id), entry.blocks_gradient) << entry.name;
        flagged += entry.blocks_gradient ? 1u : 0u;
    }

    EXPECT_EQ(flagged, barriers.size());
}

/** @brief Tests that no identity is flagged both differentiable and a gradient barrier. */
TEST(Unit_Ops, NoGradientBarrierIsFlaggedDifferentiable)
{
    for (const Ops::OpName& entry : Ops::K_ALL)
        EXPECT_FALSE(entry.differentiable && entry.blocks_gradient) << entry.name;
}

/** @brief Tests that the identities that pass a gradient or throw on one are not barriers. */
TEST(Unit_Ops, LeafCastAndTheGradientOperationsAreNotGradientBarriers)
{
    for (const Ops::OpName& entry : {Ops::K_LEAF, Ops::K_CAST, Ops::K_DYNAMIC_SCALE, Ops::K_LOG_SOFTMAX,
                                     Ops::K_RELU_BACKWARD, Ops::K_SILU_BACKWARD,
                                     Ops::K_SOFTPLUS_BACKWARD, Ops::K_MAXPOOL2D_BACKWARD,
                                     Ops::K_BATCHNORM2D_BACKWARD, Ops::K_BATCHNORM_UPDATE_STATS,
                                     Ops::K_BATCHNORM_FOLD,
                                     Ops::K_GEMM_BIAS, Ops::K_EXP_SUB_SCALE, Ops::K_ADAMW_UPDATE})
        EXPECT_FALSE(Ops::is_gradient_barrier(entry.id)) << entry.name;

    EXPECT_FALSE(Ops::is_gradient_barrier(SushiAI::OpID(0xDEADBEEFu)));
}

/** @brief Tests that SILU's table row differentiates at the forward input and does not fold. */
TEST(Unit_Ops, SiluDifferentiatesAtItsInput)
{
    const Ops::Activation* entry = Ops::activation_of(Ops::K_SILU.id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->backward, Ops::K_SILU_BACKWARD.id);
    EXPECT_EQ(entry->gradient_source, Ops::GradientSource::Input);
    EXPECT_FALSE(Ops::activation_survives_fusion(Ops::K_SILU.id));
}

/** @brief Tests that SOFTPLUS's table row differentiates at the forward input and does not fold. */
TEST(Unit_Ops, SoftplusDifferentiatesAtItsInput)
{
    const Ops::Activation* entry = Ops::activation_of(Ops::K_SOFTPLUS.id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->backward, Ops::K_SOFTPLUS_BACKWARD.id);
    EXPECT_EQ(entry->gradient_source, Ops::GradientSource::Input);
    EXPECT_FALSE(Ops::activation_survives_fusion(Ops::K_SOFTPLUS.id));
}

/** @brief Tests that is_activation_backward accepts exactly the gradient column of ACTIVATIONS. */
TEST(Unit_Ops, IsActivationBackwardAcceptsOnlyTheTablesGradientColumn)
{
    for (const Ops::Activation& entry : Ops::K_ACTIVATIONS)
    {
        EXPECT_TRUE(Ops::is_activation_backward(entry.backward))
            << Ops::name_of(entry.backward);
        EXPECT_FALSE(Ops::is_activation_backward(entry.forward))
            << Ops::name_of(entry.forward);
    }

    std::size_t accepted = 0;
    for (const Ops::OpName& entry : Ops::K_ALL)
        accepted += Ops::is_activation_backward(entry.id) ? 1u : 0u;

    EXPECT_EQ(accepted, Ops::K_ACTIVATIONS.size());
}

/** @brief Tests that BATCHNORM_FOLD is in the table and claims neither a rule nor a barrier. */
TEST(Unit_Ops, BatchNormFoldIsListedAndClaimsNoRule)
{
    EXPECT_EQ(Ops::name_of(Ops::K_BATCHNORM_FOLD.id), "sushiai.batchnorm_fold");
    EXPECT_FALSE(Ops::K_BATCHNORM_FOLD.differentiable);
    EXPECT_FALSE(Ops::K_BATCHNORM_FOLD.blocks_gradient);
}
