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

#include <set>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/core/ops.hpp>

namespace Ops = SushiAI::Ops;

TEST(Unit_Ops, EveryIdentityHashesDistinctly)
{
    std::set<std::uint32_t> seen;
    for (const Ops::OpName& entry : Ops::ALL)
        EXPECT_TRUE(seen.insert(entry.id.value).second)
            << "duplicate OpID for " << entry.name;

    EXPECT_EQ(seen.size(), Ops::ALL.size());
}

TEST(Unit_Ops, EveryNameCarriesTheLibraryPrefix)
{
    // rfind(prefix, 0) matches at position 0 only (C++17); see tests/functional/unit/README.md.
    constexpr std::string_view PREFIX = "sushiai.";
    for (const Ops::OpName& entry : Ops::ALL)
        EXPECT_EQ(entry.name.rfind(PREFIX, 0), 0u) << entry.name;
}

TEST(Unit_Ops, NameOfRoundTripsEveryIdentity)
{
    for (const Ops::OpName& entry : Ops::ALL)
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
    };

    EXPECT_EQ(Ops::ALL.size(), expected.size());

    for (std::string_view name : expected)
    {
        bool found = false;
        for (const Ops::OpName& entry : Ops::ALL)
            if (entry.name == name) { found = true; break; }
        EXPECT_TRUE(found) << name << " is missing from Ops::ALL";
    }
}

TEST(Unit_Ops, EverySpatialIdentityIsListedInAll)
{
    // ALL is what the collision proof and name_of() read; see tests/functional/unit/README.md.
    const std::array<SushiAI::OpID, 20> declared = {
        Ops::RESHAPE.id, Ops::LAYOUT_CONVERT.id,
        Ops::IM2COL.id, Ops::COL2IM.id,
        Ops::MAXPOOL2D.id, Ops::MAXPOOL2D_BACKWARD.id,
        Ops::AVGPOOL2D.id, Ops::AVGPOOL2D_BACKWARD.id,
        Ops::SPATIAL_MEAN.id, Ops::SPATIAL_MEAN_BACKWARD.id,
        Ops::UPSAMPLE2D.id, Ops::UPSAMPLE2D_BACKWARD.id,
        Ops::CONCAT_CHANNELS.id, Ops::SLICE_CHANNELS.id,
        Ops::BATCHNORM2D.id, Ops::BATCHNORM2D_BACKWARD.id,
        Ops::CHANNEL_AFFINE.id,
        Ops::BATCHNORM_UPDATE_STATS.id,
        Ops::LEAKY_RELU.id, Ops::LEAKY_RELU_BACKWARD.id,
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
    const Ops::Activation* entry = Ops::activation_of(Ops::LEAKY_RELU.id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->backward, Ops::LEAKY_RELU_BACKWARD.id);
    EXPECT_EQ(entry->gradient_source, Ops::GradientSource::OUTPUT);
    EXPECT_TRUE(Ops::activation_survives_fusion(Ops::LEAKY_RELU.id));
}
