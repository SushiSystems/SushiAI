/**************************************************************************/
/* test_fusion_selector.cpp                                               */
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
 * @file test_fusion_selector.cpp
 * @brief Tests the device fusion selector on device profiles written by hand.
 * @author Mustafa Garip
 */

#include <string_view>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/fusion_selector.hpp>

using namespace SushiAI;
using Graph::DeviceFusionSelector;
using Graph::FusedForm;
using Graph::FusionChoice;
using Graph::FusionQuery;

namespace
{
    /** @brief Returns a profile of a device with native double precision. */
    SushiBLAS::Device::Profile fp64_device()
    {
        SushiBLAS::Device::Profile profile;
        profile.supports_fp64 = true;
        return profile;
    }

    /** @brief Returns a query for a FLOAT32 match of @p form over a [4, 5] result. */
    FusionQuery query_of(FusedForm form)
    {
        FusionQuery query;
        query.matched = form;
        query.rows = 4;
        query.columns = 5;
        return query;
    }
} // namespace

/** @brief Tests that a match with no device fact against it is taken whole. */
TEST(Small_FusionSelector, AMatchIsTakenWholeWhenNothingArguesAgainstIt)
{
    const DeviceFusionSelector selector(fp64_device(), SushiBLAS::ExecutionMode::Auto);

    for (const FusedForm form : {FusedForm::GEMMBias, FusedForm::GEMMBiasGELU,
                                 FusedForm::ExpSubScale})
    {
        const FusionChoice choice = selector.choose(query_of(form));
        EXPECT_EQ(choice.form, form) << Graph::fused_form_name(form);
        EXPECT_FALSE(choice.reason.empty()) << Graph::fused_form_name(form);
    }
}

/** @brief Tests that a FLOAT64 match on a device without native doubles is spelled unfused. */
TEST(Small_FusionSelector, ADoubleMatchOnADeviceWithoutDoublesIsDeclined)
{
    const DeviceFusionSelector selector(SushiBLAS::Device::Profile{},
                                        SushiBLAS::ExecutionMode::Auto);

    FusionQuery query = query_of(FusedForm::GEMMBiasReLU);
    query.dtype = DType::Float64;

    const FusionChoice choice = selector.choose(query);
    EXPECT_EQ(choice.form, FusedForm::None);
    EXPECT_NE(choice.reason.find("double precision"), std::string_view::npos);
}

/** @brief Tests that the reference mode declines a GEMM epilogue but keeps the elementwise form. */
TEST(Small_FusionSelector, TheReferenceModeDeclinesOnlyGEMMEpilogues)
{
    const DeviceFusionSelector selector(fp64_device(), SushiBLAS::ExecutionMode::ForceReference);

    EXPECT_EQ(selector.choose(query_of(FusedForm::GEMMBiasTanh)).form, FusedForm::None);
    EXPECT_EQ(selector.choose(query_of(FusedForm::ExpSubScale)).form, FusedForm::ExpSubScale);
}

/** @brief Tests that a GEMM epilogue whose bias is strided is declined. */
TEST(Small_FusionSelector, AStridedEpilogueOperandIsDeclined)
{
    const DeviceFusionSelector selector(fp64_device(), SushiBLAS::ExecutionMode::Auto);

    FusionQuery query = query_of(FusedForm::GEMMBias);
    query.operand_is_contiguous = false;

    const FusionChoice choice = selector.choose(query);
    EXPECT_EQ(choice.form, FusedForm::None);
    EXPECT_NE(choice.reason.find("contiguous"), std::string_view::npos);
}

/** @brief Tests that the selector keeps the profile and the mode it was built with. */
TEST(Small_FusionSelector, TheSelectorKeepsTheDeviceItReasonsAbout)
{
    const DeviceFusionSelector selector(fp64_device(), SushiBLAS::ExecutionMode::ForceSYCL);

    EXPECT_TRUE(selector.profile().supports_fp64);
    EXPECT_EQ(selector.mode(), SushiBLAS::ExecutionMode::ForceSYCL);
    EXPECT_EQ(selector.choose(FusionQuery{}).form, FusedForm::None);
}
