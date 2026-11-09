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
 * @file test_fusion_selector.cpp
 * @brief Tests the device fusion selector on device profiles written by hand.
 * @author Mustafa Garip
 */

#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <support/test_names.hpp>

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

namespace
{
    /** @brief Runs once per fused form that a device with doubles takes whole. */
    class Small_FusionSelectorForm : public ::testing::TestWithParam<FusedForm>
    {
    };
} // namespace

/** @brief Tests that a match with no device fact against it is taken whole. */
TEST_P(Small_FusionSelectorForm, AMatchIsTakenWholeWhenNothingArguesAgainstIt)
{
    const FusedForm form = GetParam();
    const DeviceFusionSelector selector(fp64_device(), SushiBLAS::ExecutionMode::Auto);

    const FusionChoice choice = selector.choose(query_of(form));

    EXPECT_EQ(choice.form, form);
    EXPECT_FALSE(choice.reason.empty());
}

INSTANTIATE_TEST_SUITE_P(, Small_FusionSelectorForm,
                         ::testing::Values(FusedForm::GEMMBias, FusedForm::GEMMBiasGELU,
                                           FusedForm::ExpSubScale),
                         [](const ::testing::TestParamInfo<FusedForm>& info)
                         {
                             return ::SushiAI::Test::parameter_name(Graph::fused_form_name(info.param));
                         });

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

/** @brief Tests that the reference mode declines a GEMM epilogue. */
TEST(Small_FusionSelector, TheReferenceModeDeclinesAGEMMEpilogue)
{
    const DeviceFusionSelector selector(fp64_device(), SushiBLAS::ExecutionMode::ForceReference);

    EXPECT_EQ(selector.choose(query_of(FusedForm::GEMMBiasTanh)).form, FusedForm::None);
}

/** @brief Tests that the reference mode keeps the elementwise fused form. */
TEST(Small_FusionSelector, TheReferenceModeKeepsTheElementwiseForm)
{
    const DeviceFusionSelector selector(fp64_device(), SushiBLAS::ExecutionMode::ForceReference);

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
