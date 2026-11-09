/**************************************************************************/
/* test_parameter_groups.cpp                                              */
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
 * @file test_parameter_groups.cpp
 * @brief Tests that weight decay is decided by a parameter's role, at compile time.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/train/parameter_groups.hpp>

using namespace SushiAI;
using NN::ParameterRole;

static_assert(Train::takes_weight_decay(ParameterRole::Weight),
              "the decision is a constant expression an optimizer may rely on");

namespace
{
    /** @brief Holds a parameter role and whether it takes weight decay. */
    struct RoleRow
    {
        std::string name;       /**< The row's name. */
        ParameterRole role;     /**< The role asked. */
        bool decays;            /**< Whether it takes decay. */
    };

    /** @brief Asks the decision for the role of each row. */
    class Small_ParameterGroups : public ::testing::TestWithParam<RoleRow>
    {

    };
} // namespace

/** @brief Tests that the weight is the one role of four that takes decay. */
TEST_P(Small_ParameterGroups, TheWeightIsTheOnlyRoleThatDecays)
{
    EXPECT_EQ(Train::takes_weight_decay(GetParam().role), GetParam().decays);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ParameterGroups,
    ::testing::Values(RoleRow{"Weight", ParameterRole::Weight, true},
                      RoleRow{"Bias", ParameterRole::Bias, false},
                      RoleRow{"Scale", ParameterRole::Scale, false},
                      RoleRow{"Embedding", ParameterRole::Embedding, false}),
    [](const ::testing::TestParamInfo<RoleRow>& info) { return info.param.name; });
