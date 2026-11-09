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
 * @brief Tests that each identity of the operation table has one lowering row or is unlowered.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/unlowered.hpp>
#include <support/test_names.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Runs once per identity of K_UNLOWERED. */
        class Small_UnloweredIdentity : public ::testing::TestWithParam<OpID>
        {

        };

        /**
         * @brief Tests that every identity has one row, unless it is listed as unlowered, and
         *        that the rows plus the unlowered identities add up to the operation table.
         */
        TEST(Small_LoweringTable, EveryIdentityIsLoweredOrListedAsUnlowered)
        {
            EXPECT_TRUE(Graph::every_identity_is_accounted_for(
                SushiRuntime::span<const Ops::OpName>(Ops::K_ALL.data(), Ops::K_ALL.size())));
            EXPECT_EQ(Graph::K_LOWERINGS.size() + Graph::K_UNLOWERED.size(), Ops::K_ALL.size());
        }

        /** @brief Tests that an identity of K_UNLOWERED has no row. */
        TEST_P(Small_UnloweredIdentity, TheUnloweredIdentitiesHaveNoRow)
        {
            const OpID id = GetParam();

            EXPECT_TRUE(Graph::find_lowering(id) == nullptr) << Ops::name_of(id);
            EXPECT_EQ(Graph::lowering_row_count(id), 0u) << Ops::name_of(id);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_UnloweredIdentity,
                                 ::testing::ValuesIn(Graph::K_UNLOWERED),
                                 [](const ::testing::TestParamInfo<OpID>& info)
                                 {
                                     return parameter_name(Ops::name_of(info.param));
                                 });

    } // namespace Test
} // namespace SushiAI
