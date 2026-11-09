/**************************************************************************/
/* test_shape.cpp                                                         */
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
 * @file test_shape.cpp
 * @brief Tests the shape lowering table.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/shape.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
using Graph::LoweringRule;

namespace
{
    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        std::size_t index; /**< The position of the row in the table. */
        OpID op;           /**< The operation. */
        LoweringRule rule; /**< The row that lowers it. */
    };

    /** @brief Runs once per row the header names for the shape table. */
    class Small_ShapeRow : public ::testing::TestWithParam<ExpectedRow>
    {

    };

    /** @brief Lists the rows of the shape table in the order the header writes them. */
    const std::array<ExpectedRow, 4> K_EXPECTED_ROWS = {{
        {0, Ops::K_LEAF.id, &Graph::lower_leaf},
        {1, Ops::K_RESHAPE.id, &Graph::lower_reshape},
        {2, Ops::K_LAYOUT_CONVERT.id, &Graph::lower_layout_convert},
        {3, Ops::K_STOP_GRADIENT.id, &Graph::lower_stop_gradient},
    }};
} // namespace

/** @brief Tests that the table holds one row for each row the header names, and no other. */
TEST(Small_ShapeLowerings, TheTableHoldsExactlyTheRowsTheHeaderNames)
{
    EXPECT_EQ(Graph::K_SHAPE_LOWERINGS.size(), K_EXPECTED_ROWS.size());
}

/** @brief Tests that a row pairs its operation with the function the header names. */
TEST_P(Small_ShapeRow, EachRowPairsItsOperationWithItsLowering)
{
    const ExpectedRow row = GetParam();
    const std::string name(Ops::name_of(row.op));
    ASSERT_LT(row.index, Graph::K_SHAPE_LOWERINGS.size());
    const Graph::LoweringEntry& entry = Graph::K_SHAPE_LOWERINGS[row.index];

    EXPECT_EQ(entry.op, row.op) << name;
    EXPECT_EQ(entry.rule, row.rule) << name;
    EXPECT_EQ(entry.scratch, nullptr) << name;
    EXPECT_EQ(Graph::find_lowering(row.op), row.rule) << name;
}

INSTANTIATE_TEST_SUITE_P(, Small_ShapeRow, ::testing::ValuesIn(K_EXPECTED_ROWS),
                         [](const ::testing::TestParamInfo<ExpectedRow>& info)
                         {
                             std::string name(Ops::name_of(info.param.op));
                             name.replace(0, name.find('.') + 1, "");
                             return ::SushiAI::Test::parameter_name(name);
                         });
