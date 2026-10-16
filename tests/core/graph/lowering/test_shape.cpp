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
 * @file test_shape.cpp
 * @brief Tests the shape lowering table.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/shape.hpp>

using namespace SushiAI;
using Graph::LoweringRule;

namespace
{
    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        OpID op;           /**< The operation. */
        LoweringRule rule; /**< The row that lowers it. */
    };
} // namespace

/** @brief Tests that each row pairs its operation with the function the header names. */
TEST(Small_ShapeLowerings, EachRowPairsItsOperationWithItsLowering)
{
    const std::array<ExpectedRow, 4> expected = {{
        {Ops::K_LEAF.id, &Graph::lower_leaf},
        {Ops::K_RESHAPE.id, &Graph::lower_reshape},
        {Ops::K_LAYOUT_CONVERT.id, &Graph::lower_layout_convert},
        {Ops::K_STOP_GRADIENT.id, &Graph::lower_stop_gradient},
    }};

    ASSERT_EQ(Graph::K_SHAPE_LOWERINGS.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const Graph::LoweringEntry& row = Graph::K_SHAPE_LOWERINGS[i];
        EXPECT_EQ(row.op, expected[i].op) << Ops::name_of(expected[i].op);
        EXPECT_EQ(row.rule, expected[i].rule) << Ops::name_of(expected[i].op);
        EXPECT_EQ(row.scratch, nullptr) << Ops::name_of(expected[i].op);
        EXPECT_EQ(Graph::find_lowering(expected[i].op), expected[i].rule)
            << Ops::name_of(expected[i].op);
    }
}
