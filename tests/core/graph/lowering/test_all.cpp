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
 * @file test_all.cpp
 * @brief Tests that each identity of the operation table has one lowering row or is unlowered.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/unlowered.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Tests that every identity has one row, unless it is listed as unlowered. */
        TEST(Small_LoweringTable, EveryIdentityIsLoweredOrListedAsUnlowered)
        {
            for (const Ops::OpName& entry : Ops::K_ALL)
            {
                const bool unlowered = Graph::is_unlowered(entry.id);
                const std::size_t expected_rows = unlowered ? 0u : 1u;

                EXPECT_EQ(Graph::lowering_row_count(entry.id), expected_rows) << entry.name;
                EXPECT_EQ(Graph::find_lowering(entry.id) != nullptr, !unlowered) << entry.name;
            }
        }

        /** @brief Tests that the four identities of K_UNLOWERED have no row. */
        TEST(Small_LoweringTable, TheUnloweredIdentitiesHaveNoRow)
        {
            for (const OpID id : Graph::K_UNLOWERED)
            {
                EXPECT_TRUE(Graph::find_lowering(id) == nullptr) << Ops::name_of(id);
                EXPECT_EQ(Graph::lowering_row_count(id), 0u) << Ops::name_of(id);
            }

            EXPECT_EQ(Graph::K_UNLOWERED.size(), 4u);
        }

        /** @brief Tests that no row of the combined table names an identity with a second row. */
        TEST(Small_LoweringTable, NoOperationHasTwoRows)
        {
            for (const Graph::LoweringEntry& row : Graph::K_LOWERINGS)
            {
                EXPECT_EQ(Graph::lowering_row_count(row.op), 1u) << Ops::name_of(row.op);
            }
        }

        /** @brief Tests that every row names a declared identity and that the counts add up. */
        TEST(Small_LoweringTable, TheCombinedTableCoversTheOperationTable)
        {
            EXPECT_TRUE(Graph::every_identity_is_accounted_for(
                SushiRuntime::span<const Ops::OpName>(Ops::K_ALL.data(), Ops::K_ALL.size())));
            EXPECT_EQ(Graph::K_LOWERINGS.size() + Graph::K_UNLOWERED.size(), Ops::K_ALL.size());
        }

        /** @brief Tests the row count of each domain table and of the joined table. */
        TEST(Small_LoweringTable, EachDomainTableHasItsRowCount)
        {
            EXPECT_EQ(Graph::K_ELEMENTWISE_LOWERINGS.size(), 17u);
            EXPECT_EQ(Graph::K_ACTIVATION_LOWERINGS.size(), 14u);
            EXPECT_EQ(Graph::K_LINALG_LOWERINGS.size(), 6u);
            EXPECT_EQ(Graph::K_LOSS_LOWERINGS.size(), 3u);
            EXPECT_EQ(Graph::K_SHAPE_LOWERINGS.size(), 4u);
            EXPECT_EQ(Graph::K_SPATIAL_LOWERINGS.size(), 15u);
            EXPECT_EQ(Graph::K_NORM_LOWERINGS.size(), 6u);
            EXPECT_EQ(Graph::K_FUSED_LOWERINGS.size(), 6u);
            EXPECT_EQ(Graph::K_RANDOM_LOWERINGS.size(), 1u);
            EXPECT_EQ(Graph::K_LOWERINGS.size(), 72u);
        }

    } // namespace Test
} // namespace SushiAI
