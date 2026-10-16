/**************************************************************************/
/* test_node_range.cpp                                                    */
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
 * @file test_node_range.cpp
 * @brief Tests that a default node range names every node of a graph.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/node_range.hpp>

using namespace SushiAI;

/** @brief Tests that a default range starts at node 0 and ends at the graph's node count. */
TEST(Small_NodeRange, TheDefaultRangeIsTheWholeGraph)
{
    const Graph::NodeRange range;

    EXPECT_EQ(range.first, 0u);
    EXPECT_EQ(range.end, Graph::K_NO_NODE);
}

/** @brief Tests that a range written as a pair keeps its two ends in order. */
TEST(Small_NodeRange, ARangeKeepsTheEndsItIsGiven)
{
    const Graph::NodeRange range{2, 5};

    EXPECT_EQ(range.first, 2u);
    EXPECT_EQ(range.end, 5u);
}
