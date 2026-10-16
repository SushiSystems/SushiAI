/**************************************************************************/
/* test_expert_trace.cpp                                                  */
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
 * @file test_expert_trace.cpp
 * @brief Tests the port names and the empty state of ExpertTrace.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/expert_trace.hpp>
#include <SushiAI/nn/node.hpp>

using namespace SushiAI;
using Language::Blocks::ExpertTrace;
using Language::Blocks::K_BALANCE_PORT;
using Language::Blocks::K_USAGE_PORT;

/** @brief Tests that a default trace names no value in any of its three fields. */
TEST(Small_BlockExpertTrace, ADefaultTraceHoldsNoValue)
{
    const ExpertTrace trace;

    EXPECT_EQ(trace.output, Graph::K_NO_VALUE);
    EXPECT_EQ(trace.balance, Graph::K_NO_VALUE);
    EXPECT_EQ(trace.usage, Graph::K_NO_VALUE);
}

/** @brief Tests that the two extra ports differ from each other and from the single port. */
TEST(Small_BlockExpertTrace, ThePortNamesAreDistinct)
{
    EXPECT_EQ(K_BALANCE_PORT, "balance");
    EXPECT_EQ(K_USAGE_PORT, "usage");
    EXPECT_NE(K_BALANCE_PORT, K_USAGE_PORT);
    EXPECT_NE(K_BALANCE_PORT, NN::K_SINGLE_PORT_NAME);
    EXPECT_NE(K_USAGE_PORT, NN::K_SINGLE_PORT_NAME);
}
