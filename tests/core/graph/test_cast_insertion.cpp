/**************************************************************************/
/* test_cast_insertion.cpp                                                */
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
 * @file test_cast_insertion.cpp
 * @brief Tests the report of the cast pass, the mismatch finder and its one-line description.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>

using namespace SushiAI;
using Graph::CastReport;
using Graph::DTypeMismatch;
using Graph::GraphBuilder;
using Graph::ValueID;

/** @brief Tests that a pass with nothing to convert reports no cast and the short summary. */
TEST(Small_CastInsertion, APassWithNothingToConvertGivesTheShortReport)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3}, "x");
    builder.relu(x);
    const std::size_t nodes = graph.node_count();

    const CastReport report = Graph::insert_casts(graph);

    EXPECT_EQ(report.count(), 0u);
    EXPECT_EQ(report.count_into(DType::Half), 0u);
    EXPECT_TRUE(report.casts().empty());
    EXPECT_EQ(report.nodes_before(), nodes);
    EXPECT_EQ(report.nodes_after(), nodes);
    EXPECT_EQ(report.policy(), "fp32");

    const std::string text = report.to_string();
    EXPECT_NE(text.find("casts inserted : 0"), std::string::npos) << text;
    EXPECT_EQ(text.find("values retyped"), std::string::npos) << text;
}

/** @brief Tests that a graph already in the half type agrees with the mixed policy. */
TEST(Small_CastInsertion, AGraphAlreadyNarrowHasNoMismatchUnderTheMixedPolicy)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 3}, "x", DType::Half);
    const ValueID w = builder.parameter(Shape{5, 3}, "w", DType::Half);
    builder.matmul(x, w, false, true);

    const Graph::MixedPrecisionPolicy policy;

    EXPECT_TRUE(Graph::find_dtype_mismatches(graph, policy).empty());
}

/** @brief Tests that the description names the node, its operation, both types and the side. */
TEST(Small_CastInsertion, TheDescriptionNamesTheNodeTheTypesAndTheSide)
{
    DTypeMismatch mismatch;
    mismatch.node = 3;
    mismatch.op = Ops::K_MATMUL.id;
    mismatch.value = 7;
    mismatch.found = DType::Float32;
    mismatch.expected = DType::Half;
    mismatch.is_result = true;

    const std::string expected = std::string("node 3 (") + std::string(Ops::K_MATMUL.name)
                               + ") computes in " + dtype_name(DType::Half)
                               + " but its result value 7 is " + dtype_name(DType::Float32);
    EXPECT_EQ(Graph::describe(mismatch), expected);

    mismatch.is_result = false;
    EXPECT_NE(Graph::describe(mismatch).find("its operand value 7"), std::string::npos);
}
