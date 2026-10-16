/**************************************************************************/
/* test_minimal_case.cpp                                                  */
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
 * @file test_minimal_case.cpp
 * @brief Tests that make_minimal() builds the node a MinimalCase describes, or names its row.
 * @author Mustafa Garip
 */

#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/layer_rows/layer_registry.hpp>
#include <SushiAI/layer_rows/minimal_case.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_registry.hpp>

using namespace SushiAI;

/** @brief Shortens the namespace of the test helpers, which Test alone hides in a case. */
namespace SaTest = SushiAI::Test;
using SaTest::MinimalCase;

/** @brief Tests that span_of() views every shape of the vector, in order. */
TEST(Small_LayerRowsMinimalCase, SpanOfViewsEveryShape)
{
    const std::vector<Shape> shapes{Shape{6}, Shape{2, 3}};

    const NN::ShapeSpan span = SaTest::span_of(shapes);

    ASSERT_EQ(span.size(), 2u);
    EXPECT_EQ(span.data(), shapes.data());
    EXPECT_EQ(span[1], (Shape{2, 3}));
}

/** @brief Tests that a layer row and a node row of two inputs build their stated output. */
TEST(Small_LayerRowsMinimalCase, AMinimalCaseBuildsItsStatedOutput)
{
    const NN::NodeRegistry registry = SaTest::make_layer_registry();
    const std::vector<MinimalCase> cases{
        {"linear", R"({"units": 5})", {Shape{6}}, Shape{5}},
        {"add", "{}", {Shape{2, 2, 3}, Shape{2, 2, 3}}, Shape{2, 2, 3}},
    };

    for (const MinimalCase& minimal : cases)
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                                SaTest::make_minimal(registry, minimal, minimal.attributes));

        EXPECT_EQ(node->input_count(), minimal.inputs.size()) << minimal.type;
        ASSERT_EQ(node->ports().size(), 1u) << minimal.type;
        EXPECT_EQ(node->ports()[0].shape, minimal.output) << minimal.type;
    }
}

/** @brief Tests that a row the factory refuses is named as layer 0 and its type. */
TEST(Small_LayerRowsMinimalCase, ARefusedRowIsNamedAsLayerZero)
{
    const NN::NodeRegistry registry = SaTest::make_layer_registry();
    const MinimalCase image{"conv2d", R"({"filters": 5, "kernel": 3})", {Shape{784}},
                            Shape{6, 6, 5}};

    SA_EXPECT_FAILURE(SaTest::make_minimal(registry, image, image.attributes),
                      StatusCode::InvalidDescription, "layer 0 'conv2d'", "[784]");
}
