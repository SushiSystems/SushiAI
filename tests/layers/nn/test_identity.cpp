/**************************************************************************/
/* test_identity.cpp                                                      */
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
 * @file test_identity.cpp
 * @brief Tests that the Identity layer returns its input, emits nothing and holds no state.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <tuple>
#include <gtest/gtest.h>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/identity.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using NN::Identity;

static_assert(NN::Module<Identity>, "Identity must satisfy the Module contract");
static_assert(!NN::ModuleContainer<Identity>, "Identity is a leaf, not a container");
static_assert(std::tuple_size<decltype(Identity::fields())>::value == 0,
              "Identity must declare no parameter field");

/** @brief Tests that forward returns the id it was given. */
TEST(Unit_Identity, ForwardReturnsItsInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");

    const Identity layer{};

    EXPECT_EQ(layer.forward(builder, x), x);
}

/** @brief Tests that forward adds no node and no value to the graph. */
TEST(Unit_Identity, ForwardEmitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    const std::size_t nodes_before = graph.node_count();
    const std::size_t values_before = graph.value_count();

    const Identity layer{};
    (void)layer.forward(builder, x);

    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_EQ(graph.value_count(), values_before);
}

/** @brief Tests that output_shape() returns its input for a feature vector and for an image. */
TEST(Unit_Identity, OutputShapeReturnsItsInput)
{
    const Identity layer{};

    EXPECT_EQ(layer.output_shape(Shape{7}), (Shape{7}));
    EXPECT_EQ(layer.output_shape(Shape{8, 6, 4}), (Shape{8, 6, 4}));
}

/** @brief Tests that the parameter walk visits nothing after a trace. */
TEST(Unit_Identity, TheParameterWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    Identity layer{};
    (void)layer.forward(builder, x);

    std::size_t visits = 0;
    NN::for_each_parameter(layer, [&visits](const std::string&, NN::Parameter&)
    {
        ++visits;
    });
    EXPECT_EQ(visits, 0u);
}
