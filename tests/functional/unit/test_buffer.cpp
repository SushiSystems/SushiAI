/**************************************************************************/
/* test_buffer.cpp                                                        */
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
 * @file test_buffer.cpp
 * @brief Tests Buffer: its description, its one leaf per graph and its identity.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/nn/buffer.hpp>
#include "../../common/test_status.hpp"

using namespace SushiAI;
using NN::Buffer;

namespace
{
    /** @brief Holds the absolute tolerance of a double stored and read back unchanged. */
    constexpr double K_EXACT_TOLERANCE = 1e-15;

    /** @brief Reads @p buffer before define() was called, for a death test. */
    void read_undefined(Buffer& buffer)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        (void)buffer.read(builder, "running_mean");
    }
}

/** @brief Tests that a default-constructed buffer is undefined and has no identity. */
TEST(Unit_Buffer, ANewBufferIsUndefined)
{
    const Buffer buffer;

    EXPECT_FALSE(buffer.declared());
    EXPECT_EQ(buffer.state(), Graph::K_NO_STATE);
}

/** @brief Tests that define() records the shape, the element type and the starting value. */
TEST(Unit_Buffer, DefineRecordsTheDescription)
{
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 1.0, DType::Float64));

    EXPECT_TRUE(buffer.declared());
    EXPECT_EQ(buffer.shape(), (Shape{8}));
    EXPECT_EQ(buffer.dtype(), DType::Float64);
    EXPECT_NEAR(buffer.initial_value(), 1.0, K_EXACT_TOLERANCE);
}

/** @brief Tests that define() defaults the element type to FLOAT32. */
TEST(Unit_Buffer, DefineDefaultsToFloat32)
{
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0));

    EXPECT_EQ(buffer.dtype(), DType::Float32);
}

/** @brief Tests that a buffer that is defined and never read adds no value to a graph. */
TEST(Unit_Buffer, ADefinedBufferAddsNoLeafUntilItIsRead)
{
    Graph::Graph graph;
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0));

    EXPECT_EQ(graph.value_count(), 0u);
    EXPECT_EQ(buffer.id_in(graph), Graph::K_NO_VALUE);
}

/** @brief Tests that define() draws the identity, which a later read and define() keep. */
TEST(Unit_Buffer, DefineDrawsTheIdentity)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Buffer first;
    Buffer second;
    SA_EXPECT_OK(first.define(Shape{8}, 0.0));
    SA_EXPECT_OK(second.define(Shape{8}, 0.0));

    const Graph::StateId drawn = first.state();
    EXPECT_NE(drawn, Graph::K_NO_STATE);
    EXPECT_NE(second.state(), Graph::K_NO_STATE);
    EXPECT_NE(second.state(), drawn);

    SA_EXPECT_OK(first.define(Shape{8}, 0.0));
    (void)first.read(builder, "running_mean");
    EXPECT_EQ(first.state(), drawn);
}

/** @brief Tests that read() declares a persistent leaf of the defined shape with no gradient. */
TEST(Unit_Buffer, ReadDeclaresABufferLeaf)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 1.0, DType::Float64));
    const Graph::ValueId leaf = buffer.read(builder, "running_var");

    EXPECT_EQ(graph.value_count(), 1u);
    EXPECT_EQ(graph.value(leaf).lifetime, Graph::Lifetime::Buffer);
    EXPECT_FALSE(graph.value(leaf).requires_grad);
    EXPECT_EQ(graph.value(leaf).shape, (Shape{8}));
    EXPECT_EQ(graph.value(leaf).dtype, DType::Float64);
    EXPECT_EQ(buffer.id_in(graph), leaf);
    EXPECT_EQ(graph.state_leaf(buffer.state()), leaf);
}

/** @brief Tests that a second read in one graph returns the first leaf and adds no value. */
TEST(Unit_Buffer, ASecondReadInOneGraphReturnsTheFirstLeaf)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0));
    const Graph::ValueId first_leaf = buffer.read(builder, "running_mean");
    const std::size_t values_after_first = graph.value_count();

    EXPECT_EQ(buffer.read(builder, "running_mean"), first_leaf);
    EXPECT_EQ(graph.value_count(), values_after_first);
    EXPECT_EQ(graph.state_leaves().size(), 1u);
}

/** @brief Tests that one buffer read in two graphs has its own leaf in each, under one StateId. */
TEST(Unit_Buffer, ABufferReadInTwoGraphsHasALeafInEach)
{
    Graph::Graph first;
    Graph::Graph second;
    const Graph::GraphBuilder first_builder(first);
    const Graph::GraphBuilder second_builder(second);

    // The padding input gives the first graph's leaf an id the second graph does not repeat.
    (void)first_builder.input(Shape{2, 8}, "padding");

    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0));
    const Graph::ValueId first_leaf = buffer.read(first_builder, "running_mean");
    const Graph::StateId state_after_first = buffer.state();
    const Graph::ValueId second_leaf = buffer.read(second_builder, "running_mean");

    EXPECT_EQ(first_leaf, 1u);
    EXPECT_EQ(second_leaf, 0u);
    EXPECT_EQ(buffer.id_in(first), first_leaf);
    EXPECT_EQ(buffer.id_in(second), second_leaf);
    EXPECT_NE(state_after_first, Graph::K_NO_STATE);
    EXPECT_EQ(buffer.state(), state_after_first);
}

/** @brief Tests that a copied buffer keeps the description and is new state with its own leaf. */
TEST(Unit_Buffer, ACopiedBufferHasANewStateId)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Buffer original;
    SA_EXPECT_OK(original.define(Shape{8}, 1.0));
    const Graph::ValueId original_leaf = original.read(builder, "running_var");

    Buffer copy = original;
    EXPECT_TRUE(copy.declared());
    EXPECT_EQ(copy.shape(), (Shape{8}));
    EXPECT_NEAR(copy.initial_value(), 1.0, K_EXACT_TOLERANCE);
    EXPECT_NE(copy.state(), Graph::K_NO_STATE);
    EXPECT_NE(copy.state(), original.state());
    EXPECT_EQ(copy.id_in(graph), Graph::K_NO_VALUE);

    const Graph::ValueId copy_leaf = copy.read(builder, "running_var");

    EXPECT_NE(copy_leaf, original_leaf);
    EXPECT_NE(copy.state(), original.state());
    EXPECT_EQ(original.id_in(graph), original_leaf);
}

/** @brief Tests that a moved buffer keeps its StateId and the leaf it has in a graph. */
TEST(Unit_Buffer, AMovedBufferKeepsItsStateId)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Buffer original;
    SA_EXPECT_OK(original.define(Shape{8}, 1.0));
    const Graph::ValueId leaf = original.read(builder, "running_var");
    const Graph::StateId state = original.state();
    const std::size_t values_after_read = graph.value_count();

    Buffer moved = std::move(original);

    EXPECT_EQ(moved.state(), state);
    EXPECT_EQ(moved.id_in(graph), leaf);
    EXPECT_EQ(moved.read(builder, "running_var"), leaf);
    EXPECT_EQ(graph.value_count(), values_after_read);
}

/** @brief Tests that a moved-from buffer is undefined and holds no identity. */
TEST(Unit_Buffer, AMovedFromBufferIsUndefined)
{
    Buffer original;
    SA_EXPECT_OK(original.define(Shape{8}, 1.0));

    Buffer constructed = std::move(original);
    EXPECT_FALSE(original.declared());
    EXPECT_EQ(original.state(), Graph::K_NO_STATE);

    Buffer assigned;
    assigned = std::move(constructed);
    EXPECT_FALSE(constructed.declared());
    EXPECT_EQ(constructed.state(), Graph::K_NO_STATE);
    EXPECT_TRUE(assigned.declared());
    EXPECT_NE(assigned.state(), Graph::K_NO_STATE);
}

/** @brief Tests that a copy assignment gives the target a new identity and the description. */
TEST(Unit_Buffer, ACopyAssignedBufferHasANewStateId)
{
    Buffer original;
    Buffer target;
    SA_EXPECT_OK(original.define(Shape{8}, 1.0, DType::Float64));
    SA_EXPECT_OK(target.define(Shape{2}, 0.0));
    const Graph::StateId replaced = target.state();

    target = original;

    EXPECT_EQ(target.shape(), (Shape{8}));
    EXPECT_EQ(target.dtype(), DType::Float64);
    EXPECT_NE(target.state(), Graph::K_NO_STATE);
    EXPECT_NE(target.state(), replaced);
    EXPECT_NE(target.state(), original.state());
}

/** @brief Tests that a copy of an undefined buffer is undefined and holds no identity. */
TEST(Unit_Buffer, ACopyOfAnUndefinedBufferHasNoIdentity)
{
    const Buffer original;
    const Buffer copy = original;

    EXPECT_FALSE(copy.declared());
    EXPECT_EQ(copy.state(), Graph::K_NO_STATE);
}

/** @brief Tests that define() called again with the same description changes nothing. */
TEST(Unit_Buffer, DefiningTwiceWithOneDescriptionIsAccepted)
{
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 1.0, DType::Float64));
    SA_EXPECT_OK(buffer.define(Shape{8}, 1.0, DType::Float64));

    EXPECT_EQ(buffer.shape(), (Shape{8}));
    EXPECT_EQ(buffer.dtype(), DType::Float64);
    EXPECT_NEAR(buffer.initial_value(), 1.0, K_EXACT_TOLERANCE);
}

/** @brief Tests that define() with another shape fails, names both and keeps the first. */
TEST(Unit_Buffer, DefiningWithAnotherShapeFails)
{
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0));

    SA_EXPECT_FAILURE(buffer.define(Shape{4}, 0.0), StatusCode::InvalidDescription,
                      "Buffer was defined as", "[8]", "[4]");
    EXPECT_EQ(buffer.shape(), (Shape{8}));
}

/** @brief Tests that define() with another element type fails and names both types. */
TEST(Unit_Buffer, DefiningWithAnotherElementTypeFails)
{
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0, DType::Float32));

    SA_EXPECT_FAILURE(buffer.define(Shape{8}, 0.0, DType::Float64),
                      StatusCode::InvalidDescription, "Buffer was defined as",
                      dtype_name(DType::Float32), dtype_name(DType::Float64));
    EXPECT_EQ(buffer.dtype(), DType::Float32);
}

/** @brief Tests that define() with another starting value fails and keeps the first value. */
TEST(Unit_Buffer, DefiningWithAnotherStartingValueFails)
{
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0));

    SA_EXPECT_FAILURE(buffer.define(Shape{8}, 1.0), StatusCode::InvalidDescription,
                      "Buffer was defined as", "is now defined as");
    EXPECT_NEAR(buffer.initial_value(), 0.0, K_EXACT_TOLERANCE);
}

/** @brief Tests that read() before define() stops the process and names the buffer. */
TEST(Unit_BufferDeathTest, ReadingBeforeDefineIsFatal)
{
    Buffer buffer;

    SA_EXPECT_FATAL(read_undefined(buffer), "'running_mean'", "before define()");
}
