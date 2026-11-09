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
 * @file test_buffer.cpp
 * @brief Tests Buffer: its description, its one leaf per graph and its identity.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/nn/buffer.hpp>

using namespace SushiAI;
using NN::Buffer;

namespace
{
    /** @brief Reads @p buffer before define() was called, for a death test. */
    void read_undefined(Buffer& buffer)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        (void)buffer.read(builder, "running_mean");
    }
}

/** @brief Tests that define() records the shape, the element type and the starting value. */
TEST(Small_Buffer, DefineRecordsTheDescription)
{
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 1.0, DType::Float64));

    EXPECT_TRUE(buffer.declared());
    EXPECT_EQ(buffer.shape(), (Shape{8}));
    EXPECT_EQ(buffer.dtype(), DType::Float64);
    EXPECT_EQ(buffer.initial_value(), 1.0);
}

/** @brief Tests that define() defaults the element type to FLOAT32. */
TEST(Small_Buffer, DefineDefaultsToFloat32)
{
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0));

    EXPECT_EQ(buffer.dtype(), DType::Float32);
}

/** @brief Tests that a buffer that is defined and never read adds no value to a graph. */
TEST(Small_Buffer, ADefinedBufferAddsNoLeafUntilItIsRead)
{
    Graph::Graph graph;
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0));

    EXPECT_EQ(graph.value_count(), 0u);
    EXPECT_EQ(buffer.id_in(graph), Graph::K_NO_VALUE);
}

/** @brief Tests that define() draws an identity distinct from another buffer's. */
TEST(Small_Buffer, DefineDrawsADistinctIdentity)
{
    Buffer first;
    Buffer second;
    SA_EXPECT_OK(first.define(Shape{8}, 0.0));
    SA_EXPECT_OK(second.define(Shape{8}, 0.0));

    EXPECT_NE(first.state(), Graph::K_NO_STATE);
    EXPECT_NE(second.state(), Graph::K_NO_STATE);
    EXPECT_NE(second.state(), first.state());
}

/** @brief Tests that a second define() and a read keep the identity the first define() drew. */
TEST(Small_Buffer, DefineAndReadKeepTheIdentity)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0));
    const Graph::StateID drawn = buffer.state();

    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0));
    (void)buffer.read(builder, "running_mean");

    EXPECT_EQ(buffer.state(), drawn);
}

/** @brief Tests that read() declares a persistent leaf of the defined shape with no gradient. */
TEST(Small_Buffer, ReadDeclaresABufferLeaf)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 1.0, DType::Float64));
    const Graph::ValueID leaf = buffer.read(builder, "running_var");

    EXPECT_EQ(graph.value_count(), 1u);
    EXPECT_EQ(graph.value(leaf).lifetime, Graph::Lifetime::Buffer);
    EXPECT_FALSE(graph.value(leaf).requires_grad);
    EXPECT_EQ(graph.value(leaf).shape, (Shape{8}));
    EXPECT_EQ(graph.value(leaf).dtype, DType::Float64);
    EXPECT_EQ(buffer.id_in(graph), leaf);
    EXPECT_EQ(graph.state_leaf(buffer.state()), leaf);
}

/** @brief Tests that a second read in one graph returns the first leaf and adds no value. */
TEST(Small_Buffer, ASecondReadInOneGraphReturnsTheFirstLeaf)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0));
    const Graph::ValueID first_leaf = buffer.read(builder, "running_mean");
    const std::size_t values_after_first = graph.value_count();

    EXPECT_EQ(buffer.read(builder, "running_mean"), first_leaf);
    EXPECT_EQ(graph.value_count(), values_after_first);
    EXPECT_EQ(graph.state_leaves().size(), 1u);
}

/** @brief Tests that one buffer read in two graphs has its own leaf in each, under one StateID. */
TEST(Small_Buffer, ABufferReadInTwoGraphsHasALeafInEach)
{
    Graph::Graph first;
    Graph::Graph second;
    const Graph::GraphBuilder first_builder(first);
    const Graph::GraphBuilder second_builder(second);

    // The padding input gives the first graph's leaf an id the second graph does not repeat.
    (void)first_builder.input(Shape{2, 8}, "padding");

    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 0.0));
    const Graph::ValueID first_leaf = buffer.read(first_builder, "running_mean");
    const Graph::StateID state_after_first = buffer.state();
    const Graph::ValueID second_leaf = buffer.read(second_builder, "running_mean");

    EXPECT_NE(first_leaf, second_leaf);
    EXPECT_EQ(buffer.id_in(first), first_leaf);
    EXPECT_EQ(buffer.id_in(second), second_leaf);
    EXPECT_EQ(first.state_leaf(state_after_first), first_leaf);
    EXPECT_EQ(second.state_leaf(state_after_first), second_leaf);
    EXPECT_NE(state_after_first, Graph::K_NO_STATE);
    EXPECT_EQ(buffer.state(), state_after_first);
}

/** @brief Tests that a copied buffer keeps the description and is new state with its own leaf. */
TEST(Small_Buffer, ACopiedBufferHasANewStateID)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Buffer original;
    SA_EXPECT_OK(original.define(Shape{8}, 1.0));
    const Graph::ValueID original_leaf = original.read(builder, "running_var");

    Buffer copy = original;
    EXPECT_TRUE(copy.declared());
    EXPECT_EQ(copy.shape(), (Shape{8}));
    EXPECT_EQ(copy.initial_value(), 1.0);
    EXPECT_NE(copy.state(), Graph::K_NO_STATE);
    EXPECT_NE(copy.state(), original.state());
    EXPECT_EQ(copy.id_in(graph), Graph::K_NO_VALUE);

    const Graph::ValueID copy_leaf = copy.read(builder, "running_var");

    EXPECT_NE(copy_leaf, original_leaf);
    EXPECT_NE(copy.state(), original.state());
    EXPECT_EQ(original.id_in(graph), original_leaf);
}

/** @brief Tests that a moved buffer keeps its StateID and the leaf it has in a graph. */
TEST(Small_Buffer, AMovedBufferKeepsItsStateID)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Buffer original;
    SA_EXPECT_OK(original.define(Shape{8}, 1.0));
    const Graph::ValueID leaf = original.read(builder, "running_var");
    const Graph::StateID state = original.state();
    const std::size_t values_after_read = graph.value_count();

    Buffer moved = std::move(original);

    EXPECT_EQ(moved.state(), state);
    EXPECT_EQ(moved.id_in(graph), leaf);
    EXPECT_EQ(moved.read(builder, "running_var"), leaf);
    EXPECT_EQ(graph.value_count(), values_after_read);
}

/** @brief Tests that a buffer moved into a new one is left undefined with no identity. */
TEST(Small_Buffer, AMoveConstructedFromBufferIsUndefined)
{
    Buffer original;
    SA_EXPECT_OK(original.define(Shape{8}, 1.0));

    const Buffer constructed = std::move(original);

    EXPECT_FALSE(original.declared());
    EXPECT_EQ(original.state(), Graph::K_NO_STATE);
    EXPECT_TRUE(constructed.declared());
}

/** @brief Tests that a buffer moved into an assigned one is left undefined with no identity. */
TEST(Small_Buffer, AMoveAssignedFromBufferIsUndefined)
{
    Buffer original;
    SA_EXPECT_OK(original.define(Shape{8}, 1.0));
    Buffer assigned;

    assigned = std::move(original);

    EXPECT_FALSE(original.declared());
    EXPECT_EQ(original.state(), Graph::K_NO_STATE);
    EXPECT_TRUE(assigned.declared());
    EXPECT_NE(assigned.state(), Graph::K_NO_STATE);
}

/** @brief Tests that a copy assignment gives the target a new identity and the description. */
TEST(Small_Buffer, ACopyAssignedBufferHasANewStateID)
{
    Buffer original;
    Buffer target;
    SA_EXPECT_OK(original.define(Shape{8}, 1.0, DType::Float64));
    SA_EXPECT_OK(target.define(Shape{2}, 0.0));
    const Graph::StateID replaced = target.state();

    target = original;

    EXPECT_EQ(target.shape(), (Shape{8}));
    EXPECT_EQ(target.dtype(), DType::Float64);
    EXPECT_NE(target.state(), Graph::K_NO_STATE);
    EXPECT_NE(target.state(), replaced);
    EXPECT_NE(target.state(), original.state());
}

/** @brief Tests that a copy of an undefined buffer is undefined and holds no identity. */
TEST(Small_Buffer, ACopyOfAnUndefinedBufferHasNoIdentity)
{
    const Buffer original;
    const Buffer copy = original;

    EXPECT_FALSE(copy.declared());
    EXPECT_EQ(copy.state(), Graph::K_NO_STATE);
}

/** @brief Tests that define() called again with the same description changes nothing. */
TEST(Small_Buffer, DefiningTwiceWithOneDescriptionIsAccepted)
{
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(Shape{8}, 1.0, DType::Float64));
    SA_EXPECT_OK(buffer.define(Shape{8}, 1.0, DType::Float64));

    EXPECT_EQ(buffer.shape(), (Shape{8}));
    EXPECT_EQ(buffer.dtype(), DType::Float64);
    EXPECT_EQ(buffer.initial_value(), 1.0);
}

/** @brief Holds the first description, the changed one and what the refusal names. */
struct RedefineRow
{
    std::string name;    /**< The row's name. */
    Shape shape;         /**< The shape of the first define(). */
    double value;        /**< The starting value of the first define(). */
    DType dtype;         /**< The element type of the first define(). */
    Shape other_shape;   /**< The shape of the second define(). */
    double other_value;  /**< The starting value of the second define(). */
    DType other_dtype;   /**< The element type of the second define(). */
    std::string first;   /**< A substring the message holds. */
    std::string second;  /**< A second substring the message holds. */
};

/** @brief Runs the redefinition refusal over each field that can change. */
class Small_BufferRedefine : public ::testing::TestWithParam<RedefineRow>
{
};

/** @brief Tests that define() with a changed description fails, names it and keeps the first. */
TEST_P(Small_BufferRedefine, DefiningWithAChangedDescriptionFails)
{
    const RedefineRow& row = GetParam();
    Buffer buffer;
    SA_EXPECT_OK(buffer.define(row.shape, row.value, row.dtype));

    SA_EXPECT_FAILURE(buffer.define(row.other_shape, row.other_value, row.other_dtype),
                      StatusCode::InvalidDescription, "Buffer was defined as", row.first,
                      row.second);
    EXPECT_EQ(buffer.shape(), row.shape);
    EXPECT_EQ(buffer.dtype(), row.dtype);
    EXPECT_EQ(buffer.initial_value(), row.value);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BufferRedefine,
    ::testing::Values(
        RedefineRow{"Shape", Shape{8}, 0.0, DType::Float32, Shape{4}, 0.0, DType::Float32, "[8]",
                    "[4]"},
        RedefineRow{"ElementType", Shape{8}, 0.0, DType::Float32, Shape{8}, 0.0, DType::Float64,
                    dtype_name(DType::Float32), dtype_name(DType::Float64)},
        RedefineRow{"StartingValue", Shape{8}, 0.0, DType::Float32, Shape{8}, 1.0, DType::Float32,
                    "is now defined as", ""}),
    [](const ::testing::TestParamInfo<RedefineRow>& info) { return info.param.name; });

/** @brief Tests that read() before define() stops the process and names the buffer. */
TEST(Medium_BufferDeathTest, ReadingBeforeDefineIsFatal)
{
    Buffer buffer;

    SA_EXPECT_FATAL(read_undefined(buffer), "'running_mean'", "before define()");
}
