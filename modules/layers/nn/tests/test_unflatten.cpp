/**************************************************************************/
/* test_unflatten.cpp                                                     */
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
 * @file test_unflatten.cpp
 * @brief Tests the trace, the shape rule and the rejected inputs of the Unflatten layer.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/unflatten.hpp>
#include <common/test_throws.hpp>

using namespace SushiAI;
using NN::Unflatten;

namespace
{
    /** @brief Returns the shape of @p id with the batch axis removed. */
    Shape per_sample(const Graph::Graph& graph, Graph::ValueID id)
    {
        const Shape& full = graph.value(id).shape;
        const SushiRuntime::span<const int64_t> inner(full.data() + 1,
                                                      static_cast<std::size_t>(full.rank() - 1));
        return Shape(inner);
    }
} // namespace

/** @brief Tests that Unflatten satisfies the Module trait and is not a container. */
TEST(Unit_Unflatten, SatisfiesTheModuleContract)
{
    static_assert(NN::Module<Unflatten>, "Unflatten must be a Module");
    static_assert(!NN::ModuleContainer<Unflatten>, "Unflatten must be a leaf");
    static_assert(!NN::Detail::HasParameterWalk<Unflatten>::value,
                  "Unflatten must be enumerated by its fields()");
    SUCCEED();
}

/** @brief Tests that the traced result is [N, height, width, channels] with the input's dtype. */
TEST(Unit_Unflatten, TracesTheImageShape)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{4, 5, 3};
    const Graph::ValueID x = builder.input(Shape{2, 60}, "x", DType::Float64);
    const Graph::ValueID y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 5, 3}));
    EXPECT_EQ(graph.value(y).dtype, DType::Float64);
}

/** @brief Tests that forward() leaves the default NHWC tag on a result finalise() accepts. */
TEST(Unit_Unflatten, TagsTheResultNHWC)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{4, 5, 3};
    const Graph::ValueID x = builder.input(Shape{2, 60}, "x");
    const Graph::ValueID y = layer.forward(builder, x);

    ASSERT_EQ(graph.value(y).shape.rank(), 4);
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
    EXPECT_NO_THROW(graph.finalise());
}

/** @brief Tests that the trace adds exactly one node, a RESHAPE reading the input. */
TEST(Unit_Unflatten, EmitsOneReshapeNode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{4, 5, 3};
    const Graph::ValueID x = builder.input(Shape{2, 60}, "x");
    const std::size_t nodes_before = graph.node_count();
    const Graph::ValueID y = layer.forward(builder, x);

    ASSERT_EQ(graph.node_count(), nodes_before + 1);
    const Graph::Node& node = graph.node(graph.value(y).producer);
    EXPECT_EQ(node.op, Ops::K_RESHAPE.id);
    ASSERT_EQ(node.inputs.size(), 1u);
    EXPECT_EQ(node.inputs.front(), x);
}

/** @brief Tests that output_shape() equals the traced per-sample shape of a multi-channel image. */
TEST(Unit_Unflatten, OutputShapeAgreesWithTheTraceForAnRGBImage)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{4, 5, 3};
    const Graph::ValueID x = builder.input(Shape{2, 60}, "x");
    const Graph::ValueID y = layer.forward(builder, x);

    EXPECT_EQ(layer.output_shape(per_sample(graph, x)), per_sample(graph, y));
    EXPECT_EQ(layer.output_shape(Shape{60}), (Shape{4, 5, 3}));
}

/** @brief Tests that output_shape() equals the traced per-sample shape of a one-channel image. */
TEST(Unit_Unflatten, OutputShapeAgreesWithTheTraceForAGreyImage)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{28, 28, 1};
    const Graph::ValueID x = builder.input(Shape{7, 784}, "x");
    const Graph::ValueID y = layer.forward(builder, x);

    EXPECT_EQ(layer.output_shape(per_sample(graph, x)), per_sample(graph, y));
    EXPECT_EQ(layer.output_shape(Shape{784}), (Shape{28, 28, 1}));
}

/** @brief Tests that forward() throws the layer's positive-extent error for a zero extent. */
TEST(Unit_Unflatten, ForwardRejectsAZeroExtent)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{4, 0, 3};
    const Graph::ValueID x = builder.input(Shape{2, 60}, "x");

    // The count check would also throw here, with a message that does not name the extents.
    SA_EXPECT_ERROR((void)layer.forward(builder, x), "Unflatten needs positive extents");
}

/** @brief Tests that forward() throws the layer's positive-extent error for a negative extent. */
TEST(Unit_Unflatten, ForwardRejectsANegativeExtent)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{-4, -5, 3};
    const Graph::ValueID x = builder.input(Shape{2, 60}, "x");

    // Shape's constructor would also throw here, with a message that does not name the layer.
    SA_EXPECT_ERROR((void)layer.forward(builder, x), "Unflatten needs positive extents");
}

/** @brief Tests that forward() throws on a rank-4 input whose element count would fit. */
TEST(Unit_Unflatten, ForwardRejectsAnInputThatIsNotRankTwo)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    // [2, 60, 1, 1] has inner extent 60 and 120 elements, so only the rank check can throw.
    const Unflatten layer{4, 5, 3};
    const Graph::ValueID x = builder.input(Shape{2, 60, 1, 1}, "x");

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}

/** @brief Tests that forward() throws the layer's own error for a mismatched inner extent. */
TEST(Unit_Unflatten, ForwardRejectsAMismatchedElementCount)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{4, 5, 3};
    const Graph::ValueID x = builder.input(Shape{3, 40}, "x");

    // reshape() would also reject it, with a message that does not name the layer.
    SA_EXPECT_ERROR((void)layer.forward(builder, x), "Unflatten(4, 5, 3)", "inner extent is 60");
}

/** @brief Tests that forward() throws on an input tagged NCHW. */
TEST(Unit_Unflatten, ForwardRejectsAnNCHWInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{4, 5, 3};
    const Graph::ValueID x = builder.input(Shape{2, 60}, "x");
    graph.value(x).layout = Graph::TensorLayout::NCHW;

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}

/** @brief Tests that output_shape() throws the layer's positive-extent error for a zero extent. */
TEST(Unit_Unflatten, OutputShapeRejectsAZeroExtent)
{
    const Unflatten layer{4, 5, 0};

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{60}), "Unflatten needs positive extents");
}

/** @brief Tests that output_shape() throws the layer's positive-extent error for a negative one. */
TEST(Unit_Unflatten, OutputShapeRejectsANegativeExtent)
{
    const Unflatten layer{-4, -5, 3};

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{60}), "Unflatten needs positive extents");
}

/** @brief Tests that output_shape() throws on a per-sample shape that is not rank 1. */
TEST(Unit_Unflatten, OutputShapeRejectsAShapeThatIsNotRankOne)
{
    const Unflatten layer{4, 5, 3};

    EXPECT_THROW((void)layer.output_shape(Shape{60, 1}), Error);
}

/** @brief Tests that output_shape() throws when the feature count does not match. */
TEST(Unit_Unflatten, OutputShapeRejectsAMismatchedElementCount)
{
    const Unflatten layer{4, 5, 3};

    EXPECT_THROW((void)layer.output_shape(Shape{59}), Error);
}

/** @brief Tests that the parameter walk visits nothing before and after a trace. */
TEST(Unit_Unflatten, HasNoParameters)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Unflatten layer{4, 5, 3};
    EXPECT_EQ(NN::parameter_count(layer), 0u);

    const Graph::ValueID x = builder.input(Shape{2, 60}, "x");
    const std::size_t values_before = graph.value_count();
    (void)layer.forward(builder, x);

    EXPECT_EQ(NN::parameter_count(layer), 0u);
    EXPECT_EQ(NN::parameter_elements(layer), 0);
    // The trace declares the reshaped value and no parameter.
    EXPECT_EQ(graph.value_count(), values_before + 1);
}
