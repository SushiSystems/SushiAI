/**************************************************************************/
/* test_global_avg_pool.cpp                                               */
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
 * @file test_global_avg_pool.cpp
 * @brief Tests the shapes GlobalAvgPool traces and reports, and the inputs it refuses.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/global_avg_pool.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using NN::GlobalAvgPool;

namespace
{
    /** @brief Traces the layer over an [N, H, W, C] input and returns the result's shape. */
    Shape traced_shape(int64_t batch, int64_t height, int64_t width, int64_t channels)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{batch, height, width, channels}, "x");

        const GlobalAvgPool layer{};
        return graph.value(layer.forward(builder, x)).shape;
    }
} // namespace

/** @brief Tests that GlobalAvgPool satisfies the Module trait as a leaf with a const forward(). */
TEST(Unit_GlobalAvgPool, SatisfiesTheModuleTrait)
{
    static_assert(NN::Module<GlobalAvgPool>, "GlobalAvgPool must satisfy the Module contract");
    static_assert(!NN::ModuleContainer<GlobalAvgPool>,
                  "GlobalAvgPool is a leaf and must not look like a container");
    static_assert(std::is_same<decltype(std::declval<const GlobalAvgPool&>().forward(
                                   std::declval<const Graph::GraphBuilder&>(),
                                   std::declval<Graph::ValueID>())),
                               Graph::ValueID>::value,
                  "GlobalAvgPool declares no parameter, so forward() must be const");
    SUCCEED();
}

/** @brief Tests that forward() reduces an [N, H, W, C] input to [N, C]. */
TEST(Unit_GlobalAvgPool, ForwardReducesTheSpatialAxes)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 7, 5, 16}, "x");

    const GlobalAvgPool layer{};
    const Graph::ValueID y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{4, 16}));
}

/** @brief Tests that forward() emits one SPATIAL_MEAN node reading its input. */
TEST(Unit_GlobalAvgPool, ForwardEmitsOneSpatialMeanNode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3, 3, 8}, "x");

    const GlobalAvgPool layer{};
    const std::size_t nodes_before = graph.nodes().size();
    const Graph::ValueID y = layer.forward(builder, x);

    ASSERT_EQ(graph.nodes().size(), nodes_before + 1);
    const Graph::Node& node = graph.nodes().back();
    EXPECT_EQ(node.op, Ops::K_SPATIAL_MEAN.id);
    ASSERT_EQ(node.inputs.size(), 1u);
    EXPECT_EQ(node.inputs.front(), x);
    ASSERT_EQ(node.outputs.size(), 1u);
    EXPECT_EQ(node.outputs.front(), y);
}

/** @brief Tests that the traced result keeps the NHWC layout tag of the rank-4 input it reads. */
TEST(Unit_GlobalAvgPool, ForwardKeepsTheNHWCLayoutTag)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 6, 6, 3}, "x");
    ASSERT_EQ(graph.value(x).layout, Graph::TensorLayout::NHWC);

    const GlobalAvgPool layer{};
    const Graph::ValueID y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
    EXPECT_NO_THROW(graph.finalise());
}

/** @brief Tests that output_shape() maps a per-sample [H, W, C] shape to [C]. */
TEST(Unit_GlobalAvgPool, OutputShapeKeepsOnlyTheChannels)
{
    const GlobalAvgPool layer{};

    EXPECT_EQ(layer.output_shape(Shape{7, 5, 16}), (Shape{16}));
}

/** @brief Tests that output_shape() agrees with the traced shape for two different inputs. */
TEST(Unit_GlobalAvgPool, OutputShapeAgreesWithForward)
{
    const GlobalAvgPool layer{};

    const Shape square = layer.output_shape(Shape{7, 7, 512});
    ASSERT_EQ(square.rank(), 1);
    EXPECT_EQ(traced_shape(4, 7, 7, 512), (Shape{4, square[0]}));

    const Shape oblong = layer.output_shape(Shape{3, 9, 2});
    ASSERT_EQ(oblong.rank(), 1);
    EXPECT_EQ(traced_shape(1, 3, 9, 2), (Shape{1, oblong[0]}));
}

/** @brief Tests that output_shape() rejects a per-sample shape of rank 1. */
TEST(Unit_GlobalAvgPool, OutputShapeRejectsAFeatureVector)
{
    const GlobalAvgPool layer{};

    EXPECT_THROW((void)layer.output_shape(Shape{16}), Error);
}

/** @brief Tests that output_shape() rejects a per-sample shape of rank 2. */
TEST(Unit_GlobalAvgPool, OutputShapeRejectsARankTwoShape)
{
    const GlobalAvgPool layer{};

    EXPECT_THROW((void)layer.output_shape(Shape{7, 16}), Error);
}

/** @brief Tests that output_shape() rejects a rank-4 shape that still carries the batch axis. */
TEST(Unit_GlobalAvgPool, OutputShapeRejectsAShapeWithTheBatchAxis)
{
    const GlobalAvgPool layer{};

    EXPECT_THROW((void)layer.output_shape(Shape{4, 7, 5, 16}), Error);
}

/** @brief Tests that forward() rejects a rank-2 [N, F] input. */
TEST(Unit_GlobalAvgPool, ForwardRejectsAFeatureMatrix)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{4, 16}, "x");

    const GlobalAvgPool layer{};

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}

/** @brief Tests that forward() rejects a rank-3 input that lacks the batch axis. */
TEST(Unit_GlobalAvgPool, ForwardRejectsAnInputWithoutTheBatchAxis)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{7, 5, 16}, "x");

    const GlobalAvgPool layer{};

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}

/** @brief Tests that forward() rejects a rank-4 input tagged NCHW. */
TEST(Unit_GlobalAvgPool, ForwardRejectsAnNCHWInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 4, 4, 3}, "x");
    const Graph::ValueID x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);

    const GlobalAvgPool layer{};

    EXPECT_THROW((void)layer.forward(builder, x_nchw), Error);
}

/** @brief Tests that the parameter walk visits nothing, before and after a trace. */
TEST(Unit_GlobalAvgPool, TheParameterWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3, 3, 8}, "x");

    GlobalAvgPool layer{};
    EXPECT_EQ(NN::parameter_count(layer), 0u);

    (void)layer.forward(builder, x);

    EXPECT_EQ(NN::parameter_count(layer), 0u);
    EXPECT_EQ(NN::parameter_elements(layer), 0);
}
