/**************************************************************************/
/* test_upsample2d.cpp                                                    */
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
 * @file test_upsample2d.cpp
 * @brief Tests the shape, layout and rejections of the Upsample2d layer.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/upsample2d.hpp>
#include "../../common/test_throws.hpp"

using namespace SushiAI;
using NN::Upsample2d;

static_assert(NN::Module<Upsample2d>, "Upsample2d must satisfy the Module contract");
static_assert(!NN::ModuleContainer<Upsample2d>, "Upsample2d is a leaf, not a container");

namespace
{
    /** @brief Returns the traced shape of @p layer's result with its batch axis dropped. */
    Shape traced_per_sample_shape(const Upsample2d& layer, const Shape& batched_input)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueId x = builder.input(batched_input, "x");
        const Shape traced = graph.value(layer.forward(builder, x)).shape;
        return Shape{traced[1], traced[2], traced[3]};
    }

    /** @brief Holds a scale whose product with an extent of 5 exceeds 64 bits. */
    constexpr int64_t K_OVERFLOWING_SCALE = int64_t{1} << 62;
} // namespace

/** @brief Tests that the default scale doubles height and width and keeps batch and channels. */
TEST(Unit_Upsample2d, TheDefaultScaleDoublesHeightAndWidth)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Upsample2d layer{};
    const Graph::ValueId x = builder.input(Shape{2, 3, 5, 7}, "x");
    const Graph::ValueId y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 6, 10, 7}));
}

/** @brief Tests that the scale field is the factor applied to both spatial axes. */
TEST(Unit_Upsample2d, TheScaleFieldSetsTheFactorOnBothSpatialAxes)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Upsample2d layer{3};
    const Graph::ValueId x = builder.input(Shape{1, 4, 2, 5}, "x");
    const Graph::ValueId y = layer.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{1, 12, 6, 5}));
}

/** @brief Tests that the traced result carries the NHWC layout tag. */
TEST(Unit_Upsample2d, TheResultIsTaggedNhwc)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Upsample2d layer{};
    const Graph::ValueId x = builder.input(Shape{2, 3, 5, 7}, "x");
    const Graph::ValueId y = layer.forward(builder, x);

    ASSERT_EQ(graph.value(y).shape.rank(), 4);
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::Nhwc);
}

/** @brief Tests that forward emits one UPSAMPLE2D node that reads x with scale on both axes. */
TEST(Unit_Upsample2d, ForwardEmitsOneUpsampleNodeCarryingTheScale)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Upsample2d layer{3};
    const Graph::ValueId x = builder.input(Shape{1, 2, 2, 3}, "x");
    const std::size_t nodes_before = graph.node_count();
    const Graph::ValueId y = layer.forward(builder, x);

    ASSERT_EQ(graph.node_count(), nodes_before + 1);
    const Graph::Node& node = graph.node(graph.value(y).producer);
    EXPECT_EQ(node.op, Ops::K_UPSAMPLE2D.id);
    EXPECT_EQ(node.inputs, (std::vector<Graph::ValueId>{x}));
    EXPECT_DOUBLE_EQ(node.attributes.get(0), 3.0);
    EXPECT_DOUBLE_EQ(node.attributes.get(1), 3.0);
}

/** @brief Tests that output_shape maps [H, W, C] to [H * scale, W * scale, C]. */
TEST(Unit_Upsample2d, OutputShapeScalesHeightAndWidthOnly)
{
    EXPECT_EQ(Upsample2d{}.output_shape(Shape{3, 5, 7}), (Shape{6, 10, 7}));
    EXPECT_EQ(Upsample2d{4}.output_shape(Shape{2, 1, 9}), (Shape{8, 4, 9}));
}

/** @brief Tests that output_shape equals the traced shape without its batch axis. */
TEST(Unit_Upsample2d, OutputShapeAgreesWithTheTracedShape)
{
    const Upsample2d doubling{};
    EXPECT_EQ(doubling.output_shape(Shape{3, 5, 7}),
              traced_per_sample_shape(doubling, Shape{2, 3, 5, 7}));

    const Upsample2d tripling{3};
    EXPECT_EQ(tripling.output_shape(Shape{4, 2, 5}),
              traced_per_sample_shape(tripling, Shape{1, 4, 2, 5}));

    const Upsample2d identity{1};
    EXPECT_EQ(identity.output_shape(Shape{6, 6, 1}),
              traced_per_sample_shape(identity, Shape{5, 6, 6, 1}));
}

/** @brief Tests that output_shape rejects a scale of zero in a message naming layer and scale. */
TEST(Unit_Upsample2d, OutputShapeRejectsAZeroScale)
{
    SA_EXPECT_ERROR((void)Upsample2d{0}.output_shape(Shape{3, 5, 7}),
                    "Upsample2d", "[3, 5, 7] with scale=0");
}

/** @brief Tests that output_shape rejects a negative scale in a message naming layer and scale. */
TEST(Unit_Upsample2d, OutputShapeRejectsANegativeScale)
{
    SA_EXPECT_ERROR((void)Upsample2d{-2}.output_shape(Shape{3, 5, 7}),
                    "Upsample2d", "[3, 5, 7] with scale=-2");
}

/** @brief Tests that output_shape rejects a feature vector in a message naming layer and shape. */
TEST(Unit_Upsample2d, OutputShapeRejectsARankOnePerSampleShape)
{
    SA_EXPECT_ERROR((void)Upsample2d{}.output_shape(Shape{16}), "Upsample2d", "[H, W, C]", "[16]");
}

/** @brief Tests that output_shape rejects a height whose scaled extent overflows 64 bits. */
TEST(Unit_Upsample2d, OutputShapeRejectsAScaledHeightThatOverflows)
{
    const Upsample2d layer{K_OVERFLOWING_SCALE};

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{5, 1, 7}), "Upsample2d", "fit in 64 bits");
}

/** @brief Tests that output_shape throws for a shape that still carries its batch axis. */
TEST(Unit_Upsample2d, OutputShapeRejectsARankFourShape)
{
    const Upsample2d layer{};
    EXPECT_THROW((void)layer.output_shape(Shape{2, 3, 5, 7}), Error);
}

/** @brief Tests that forward rejects a scale of zero in a message naming layer and scale. */
TEST(Unit_Upsample2d, ForwardRejectsAZeroScale)
{
    SA_EXPECT_ERROR((void)traced_per_sample_shape(Upsample2d{0}, Shape{1, 2, 2, 3}),
                    "Upsample2d", "[2, 2, 3] with scale=0");
}

/** @brief Tests that forward rejects a rank-2 input in a message naming layer and shape. */
TEST(Unit_Upsample2d, ForwardRejectsARankTwoInput)
{
    SA_EXPECT_ERROR((void)traced_per_sample_shape(Upsample2d{}, Shape{2, 12}),
                    "Upsample2d", "rank-4", "[2, 12]");
}

/** @brief Tests that forward rejects a width whose scaled extent overflows 64 bits. */
TEST(Unit_Upsample2d, ForwardRejectsAScaledWidthThatOverflows)
{
    const Upsample2d layer{K_OVERFLOWING_SCALE};

    SA_EXPECT_ERROR((void)traced_per_sample_shape(layer, Shape{1, 1, 5, 3}),
                    "Upsample2d", "fit in 64 bits");
}

/** @brief Tests that forward names the layout it reads for a rank-4 input tagged NCHW. */
TEST(Unit_Upsample2d, ForwardRejectsAnNchwInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Upsample2d layer{};
    const Graph::ValueId x = builder.input(Shape{1, 3, 2, 2}, "x");
    const Graph::ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);

    SA_EXPECT_ERROR((void)layer.forward(builder, x_nchw), "upsample2d", "NHWC");
}

/** @brief Tests that the parameter walk visits nothing after a trace. */
TEST(Unit_Upsample2d, TheParameterWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Upsample2d layer{};
    const Graph::ValueId x = builder.input(Shape{1, 2, 2, 3}, "x");
    (void)layer.forward(builder, x);

    std::size_t visited = 0;
    NN::for_each_parameter(layer, [&visited](const std::string&, NN::Parameter&)
    {
        ++visited;
    });

    EXPECT_EQ(visited, 0u);
    EXPECT_EQ(NN::parameter_count(layer), 0u);
}
