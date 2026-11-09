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
 * @file test_unflatten.cpp
 * @brief Tests the trace, the shape rule and the rejected inputs of the Unflatten layer.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/unflatten.hpp>

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

/** @brief Tests that the traced result is [N, height, width, channels] with the input's dtype. */
TEST(Small_Unflatten, TracesTheImageShape)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{4, 5, 3};
    const Graph::ValueID x = builder.input(Shape{2, 60}, "x", DType::Float64);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 5, 3}));
    EXPECT_EQ(graph.value(y).dtype, DType::Float64);
}

/** @brief Tests that forward() leaves the default NHWC tag on a result finalise() accepts. */
TEST(Small_Unflatten, TagsTheResultNHWC)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{4, 5, 3};
    const Graph::ValueID x = builder.input(Shape{2, 60}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    ASSERT_EQ(graph.value(y).shape.rank(), 4);
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
    SA_EXPECT_OK(graph.finalise());
}

/** @brief Holds a layer's image extents, the flat input it takes and the image it gives. */
struct ImageRow
{
    std::string name; /**< The row's name. */
    int64_t height;   /**< The height of the image. */
    int64_t width;    /**< The width of the image. */
    int64_t channels; /**< The channels of the image. */
    int64_t batch;    /**< The batch extent of the trace. */
};

/** @brief Runs the shape agreement over each image size. */
class Small_UnflattenImage : public ::testing::TestWithParam<ImageRow>
{
};

/** @brief Tests that output_shape() equals the image and the traced per-sample shape. */
TEST_P(Small_UnflattenImage, OutputShapeAgreesWithTheTrace)
{
    const ImageRow& row = GetParam();
    const int64_t features = row.height * row.width * row.channels;
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Unflatten layer{row.height, row.width, row.channels};
    const Graph::ValueID x = builder.input(Shape{row.batch, features}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(layer.output_shape(per_sample(graph, x)).value(), per_sample(graph, y));
    EXPECT_EQ(layer.output_shape(Shape{features}).value(),
              (Shape{row.height, row.width, row.channels}));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_UnflattenImage,
    ::testing::Values(ImageRow{"AnRGBImage", 4, 5, 3, 2}, ImageRow{"AGreyImage", 28, 28, 1, 7}),
    [](const ::testing::TestParamInfo<ImageRow>& info) { return info.param.name; });

/** @brief Holds extents that are not all positive. */
struct ExtentRow
{
    std::string name; /**< The row's name. */
    int64_t height;   /**< The height given to the layer. */
    int64_t width;    /**< The width given to the layer. */
    int64_t channels; /**< The channels given to the layer. */
};

/** @brief Runs the extent refusal over each extent that is not positive. */
class Small_UnflattenExtent : public ::testing::TestWithParam<ExtentRow>
{
};

/** @brief Tests that forward() returns the layer's positive-extent error. */
TEST_P(Small_UnflattenExtent, ForwardRejectsTheExtent)
{
    const ExtentRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Unflatten layer{row.height, row.width, row.channels};
    const Graph::ValueID x = builder.input(Shape{2, 60}, "x");

    // The count check or the shape constructor would also refuse it, without naming the extents.
    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription,
                      "Unflatten needs positive extents");
}

/** @brief Tests that output_shape() returns the layer's positive-extent failure. */
TEST_P(Small_UnflattenExtent, OutputShapeRejectsTheExtent)
{
    const ExtentRow& row = GetParam();
    const Unflatten layer{row.height, row.width, row.channels};

    SA_EXPECT_FAILURE(layer.output_shape(Shape{60}), StatusCode::InvalidDescription,
                      "Unflatten needs positive extents");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_UnflattenExtent,
    ::testing::Values(ExtentRow{"AZeroExtent", 4, 0, 3}, ExtentRow{"NegativeExtents", -4, -5, 3}),
    [](const ::testing::TestParamInfo<ExtentRow>& info) { return info.param.name; });

/** @brief Tests that forward() throws on a rank-4 input whose element count would fit. */
TEST(Small_Unflatten, ForwardRejectsAnInputThatIsNotRankTwo)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    // [2, 60, 1, 1] has inner extent 60 and 120 elements, so only the rank check can throw.
    const Unflatten layer{4, 5, 3};
    const Graph::ValueID x = builder.input(Shape{2, 60, 1, 1}, "x");

    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription);
}

/** @brief Tests that forward() throws the layer's own error for a mismatched inner extent. */
TEST(Small_Unflatten, ForwardRejectsAMismatchedElementCount)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{4, 5, 3};
    const Graph::ValueID x = builder.input(Shape{3, 40}, "x");

    // reshape() would also reject it, with a message that does not name the layer.
    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription,
                      "Unflatten(4, 5, 3)", "inner extent is 60");
}

/** @brief Tests that forward() throws on an input tagged NCHW. */
TEST(Small_Unflatten, ForwardRejectsAnNCHWInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Unflatten layer{4, 5, 3};
    const Graph::ValueID x = builder.input(Shape{2, 60}, "x");
    graph.value(x).layout = Graph::TensorLayout::NCHW;

    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription);
}

/** @brief Tests that output_shape() fails on a per-sample shape that is not rank 1. */
TEST(Small_Unflatten, OutputShapeRejectsAShapeThatIsNotRankOne)
{
    const Unflatten layer{4, 5, 3};

    SA_EXPECT_FAILURE(layer.output_shape(Shape{60, 1}), StatusCode::InvalidDescription);
}

/** @brief Tests that output_shape() fails when the feature count does not match. */
TEST(Small_Unflatten, OutputShapeRejectsAMismatchedElementCount)
{
    const Unflatten layer{4, 5, 3};

    SA_EXPECT_FAILURE(layer.output_shape(Shape{59}), StatusCode::InvalidDescription);
}

/** @brief Tests that the parameter walk visits nothing before and after a trace. */
TEST(Small_Unflatten, HasNoParameters)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Unflatten layer{4, 5, 3};
    EXPECT_EQ(NN::parameter_count(layer), 0u);

    const Graph::ValueID x = builder.input(Shape{2, 60}, "x");
    SA_EXPECT_OK(layer.forward(builder, x));

    EXPECT_EQ(NN::parameter_count(layer), 0u);
    EXPECT_EQ(NN::parameter_elements(layer), 0);
}
