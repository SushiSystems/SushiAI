/**************************************************************************/
/* test_leaky_relu.cpp                                                    */
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
 * @file test_leaky_relu.cpp
 * @brief Tests the LeakyReLU layer's trace, shape inference and slope range.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/leaky_relu.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using NN::LeakyReLU;

static_assert(NN::Module<LeakyReLU>, "LeakyReLU must satisfy the Module contract");
static_assert(!NN::ModuleContainer<LeakyReLU>, "LeakyReLU is a leaf, not a container");
static_assert(std::tuple_size<decltype(LeakyReLU::fields())>::value == 0,
              "LeakyReLU must declare no parameter field");

namespace
{
    /** @brief Reports whether @p batched is @p sample with one batch axis in front. */
    bool is_batch_of(const Shape& batched, const Shape& sample)
    {
        if (batched.rank() != sample.rank() + 1)
            return false;

        for (int32_t axis = 0; axis < sample.rank(); ++axis)
        {
            if (batched[axis + 1] != sample[axis])
                return false;
        }
        return true;
    }

    /** @brief Reports whether tracing a layer of @p slope over a [2, 4] input is refused. */
    bool is_slope_rejected(double slope)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

        const LeakyReLU layer{slope};
        return layer.forward(builder, x).status().code() == StatusCode::InvalidDescription;
    }
} // namespace

/** @brief Tests that tracing emits one LEAKY_RELU node reading the input with the slope. */
TEST(Small_LeakyReLU, ForwardEmitsOneLeakyReLUNodeCarryingTheSlope)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    const std::size_t nodes_before = graph.node_count();

    const LeakyReLU layer{0.2};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    ASSERT_EQ(graph.node_count(), nodes_before + 1);
    const Graph::Node& node = graph.nodes().back();
    EXPECT_EQ(node.op, Ops::K_LEAKY_RELU.id);
    ASSERT_EQ(node.inputs.size(), 1u);
    EXPECT_EQ(node.inputs.front(), x);
    ASSERT_EQ(node.outputs.size(), 1u);
    EXPECT_EQ(node.outputs.front(), y);
    EXPECT_DOUBLE_EQ(node.attributes.get(0), 0.2);
}

namespace
{
/** @brief Holds an input, whether it is NCHW, and what the result must keep of it. */
struct KeptRow
{
    std::string name;               /**< The row's name. */
    Shape input;                    /**< The input shape, batch axis included. */
    bool nchw;                      /**< Whether the input is converted to NCHW first. */
    bool states_layout;             /**< Whether the row states the layout the result carries. */
    Graph::TensorLayout layout;     /**< The layout the result carries, when the row states it. */
};
} // namespace

namespace
{
/** @brief Runs the shape, type and layout promise over each kind of input. */
class Small_LeakyReLUKept : public ::testing::TestWithParam<KeptRow>
{
};
} // namespace

/** @brief Tests that the result is shaped, typed and laid out like the input. */
TEST_P(Small_LeakyReLUKept, ForwardKeepsTheShapeTypeAndLayoutOfItsInput)
{
    const KeptRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Graph::ValueID x = builder.input(row.input, "x");
    if (row.nchw)
    {
        SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID converted,
                                builder.layout_convert(x, Graph::TensorLayout::NCHW));
        x = converted;
    }
    const LeakyReLU layer{};

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, graph.value(x).shape);
    EXPECT_EQ(graph.value(y).dtype, graph.value(x).dtype);
    EXPECT_EQ(graph.value(y).layout, graph.value(x).layout);
    if (row.states_layout)
        EXPECT_EQ(graph.value(y).layout, row.layout);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LeakyReLUKept,
    ::testing::Values(
        KeptRow{"AFeatureBatch", Shape{2, 4}, false, false, Graph::TensorLayout::NHWC},
        KeptRow{"AnImageBatch", Shape{2, 8, 6, 3}, false, true, Graph::TensorLayout::NHWC},
        KeptRow{"AnNCHWImageBatch", Shape{2, 8, 6, 3}, true, true, Graph::TensorLayout::NCHW}),
    [](const ::testing::TestParamInfo<KeptRow>& info) { return info.param.name; });

namespace
{
/** @brief Holds a batched input and the per-sample shape output_shape gives. */
struct SampleRow
{
    std::string name; /**< The row's name. */
    Shape batched;    /**< The input shape, batch axis included. */
    Shape sample;     /**< The per-sample shape. */
};
} // namespace

namespace
{
/** @brief Runs the shape agreement over each sample shape. */
class Small_LeakyReLUSample : public ::testing::TestWithParam<SampleRow>
{
};
} // namespace

/** @brief Tests that output_shape() is the per-sample shape and matches the traced shape. */
TEST_P(Small_LeakyReLUSample, OutputShapeAgreesWithForward)
{
    const SampleRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(row.batched, "x");
    const LeakyReLU layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    const Shape inferred = layer.output_shape(row.sample).value();

    EXPECT_EQ(inferred, row.sample);
    EXPECT_TRUE(is_batch_of(graph.value(y).shape, inferred));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LeakyReLUSample,
    ::testing::Values(SampleRow{"AFeatureVector", Shape{5, 7}, Shape{7}},
                      SampleRow{"AnImage", Shape{3, 8, 6, 4}, Shape{8, 6, 4}}),
    [](const ::testing::TestParamInfo<SampleRow>& info) { return info.param.name; });

namespace
{
/** @brief Holds a slope outside the open range (0, 1). */
struct SlopeRow
{
    std::string name; /**< The row's name. */
    double slope;     /**< The slope both forward and output_shape refuse. */
};
} // namespace

namespace
{
/** @brief Runs the slope refusal over each slope outside the open range. */
class Small_LeakyReLUSlope : public ::testing::TestWithParam<SlopeRow>
{
};
} // namespace

/** @brief Tests that forward() refuses the slope in the layer's own message. */
TEST_P(Small_LeakyReLUSlope, ForwardRefusesTheSlope)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    const LeakyReLU layer{GetParam().slope};

    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription, "LeakyReLU",
                      "slope");
}

/** @brief Tests that output_shape() refuses the slope forward() refuses and names it. */
TEST_P(Small_LeakyReLUSlope, OutputShapeRefusesTheSlope)
{
    SA_EXPECT_FAILURE(LeakyReLU{GetParam().slope}.output_shape(Shape{4}),
                      StatusCode::InvalidDescription, "LeakyReLU", "slope");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LeakyReLUSlope,
    ::testing::Values(SlopeRow{"Zero", 0.0}, SlopeRow{"One", 1.0}, SlopeRow{"Negative", -0.01},
                      SlopeRow{"AboveOne", 1.5},
                      SlopeRow{"NaN", std::numeric_limits<double>::quiet_NaN()}),
    [](const ::testing::TestParamInfo<SlopeRow>& info) { return info.param.name; });

/** @brief Tests that slopes a float32 step inside each end of the open range are accepted. */
TEST(Small_LeakyReLU, ForwardAcceptsSlopesJustInsideTheOpenRange)
{
    EXPECT_FALSE(is_slope_rejected(std::numeric_limits<float>::min()));
    EXPECT_FALSE(is_slope_rejected(1.0 - std::numeric_limits<float>::epsilon()));
}

/** @brief Tests that a rejected slope leaves no node in the graph. */
TEST(Small_LeakyReLU, ARejectedSlopeEmitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    const std::size_t values_before = graph.value_count();
    const std::size_t nodes_before = graph.node_count();
    const LeakyReLU layer{1.0};

    SA_EXPECT_FAILURE(layer.forward(builder, x), StatusCode::InvalidDescription);

    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_EQ(graph.value_count(), values_before);
}
