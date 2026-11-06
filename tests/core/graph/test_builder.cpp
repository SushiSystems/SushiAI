/**************************************************************************/
/* test_builder.cpp                                                       */
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
 * @file test_builder.cpp
 * @brief Tests GraphBuilder: the nodes each emitter writes and the operands it refuses, with the
 *        values that disagree.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/graph/traced.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::ValueID;
using Graph::Traced;
using Graph::LayerNormGrads;
using Graph::LayerNormResult;

namespace
{
    /**
     * @brief Returns the node that writes @p value.
     *
     * Every emitter adds the node that writes its result, so the producer exists.
     */
    const Graph::Node& producer_of(const Graph::Graph& graph, ValueID value)
    {
        return graph.node(graph.value(value).producer);
    }

    /** @brief Expects @p node to read @p inputs and to write exactly one value, @p out. */
    void expect_reads_and_writes(const Graph::Node& node, ValueID out,
                                 const std::vector<ValueID>& inputs)
    {
        EXPECT_EQ(node.inputs, inputs);
        ASSERT_EQ(node.outputs.size(), 1u);
        EXPECT_EQ(node.outputs[0], out);
    }

    /** @brief Names a case of a parameterised suite after the row it runs. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return std::string(info.param.name);
    }

    /** @brief Names a case of a suite whose rows are operations after the operation. */
    std::string operation_name(const ::testing::TestParamInfo<OpID>& info)
    {
        std::string name(Ops::name_of(info.param));
        name.replace(0, name.find('.') + 1, "");
        return name;
    }

    /** @brief Holds a label and one shape, for a case that differs in that shape alone. */
    struct ShapeRow
    {
        std::string_view name; /**< The label of the case. */
        Shape shape;           /**< The shape under test. */
    };

    /** @brief Holds a label and two shapes, for a case that differs in that pair alone. */
    struct ShapePairRow
    {
        std::string_view name; /**< The label of the case. */
        Shape first;           /**< The first operand's shape. */
        Shape second;          /**< The second operand's shape. */
    };

    /** @brief Holds a label and an element type, for a case that differs in that type alone. */
    struct DTypeRow
    {
        std::string_view name; /**< The label of the case. */
        DType dtype;           /**< The element type under test. */
    };
}

/** @brief Tests that a rank-2 product still yields [m, n] and keeps the flags it was given. */
TEST(Small_BatchedMatmul, Rank2ProductKeepsItsShapeAndFlags)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID a = builder.input(Shape{3, 4}, "a");
    const ValueID b = builder.input(Shape{5, 4}, "b");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, builder.matmul(a, b, false, true));

    EXPECT_EQ(graph.value(out).shape, Shape({3, 5}));
    const Graph::Node& node = producer_of(graph, out);
    EXPECT_EQ(node.op, Ops::K_MATMUL.id);
    EXPECT_FALSE(node.attributes.flag(0));
    EXPECT_TRUE(node.attributes.flag(1));
}

namespace
{
    /** @brief Names one pair of transpose flags. */
    struct FlagPair
    {
        std::string_view name; /**< The label of the case. */
        bool transpose_a;      /**< Whether the first operand is transposed. */
        bool transpose_b;      /**< Whether the second operand is transposed. */
    };

    /** @brief Lists the four pairs of transpose flags. */
    const std::array<FlagPair, 4> K_FLAG_PAIRS = {{
        {"NeitherTransposed", false, false},
        {"FirstTransposed", true, false},
        {"SecondTransposed", false, true},
        {"BothTransposed", true, true},
    }};

    /** @brief Runs once per pair of transpose flags. */
    class Small_BatchedMatmulFlags : public ::testing::TestWithParam<FlagPair>
    {

    };
}

/** @brief Tests that a rank-3 product yields [B, m, n] under the pair of flags it was given. */
TEST_P(Small_BatchedMatmulFlags, EachFlagPairYieldsABatchedProduct)
{
    const FlagPair flags = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    // A is [B, m, k] or [B, k, m]; B is [B, k, n] or [B, n, k]; here m=3, k=4, n=5, B=2.
    const ValueID a = builder.input(flags.transpose_a ? Shape{2, 4, 3} : Shape{2, 3, 4}, "a");
    const ValueID b = builder.input(flags.transpose_b ? Shape{2, 5, 4} : Shape{2, 4, 5}, "b");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out,
                            builder.matmul(a, b, flags.transpose_a, flags.transpose_b));

    EXPECT_EQ(graph.value(out).shape, Shape({2, 3, 5}));
    const Graph::Node& node = producer_of(graph, out);
    EXPECT_EQ(node.op, Ops::K_MATMUL.id);
    expect_reads_and_writes(node, out, {a, b});
    EXPECT_EQ(node.attributes.flag(0), flags.transpose_a);
    EXPECT_EQ(node.attributes.flag(1), flags.transpose_b);
}

INSTANTIATE_TEST_SUITE_P(, Small_BatchedMatmulFlags, ::testing::ValuesIn(K_FLAG_PAIRS),
                         row_name<FlagPair>);

/** @brief Tests that a rank-3 product takes its batch count from the operands' shared axis. */
TEST(Small_BatchedMatmul, BatchedProductTakesTheSharedBatchCount)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID a = builder.input(Shape{7, 2, 3}, "a");
    const ValueID b = builder.input(Shape{7, 3, 6}, "b");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, builder.matmul(a, b));

    EXPECT_EQ(graph.value(out).shape, Shape({7, 2, 6}));
    EXPECT_EQ(graph.value(out).layout, graph.value(a).layout);
    EXPECT_EQ(graph.value(out).dtype, DType::Float32);
}

namespace
{
    /** @brief Lists the operand pairs whose ranks differ or are neither 2 nor 3. */
    const std::array<ShapePairRow, 4> K_RANK_REFUSALS = {{
        {"RankTwoThenRankThree", Shape{3, 4}, Shape{2, 4, 5}},
        {"RankThreeThenRankTwo", Shape{2, 4, 5}, Shape{3, 4}},
        {"RankOneThenRankThree", Shape{4}, Shape{2, 4, 5}},
        {"TwoRankFour", Shape{2, 3, 4, 1}, Shape{2, 4, 5, 1}},
    }};

    /** @brief Runs once per operand pair a product refuses for its ranks. */
    class Small_BatchedMatmulRankRefusal : public ::testing::TestWithParam<ShapePairRow>
    {

    };
}

/** @brief Tests that a product refuses operands whose ranks differ or are neither 2 nor 3. */
TEST_P(Small_BatchedMatmulRankRefusal, RefusesMixedRanksAndRanksOutsideTwoAndThree)
{
    const ShapePairRow& pair = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID first = builder.input(pair.first, "first");
    const ValueID second = builder.input(pair.second, "second");
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(builder.matmul(first, second), StatusCode::InvalidDescription);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

INSTANTIATE_TEST_SUITE_P(, Small_BatchedMatmulRankRefusal, ::testing::ValuesIn(K_RANK_REFUSALS),
                         row_name<ShapePairRow>);

/** @brief Tests that a rank-3 product refuses operands whose batches or inner extents differ. */
TEST(Small_BatchedMatmul, RefusesUnequalBatchesAndInnerExtents)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID a = builder.input(Shape{2, 3, 4}, "a");
    const ValueID other_batch = builder.input(Shape{3, 4, 5}, "other_batch");
    const ValueID wrong_inner = builder.input(Shape{2, 5, 3}, "wrong_inner");
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(builder.matmul(a, other_batch), StatusCode::InvalidDescription);
    SA_EXPECT_FAILURE(builder.matmul(a, wrong_inner), StatusCode::InvalidDescription);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

/** @brief Tests that a rank-2 softmax yields a result of its operand's [rows, last] shape. */
TEST(Small_BatchedMatmul, SoftmaxOverRank2KeepsItsRowsAndLastAxis)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 5}, "x");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, builder.softmax(x));

    EXPECT_EQ(graph.value(out).shape, Shape({4, 5}));
}

/** @brief Tests that a rank-3 softmax yields a result of its operand's shape. */
TEST(Small_BatchedMatmul, SoftmaxOverRank3KeepsItsShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3, 3}, "x");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, builder.softmax(x));

    EXPECT_EQ(graph.value(out).shape, Shape({2, 3, 3}));
}

/** @brief Tests that softmax refuses an operand of rank below two. */
TEST(Small_BatchedMatmul, SoftmaxRefusesRankBelowTwo)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID vector = builder.input(Shape{5}, "vector");
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(builder.softmax(vector), StatusCode::InvalidDescription);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

namespace
{
    /** @brief Holds the value every case writes above the diagonal. */
    constexpr double K_FILL = -1.0e9;

    /** @brief Expects @p refused to name @p part, and the graph to hold no node beyond @p nodes. */
    void expect_refused(const Graph::Graph& graph, const Traced& refused, std::size_t nodes,
                        const char* part)
    {
        SA_EXPECT_FAILURE(refused, StatusCode::InvalidDescription, part);
        EXPECT_EQ(graph.node_count(), nodes) << "a refused call left a node behind";
        EXPECT_TRUE(graph.has_refusal());
    }
}

/** @brief Tests that the mask is one node over its scores, with the fill in attribute slot 0. */
TEST(Small_CausalMaskOps, TheMaskWritesItsIdentity)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID scores = builder.input(Shape{2, 5, 5}, "scores", DType::Float64);

    SA_ASSERT_OK_AND_ASSIGN(const Traced out, builder.causal_mask(scores, K_FILL));

    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());
    const Graph::Node& node = graph.node(graph.value(out.value()).producer);
    EXPECT_EQ(node.op, Ops::K_CAUSAL_MASK.id);
    EXPECT_EQ(node.inputs, (std::vector<ValueID>{scores}));
    EXPECT_EQ(node.outputs, (std::vector<ValueID>{out.value()}));
    EXPECT_EQ(node.attributes.values[0], K_FILL);
    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that the mask keeps the shape, element type and lifetime of its scores. */
TEST(Small_CausalMaskOps, TheMaskKeepsShapeDTypeAndLifetime)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID scores = builder.input(Shape{2, 5, 5}, "scores", DType::Float64);

    SA_ASSERT_OK_AND_ASSIGN(const Traced out, builder.causal_mask(scores, K_FILL));

    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(graph.value(out.value()).shape, (Shape{2, 5, 5}));
    EXPECT_EQ(graph.value(out.value()).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out.value()).lifetime, Graph::Lifetime::Activation);
}

/** @brief Tests that a single matrix and a value with two leading axes are both accepted. */
TEST(Small_CausalMaskOps, TheMaskAcceptsAnyNumberOfLeadingAxes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID matrix = builder.input(Shape{3, 3}, "matrix");
    const ValueID heads = builder.input(Shape{2, 4, 3, 3}, "heads");

    SA_ASSERT_OK_AND_ASSIGN(const Traced one, builder.causal_mask(matrix, K_FILL));
    SA_ASSERT_OK_AND_ASSIGN(const Traced many, builder.causal_mask(heads, K_FILL));

    ASSERT_TRUE(one.has_value());
    ASSERT_TRUE(many.has_value());
    EXPECT_EQ(graph.value(one.value()).shape, (Shape{3, 3}));
    EXPECT_EQ(graph.value(many.value()).shape, (Shape{2, 4, 3, 3}));
}

namespace
{
    /** @brief Holds one operand set a causal mask refuses and the part its message names. */
    struct MaskRefusal
    {
        std::string_view name; /**< The label of the case. */
        Shape scores;          /**< The shape of the scores. */
        DType dtype;           /**< The element type of the scores. */
        double fill;           /**< The fill written above the diagonal. */
        const char* part;      /**< The part the refusal names. */
    };

    /** @brief Lists the operand sets a causal mask refuses. */
    const std::array<MaskRefusal, 4> K_MASK_REFUSALS = {{
        {"Vector", Shape{5}, DType::Float32, K_FILL, "[..., T, T] scores"},
        {"OblongScores", Shape{2, 4, 5}, DType::Float32, K_FILL, "[2, 4, 5]"},
        {"IntegerScores", Shape{5, 5}, DType::Int32, K_FILL, "half, float32 or float64 scores"},
        {"InfiniteFill", Shape{5, 5}, DType::Float32, -std::numeric_limits<double>::infinity(),
         "finite fill"},
    }};

    /** @brief Runs once per operand set a causal mask refuses. */
    class Small_CausalMaskRefusal : public ::testing::TestWithParam<MaskRefusal>
    {

    };
}

/** @brief Tests that the mask refuses a vector, an oblong matrix, an integer, an infinite fill. */
TEST_P(Small_CausalMaskRefusal, TheMaskRefusesOperandsThatDoNotFit)
{
    const MaskRefusal& refusal = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID scores = builder.input(refusal.scores, "scores", refusal.dtype);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.causal_mask(scores, refusal.fill), nodes, refusal.part);
}

INSTANTIATE_TEST_SUITE_P(, Small_CausalMaskRefusal, ::testing::ValuesIn(K_MASK_REFUSALS),
                         row_name<MaskRefusal>);

namespace
{
    /** @brief The geometry most tests share: a 3x3 window, unit stride and one pixel of padding. */
    Graph::ConvGeometry same_3x3()
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3;
        g.kernel_w = 3;
        g.pad_top = 1;
        g.pad_left = 1;
        return g;
    }

    /** @brief Declares a rank-4 input and converts it to NCHW; the extents survive the permute. */
    Traced nchw_input(const GraphBuilder& builder, std::string_view debug_name,
                       DType dtype = DType::Float32)
    {
        const ValueID nhwc = builder.input(Shape{2, 5, 5, 3}, debug_name, dtype);
        return builder.layout_convert(nhwc, Graph::TensorLayout::NCHW);
    }

    /** @brief Expects @p out to be the single result of an @p op node that reads @p inputs. */
    void expect_produced_by(const Graph::Graph& graph, ValueID out, OpID op,
                            const std::vector<ValueID>& inputs)
    {
        const Graph::NodeID producer = graph.value(out).producer;
        ASSERT_NE(producer, Graph::K_NO_NODE);

        const Graph::Node& node = graph.node(producer);
        EXPECT_EQ(node.op, op) << "written by " << Ops::name_of(node.op);
        EXPECT_EQ(node.inputs, inputs);
        ASSERT_EQ(node.outputs.size(), 1u);
        EXPECT_EQ(node.outputs[0], out);
    }
}

/** @brief Tests that the forward emitter writes its identity over [input, taps] in that order. */
TEST(Small_DepthwiseOps, DepthwiseConv2dWritesItsIdentityOverInputAndTapsInOrder)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 5, 5, 3}, "x");
    const ValueID w = builder.parameter(Shape{3, 3, 3}, "w");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, builder.depthwise_conv2d(x, w, same_3x3()));

    expect_produced_by(graph, out, Ops::K_DEPTHWISE_CONV2D.id, {x, w});
}

/** @brief Tests that the forward emitter yields the window grid over the padded input. */
TEST(Small_DepthwiseOps, DepthwiseConv2dYieldsTheWindowGridOverThePaddedInput)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 5, 5, 3}, "x");
    const ValueID w = builder.parameter(Shape{3, 3, 3}, "w");

    Graph::ConvGeometry unit = same_3x3();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID unit_out, builder.depthwise_conv2d(x, w, unit));
    EXPECT_EQ(graph.value(unit_out).shape, Shape({2, 5, 5, 3}));

    Graph::ConvGeometry strided = same_3x3();
    strided.stride_h = 2;
    strided.stride_w = 2;
    SA_ASSERT_OK_AND_ASSIGN(const ValueID strided_out, builder.depthwise_conv2d(x, w, strided));
    EXPECT_EQ(graph.value(strided_out).shape, Shape({2, 3, 3, 3}));
}

/** @brief Tests that the forward emitter keeps the input's layout, element type and lifetime. */
TEST(Small_DepthwiseOps, DepthwiseConv2dKeepsLayoutDTypeAndLifetime)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 5, 5, 3}, "x", DType::Float64);
    const ValueID w = builder.parameter(Shape{3, 3, 3}, "w", DType::Float64);

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, builder.depthwise_conv2d(x, w, same_3x3()));

    EXPECT_EQ(graph.value(out).layout, Graph::TensorLayout::NHWC);
    EXPECT_EQ(graph.value(out).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out).lifetime, Graph::Lifetime::Activation);
}

namespace
{
    /** @brief Holds one geometry slot, the label of what it holds and the value it must hold. */
    struct GeometrySlot
    {
        std::string_view name; /**< The label of the case. */
        std::size_t slot;      /**< The attribute slot. */
        double expected;       /**< The value the slot holds for the geometry of the test. */
    };

    /** @brief Lists the eight slots for a 3x5 window with padding 1 on top and 2 on the left. */
    const std::array<GeometrySlot, 8> K_GEOMETRY_SLOTS = {{
        {"KernelHeight", 0, 3.0},
        {"KernelWidth", 1, 5.0},
        {"StrideHeight", 2, 1.0},
        {"StrideWidth", 3, 1.0},
        {"DilationHeight", 4, 1.0},
        {"DilationWidth", 5, 1.0},
        {"PaddingTop", 6, 1.0},
        {"PaddingLeft", 7, 2.0},
    }};

    /** @brief Runs once per geometry slot. */
    class Small_DepthwiseGeometrySlot : public ::testing::TestWithParam<GeometrySlot>
    {

    };
}

/** @brief Tests that the forward emitter stores its geometry in the slots MAXPOOL2D also uses. */
TEST_P(Small_DepthwiseGeometrySlot, DepthwiseConv2dStoresItsGeometryInTheGeometrySlots)
{
    const GeometrySlot& row = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 7, 9, 3}, "x");
    const ValueID w = builder.parameter(Shape{3, 5, 3}, "w");

    Graph::ConvGeometry g;
    g.kernel_h = 3;
    g.kernel_w = 5;
    g.pad_top = 1;
    g.pad_left = 2;

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, builder.depthwise_conv2d(x, w, g));

    const Graph::Node& node = graph.node(graph.value(out).producer);
    EXPECT_EQ(node.attributes.values[row.slot], row.expected);
}

INSTANTIATE_TEST_SUITE_P(, Small_DepthwiseGeometrySlot, ::testing::ValuesIn(K_GEOMETRY_SLOTS),
                         row_name<GeometrySlot>);

/** @brief Tests that the forward emitter refuses taps of rank four. */
TEST(Small_DepthwiseOps, DepthwiseConv2dRefusesARankFourWeight)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 5, 5, 3}, "x");
    const ValueID w = builder.parameter(Shape{3, 3, 3, 1}, "w");
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(builder.depthwise_conv2d(x, w, same_3x3()), StatusCode::InvalidDescription);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

namespace
{
    /** @brief Lists the taps whose window or channel extent differs from the input's. */
    const std::array<ShapeRow, 3> K_TAP_REFUSALS = {{
        {"TallWindow", Shape{5, 3, 3}},
        {"WideWindow", Shape{3, 5, 3}},
        {"MoreChannels", Shape{3, 3, 4}},
    }};

    /** @brief Runs once per taps shape the forward emitter refuses. */
    class Small_DepthwiseTapsRefusal : public ::testing::TestWithParam<ShapeRow>
    {

    };
}

/** @brief Tests that the forward emitter refuses taps whose window or channel extent differs. */
TEST_P(Small_DepthwiseTapsRefusal, DepthwiseConv2dRefusesTapsOfAnotherWindowOrChannelCount)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 5, 5, 3}, "x");
    const ValueID taps = builder.parameter(GetParam().shape, "taps");
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(builder.depthwise_conv2d(x, taps, same_3x3()),
                      StatusCode::InvalidDescription);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

INSTANTIATE_TEST_SUITE_P(, Small_DepthwiseTapsRefusal, ::testing::ValuesIn(K_TAP_REFUSALS),
                         row_name<ShapeRow>);

/** @brief Tests that the forward emitter refuses an NCHW input. */
TEST(Small_DepthwiseOps, DepthwiseConv2dRefusesAnNCHWInput)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, nchw_input(builder, "x"));
    const ValueID w = builder.parameter(Shape{3, 3, 3}, "w");

    SA_EXPECT_FAILURE(builder.depthwise_conv2d(x, w, same_3x3()), StatusCode::InvalidDescription);
}

/** @brief Tests that the forward emitter refuses taps of another element type. */
TEST(Small_DepthwiseOps, DepthwiseConv2dRefusesTapsOfAnotherDType)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 5, 5, 3}, "x", DType::Float32);
    const ValueID w = builder.parameter(Shape{3, 3, 3}, "w", DType::Float64);

    SA_EXPECT_FAILURE(builder.depthwise_conv2d(x, w, same_3x3()), StatusCode::InvalidDescription);
}

/** @brief Tests that the forward emitter refuses a window that does not fit the padded input. */
TEST(Small_DepthwiseOps, DepthwiseConv2dRefusesAWindowThatDoesNotFit)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 2, 2, 3}, "x");
    const ValueID w = builder.parameter(Shape{3, 3, 3}, "w");

    Graph::ConvGeometry unpadded = same_3x3();
    unpadded.pad_top = 0;
    unpadded.pad_left = 0;

    SA_EXPECT_FAILURE(builder.depthwise_conv2d(x, w, unpadded), StatusCode::InvalidDescription);
}

/** @brief Tests that the input-gradient emitter writes its identity over [gradient, taps]. */
TEST(Small_DepthwiseOps, BackwardInputWritesItsIdentityOverGradientAndTapsInOrder)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID dy = builder.input(Shape{2, 5, 5, 3}, "dy");
    const ValueID w = builder.parameter(Shape{3, 3, 3}, "w");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID dx,
                            builder.depthwise_conv2d_backward_input(dy, w, Shape{2, 5, 5, 3},
                                                                    same_3x3()));

    expect_produced_by(graph, dx, Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT.id, {dy, w});
}

/** @brief Tests that the input-gradient emitter returns the forward input's shape. */
TEST(Small_DepthwiseOps, BackwardInputRecoversTheForwardInputShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID dy = builder.input(Shape{2, 3, 3, 3}, "dy");
    const ValueID w = builder.parameter(Shape{3, 3, 3}, "w");

    Graph::ConvGeometry strided = same_3x3();
    strided.stride_h = 2;
    strided.stride_w = 2;

    SA_ASSERT_OK_AND_ASSIGN(const ValueID dx,
                            builder.depthwise_conv2d_backward_input(dy, w, Shape{2, 5, 5, 3},
                                                                    strided));

    EXPECT_EQ(graph.value(dx).shape, Shape({2, 5, 5, 3}));
    EXPECT_EQ(graph.value(dx).layout, Graph::TensorLayout::NHWC);
    EXPECT_EQ(graph.value(dx).dtype, DType::Float32);
}

namespace
{
    /** @brief Lists the gradients that sit on another window grid than the input shape implies. */
    const std::array<ShapeRow, 3> K_GRID_REFUSALS = {{
        {"ShortGrid", Shape{2, 4, 5, 3}},
        {"WideChannels", Shape{2, 5, 5, 4}},
        {"Flat", Shape{2, 5, 3}},
    }};

    /** @brief Runs once per gradient shape the input-gradient emitter refuses. */
    class Small_DepthwiseGridRefusal : public ::testing::TestWithParam<ShapeRow>
    {

    };
}

/** @brief Tests that the input-gradient emitter refuses a gradient on another window grid. */
TEST_P(Small_DepthwiseGridRefusal, BackwardInputRefusesAGradientOnAnotherGrid)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID dy = builder.input(GetParam().shape, "dy");
    const ValueID w = builder.parameter(Shape{3, 3, 3}, "w");
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(builder.depthwise_conv2d_backward_input(dy, w, Shape{2, 5, 5, 3},
                                                              same_3x3()),
                      StatusCode::InvalidDescription);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

INSTANTIATE_TEST_SUITE_P(, Small_DepthwiseGridRefusal, ::testing::ValuesIn(K_GRID_REFUSALS),
                         row_name<ShapeRow>);

/** @brief Tests that the tap-gradient emitter writes its identity over [gradient, input]. */
TEST(Small_DepthwiseOps, BackwardWeightWritesItsIdentityOverGradientAndInputInOrder)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID dy = builder.input(Shape{2, 5, 5, 3}, "dy");
    const ValueID x = builder.input(Shape{2, 5, 5, 3}, "x");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID dw,
                            builder.depthwise_conv2d_backward_weight(dy, x, same_3x3()));

    expect_produced_by(graph, dw, Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT.id, {dy, x});
}

/** @brief Tests that the tap-gradient emitter yields a [kernel_h, kernel_w, C] gradient. */
TEST(Small_DepthwiseOps, BackwardWeightYieldsTheTapShapeInTheOperandsDType)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID dy = builder.input(Shape{2, 5, 5, 3}, "dy", DType::Float64);
    const ValueID x = builder.input(Shape{2, 5, 5, 3}, "x", DType::Float64);

    SA_ASSERT_OK_AND_ASSIGN(const ValueID dw,
                            builder.depthwise_conv2d_backward_weight(dy, x, same_3x3()));

    EXPECT_EQ(graph.value(dw).shape, Shape({3, 3, 3}));
    EXPECT_EQ(graph.value(dw).layout, Graph::TensorLayout::NHWC);
    EXPECT_EQ(graph.value(dw).dtype, DType::Float64);
    EXPECT_EQ(graph.value(dw).lifetime, Graph::Lifetime::Activation);
}

namespace
{
    /** @brief Holds one gradient and input pair the tap-gradient emitter refuses. */
    struct WeightGradientRefusal
    {
        std::string_view name; /**< The label of the case. */
        Shape gradient;        /**< The shape of the gradient. */
        DType gradient_dtype;  /**< The element type of the gradient. */
        Shape input;           /**< The shape of the forward input. */
        DType input_dtype;     /**< The element type of the forward input. */
    };

    /** @brief Lists the pairs that disagree in batch, channels, element type or rank. */
    const std::array<WeightGradientRefusal, 4> K_WEIGHT_GRADIENT_REFUSALS = {{
        {"OtherBatch", Shape{2, 5, 5, 3}, DType::Float32, Shape{1, 5, 5, 3}, DType::Float32},
        {"OtherChannels", Shape{2, 5, 5, 3}, DType::Float32, Shape{2, 5, 5, 4}, DType::Float32},
        {"OtherDType", Shape{2, 5, 5, 3}, DType::Float32, Shape{2, 5, 5, 3}, DType::Float64},
        {"RankThreeGradient", Shape{2, 5, 3}, DType::Float32, Shape{2, 5, 5, 3}, DType::Float32},
    }};

    /** @brief Runs once per pair the tap-gradient emitter refuses. */
    class Small_DepthwiseWeightRefusal : public ::testing::TestWithParam<WeightGradientRefusal>
    {

    };
}

/** @brief Tests that the tap-gradient emitter refuses operands that disagree in batch or depth. */
TEST_P(Small_DepthwiseWeightRefusal, BackwardWeightRefusesMismatchedBatchChannelsOrDType)
{
    const WeightGradientRefusal& pair = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID dy = builder.input(pair.gradient, "dy", pair.gradient_dtype);
    const ValueID x = builder.input(pair.input, "x", pair.input_dtype);
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(builder.depthwise_conv2d_backward_weight(dy, x, same_3x3()),
                      StatusCode::InvalidDescription);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

INSTANTIATE_TEST_SUITE_P(, Small_DepthwiseWeightRefusal,
                         ::testing::ValuesIn(K_WEIGHT_GRADIENT_REFUSALS),
                         row_name<WeightGradientRefusal>);

namespace
{
    /** @brief Names one two-operand emitter and the identity it must write. */
    struct BinaryEmitter
    {
        /** @brief Label for a failure. */
        std::string_view name;

        /** @brief Expected identity. */
        OpID op;

        /** @brief The emitter. */
        Traced (GraphBuilder::*emit)(const Traced&, const Traced&) const;
    };

    /** @brief Names one single-operand emitter and the identity it must write. */
    struct UnaryEmitter
    {
        /** @brief Label for a failure. */
        std::string_view name;

        /** @brief Expected identity. */
        OpID op;

        /** @brief The emitter. */
        Traced (GraphBuilder::*emit)(const Traced&) const;
    };

    /** @brief Lists every two-operand emitter under test. */
    const std::array<BinaryEmitter, 4> K_BINARY_EMITTERS = {{
        {"div",        Ops::K_DIV.id,        &GraphBuilder::div},
        {"minimum",    Ops::K_MINIMUM.id,    &GraphBuilder::minimum},
        {"maximum",    Ops::K_MAXIMUM.id,    &GraphBuilder::maximum},
        {"less_equal", Ops::K_LESS_EQUAL.id, &GraphBuilder::less_equal},
    }};

    /** @brief Lists every single-operand emitter under test. */
    const std::array<UnaryEmitter, 4> K_UNARY_EMITTERS = {{
        {"abs",      Ops::K_ABS.id,      &GraphBuilder::abs},
        {"atan",     Ops::K_ATAN.id,     &GraphBuilder::atan},
        {"softplus", Ops::K_SOFTPLUS.id, &GraphBuilder::softplus},
        {"silu",     Ops::K_SILU.id,     &GraphBuilder::silu},
    }};

    /** @brief Lists every activation gradient that reads the forward input. */
    const std::array<OpID, 2> K_INPUT_READING_GRADIENTS = {{
        Ops::K_SILU_BACKWARD.id,
        Ops::K_SOFTPLUS_BACKWARD.id,
    }};

    /** @brief Declares a rank-4 input and converts it to NCHW; the extents survive the permute. */
    Traced small_nchw_input(const GraphBuilder& builder, std::string_view debug_name,
                            DType dtype = DType::Float32)
    {
        const ValueID nhwc = builder.input(Shape{2, 2, 2, 2}, debug_name, dtype);
        return builder.layout_convert(nhwc, Graph::TensorLayout::NCHW);
    }

    /** @brief Expects @p out to copy the shape, dtype and layout of @p like as an activation. */
    void expect_described_like(const Graph::Graph& graph, ValueID out, ValueID like)
    {
        EXPECT_EQ(graph.value(out).shape, graph.value(like).shape);
        EXPECT_EQ(graph.value(out).dtype, graph.value(like).dtype);
        EXPECT_EQ(graph.value(out).layout, graph.value(like).layout);
        EXPECT_EQ(graph.value(out).lifetime, Graph::Lifetime::Activation);
    }
}

namespace
{
    /** @brief Runs once per two-operand emitter. */
    class Small_BinaryEmitter : public ::testing::TestWithParam<BinaryEmitter>
    {

    };

    /** @brief Runs once per single-operand emitter. */
    class Small_UnaryEmitter : public ::testing::TestWithParam<UnaryEmitter>
    {

    };

    /** @brief Runs once per activation gradient that reads the forward input. */
    class Small_InputReadingGradient : public ::testing::TestWithParam<OpID>
    {

    };

    /** @brief Runs once per element type LESS_EQUAL yields its mask in. */
    class Small_LessEqualMask : public ::testing::TestWithParam<DTypeRow>
    {

    };

    /** @brief Lists the element types LESS_EQUAL is called with. */
    const std::array<DTypeRow, 2> K_MASK_DTYPES = {{
        {"Float32", DType::Float32},
        {"Float64", DType::Float64},
    }};

    /**
     * @brief Chains @p a through every two-operand emitter against @p b and every single-operand
     *        emitter, and ends in a select on a LESS_EQUAL mask.
     */
    Traced chain_through_every_emitter(const GraphBuilder& builder, ValueID a, ValueID b)
    {
        Traced chained = a;
        for (const BinaryEmitter& emitter : K_BINARY_EMITTERS)
            chained = (builder.*emitter.emit)(chained, b);
        for (const UnaryEmitter& emitter : K_UNARY_EMITTERS)
            chained = (builder.*emitter.emit)(chained);

        return builder.select(builder.less_equal(a, b), chained, b);
    }
}

/** @brief Tests that each two-operand emitter writes its own identity over [a, b] in that order. */
TEST_P(Small_BinaryEmitter, BinaryEmittersWriteTheirIdentityOverBothOperandsInOrder)
{
    const BinaryEmitter& emitter = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID b = builder.input(Shape{4, 3}, "b");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, (builder.*emitter.emit)(a, b));

    expect_produced_by(graph, out, emitter.op, {a, b});
}

/** @brief Tests that each two-operand emitter gives its result the first operand's description. */
TEST_P(Small_BinaryEmitter, BinaryEmittersKeepShapeDTypeAndLayout)
{
    const BinaryEmitter& emitter = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID a, small_nchw_input(builder, "a", DType::Float64));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID b, small_nchw_input(builder, "b", DType::Float64));

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, (builder.*emitter.emit)(a, b));

    expect_described_like(graph, out, a);
    EXPECT_EQ(graph.value(out).layout, Graph::TensorLayout::NCHW);
    EXPECT_EQ(graph.value(out).dtype, DType::Float64);
}

/** @brief Tests that each two-operand emitter refuses a broadcastable pair of unequal shapes. */
TEST_P(Small_BinaryEmitter, BinaryEmittersRejectUnequalShapesWithoutBroadcasting)
{
    const BinaryEmitter& emitter = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID matrix = builder.input(Shape{4, 3}, "matrix");
    const ValueID column = builder.input(Shape{4, 1}, "column");
    const ValueID row = builder.input(Shape{3}, "row");
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE((builder.*emitter.emit)(matrix, column), StatusCode::InvalidDescription);
    SA_EXPECT_FAILURE((builder.*emitter.emit)(column, matrix), StatusCode::InvalidDescription);
    SA_EXPECT_FAILURE((builder.*emitter.emit)(matrix, row), StatusCode::InvalidDescription);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

/** @brief Tests that each two-operand emitter refuses operands of two element types. */
TEST_P(Small_BinaryEmitter, BinaryEmittersRejectUnequalDtypes)
{
    const BinaryEmitter& emitter = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID single = builder.input(Shape{4, 3}, "single", DType::Float32);
    const ValueID wide = builder.input(Shape{4, 3}, "wide", DType::Float64);

    SA_EXPECT_FAILURE((builder.*emitter.emit)(single, wide), StatusCode::InvalidDescription);
    SA_EXPECT_FAILURE((builder.*emitter.emit)(wide, single), StatusCode::InvalidDescription);
}

/** @brief Tests that Graph::finalise refuses an emitted node whose operands sit in two layouts. */
TEST_P(Small_BinaryEmitter, FinaliseRejectsOperandsInTwoLayouts)
{
    const BinaryEmitter& emitter = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID nchw, small_nchw_input(builder, "nchw"));
    const ValueID nhwc = builder.input(Shape{2, 2, 2, 2}, "nhwc");

    (void)(builder.*emitter.emit)(nchw, nhwc);

    SA_EXPECT_FAILURE(graph.finalise(), StatusCode::InvalidDescription,
                      "reads operands in two different layouts");
}

INSTANTIATE_TEST_SUITE_P(, Small_BinaryEmitter, ::testing::ValuesIn(K_BINARY_EMITTERS),
                         row_name<BinaryEmitter>);

/** @brief Tests that LESS_EQUAL yields its mask in the operands' element type, not a mask type. */
TEST_P(Small_LessEqualMask, LessEqualYieldsAMaskInTheOperandsOwnDType)
{
    const DType dtype = GetParam().dtype;

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID a = builder.input(Shape{4, 3}, "a", dtype);
    const ValueID b = builder.input(Shape{4, 3}, "b", dtype);

    SA_ASSERT_OK_AND_ASSIGN(const ValueID mask, builder.less_equal(a, b));

    EXPECT_EQ(graph.value(mask).dtype, dtype);
}

INSTANTIATE_TEST_SUITE_P(, Small_LessEqualMask, ::testing::ValuesIn(K_MASK_DTYPES),
                         row_name<DTypeRow>);

/** @brief Tests that each single-operand emitter writes its own identity over its operand. */
TEST_P(Small_UnaryEmitter, UnaryEmittersWriteTheirIdentityOverTheirOperand)
{
    const UnaryEmitter& emitter = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 3}, "x");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, (builder.*emitter.emit)(x));

    EXPECT_NE(out, x) << "the emitter must write a new value, not its operand";
    expect_produced_by(graph, out, emitter.op, {x});
}

/** @brief Tests that each single-operand emitter gives its result the operand's description. */
TEST_P(Small_UnaryEmitter, UnaryEmittersKeepShapeDTypeAndLayout)
{
    const UnaryEmitter& emitter = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, small_nchw_input(builder, "x", DType::Float64));

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, (builder.*emitter.emit)(x));

    expect_described_like(graph, out, x);
    EXPECT_EQ(graph.value(out).layout, Graph::TensorLayout::NCHW);
    EXPECT_EQ(graph.value(out).dtype, DType::Float64);
}

INSTANTIATE_TEST_SUITE_P(, Small_UnaryEmitter, ::testing::ValuesIn(K_UNARY_EMITTERS),
                         row_name<UnaryEmitter>);

/** @brief Tests that SELECT writes one node reading [condition, a, b] in that order. */
TEST(Small_ElementwiseOps, SelectReadsItsConditionThenBothBranches)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID condition = builder.input(Shape{4, 3}, "condition");
    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID b = builder.input(Shape{4, 3}, "b");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, builder.select(condition, a, b));

    expect_produced_by(graph, out, Ops::K_SELECT.id, {condition, a, b});
}

/** @brief Tests that SELECT gives its result the shape, dtype and layout of its branches. */
TEST(Small_ElementwiseOps, SelectKeepsShapeDTypeAndLayout)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID condition,
                            small_nchw_input(builder, "condition", DType::Float64));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID a, small_nchw_input(builder, "a", DType::Float64));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID b, small_nchw_input(builder, "b", DType::Float64));

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, builder.select(condition, a, b));

    expect_described_like(graph, out, a);
    EXPECT_EQ(graph.value(out).layout, Graph::TensorLayout::NCHW);
}

/** @brief Tests that SELECT refuses a condition whose shape differs from its branches. */
TEST(Small_ElementwiseOps, SelectRejectsAConditionOfAnotherShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID per_row = builder.input(Shape{4, 1}, "per_row");
    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID b = builder.input(Shape{4, 3}, "b");
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(builder.select(per_row, a, b), StatusCode::InvalidDescription);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
}

/** @brief Tests that SELECT refuses a condition whose element type differs from its branches. */
TEST(Small_ElementwiseOps, SelectRejectsAConditionOfAnotherDType)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID condition = builder.input(Shape{4, 3}, "condition", DType::Int64);
    const ValueID a = builder.input(Shape{4, 3}, "a", DType::Float32);
    const ValueID b = builder.input(Shape{4, 3}, "b", DType::Float32);

    SA_EXPECT_FAILURE(builder.select(condition, a, b), StatusCode::InvalidDescription);
}

/** @brief Tests that SELECT refuses two branches of unequal shape. */
TEST(Small_ElementwiseOps, SelectRejectsBranchesOfUnequalShape)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID condition = builder.input(Shape{4, 3}, "condition");
    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID narrow = builder.input(Shape{4, 1}, "narrow");

    SA_EXPECT_FAILURE(builder.select(condition, a, narrow), StatusCode::InvalidDescription);
    SA_EXPECT_FAILURE(builder.select(condition, narrow, a), StatusCode::InvalidDescription);
}

/** @brief Tests that SELECT refuses two branches of unequal element type. */
TEST(Small_ElementwiseOps, SelectRejectsBranchesOfUnequalDType)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID condition = builder.input(Shape{4, 3}, "condition", DType::Float32);
    const ValueID a = builder.input(Shape{4, 3}, "a", DType::Float32);
    const ValueID wide = builder.input(Shape{4, 3}, "wide", DType::Float64);

    SA_EXPECT_FAILURE(builder.select(condition, a, wide), StatusCode::InvalidDescription);
    SA_EXPECT_FAILURE(builder.select(condition, wide, a), StatusCode::InvalidDescription);
}

/** @brief Tests that the activation-gradient emitter writes each input-reading gradient. */
TEST_P(Small_InputReadingGradient, ActivationBackwardEmitsTheInputReadingGradients)
{
    const OpID gradient = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID dy = builder.input(Shape{4, 3}, "dy");
    const ValueID x = builder.input(Shape{4, 3}, "x");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID dx, builder.activation_backward(gradient, dy, x));

    expect_produced_by(graph, dx, gradient, {dy, x});
    expect_described_like(graph, dx, dy);
}

/** @brief Tests that each input-reading gradient refuses a dy shaped unlike the input. */
TEST_P(Small_InputReadingGradient, InputReadingGradientsRejectAGradientOfAnotherShape)
{
    const OpID gradient = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID dy = builder.input(Shape{4, 1}, "dy");
    const ValueID x = builder.input(Shape{4, 3}, "x");

    SA_EXPECT_FAILURE(builder.activation_backward(gradient, dy, x),
                      StatusCode::InvalidDescription);
}

/** @brief Tests that each input-reading gradient refuses a dy of another element type. */
TEST_P(Small_InputReadingGradient, InputReadingGradientsRejectAGradientOfAnotherDType)
{
    const OpID gradient = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID single = builder.input(Shape{4, 3}, "single", DType::Float32);
    const ValueID wide = builder.input(Shape{4, 3}, "wide", DType::Float64);

    SA_EXPECT_FAILURE(builder.activation_backward(gradient, single, wide),
                      StatusCode::InvalidDescription);
    SA_EXPECT_FAILURE(builder.activation_backward(gradient, wide, single),
                      StatusCode::InvalidDescription);
}

INSTANTIATE_TEST_SUITE_P(, Small_InputReadingGradient,
                         ::testing::ValuesIn(K_INPUT_READING_GRADIENTS), operation_name);

namespace
{
    /** @brief Lists the identities that are not activation gradients. */
    const std::array<OpID, 3> K_NON_GRADIENTS = {{
        Ops::K_SILU.id,
        Ops::K_SOFTPLUS.id,
        Ops::K_DIV.id,
    }};

    /** @brief Runs once per identity the activation-gradient emitter stops the process for. */
    class Medium_ActivationBackwardDeathTest : public ::testing::TestWithParam<OpID>
    {

    };
}

/** @brief Tests that the activation-gradient emitter refuses an identity outside ACTIVATIONS. */
TEST_P(Medium_ActivationBackwardDeathTest,
       ActivationBackwardRejectsAnIdentityThatIsNotAnActivationGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID dy = builder.input(Shape{4, 3}, "dy");
    const ValueID x = builder.input(Shape{4, 3}, "x");

    SA_EXPECT_FATAL((void)builder.activation_backward(GetParam(), dy, x),
                    "is not an activation gradient");
}

INSTANTIATE_TEST_SUITE_P(, Medium_ActivationBackwardDeathTest,
                         ::testing::ValuesIn(K_NON_GRADIENTS), operation_name);

/** @brief Tests that Graph::finalise accepts a graph that uses every emitter in one layout. */
TEST(Small_ElementwiseOps, FinaliseAcceptsAGraphUsingEveryEmitter)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID a, small_nchw_input(builder, "a"));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID b, small_nchw_input(builder, "b"));

    SA_EXPECT_OK(chain_through_every_emitter(builder, a, b));

    SA_EXPECT_OK(graph.finalise());
}

/** @brief Tests that Graph::finalise refuses a SELECT whose condition sits in another layout. */
TEST(Small_ElementwiseOps, FinaliseRejectsASelectConditionInAnotherLayout)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID condition = builder.input(Shape{2, 2, 2, 2}, "condition");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID a, small_nchw_input(builder, "a"));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID b, small_nchw_input(builder, "b"));

    SA_EXPECT_OK(builder.select(condition, a, b));

    SA_EXPECT_FAILURE(graph.finalise(), StatusCode::InvalidDescription,
                      "reads operands in two different layouts");
}

/** @brief Tests that a result computed from a parameter requires a gradient after finalise. */
TEST(Small_ElementwiseOps, ResultsOfAParameterRequireAGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID weight = builder.parameter(Shape{4, 3}, "weight");
    const ValueID constant = builder.input(Shape{4, 3}, "constant");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID from_parameter,
                            builder.silu(builder.div(weight, constant)));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID from_inputs,
                            builder.atan(builder.maximum(constant, constant)));
    SA_EXPECT_OK(graph.finalise());

    EXPECT_TRUE(graph.value(from_parameter).requires_grad);
    EXPECT_FALSE(graph.value(from_inputs).requires_grad);
}

/** @brief Tests that a comparison of a parameter yields a mask that requires no gradient. */
TEST(Small_ElementwiseOps, AComparisonOfAParameterRequiresNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID weight = builder.parameter(Shape{4, 3}, "weight");
    const ValueID bound = builder.input(Shape{4, 3}, "bound");

    SA_ASSERT_OK_AND_ASSIGN(const ValueID mask, builder.less_equal(weight, bound));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID selected, builder.select(mask, bound, bound));
    SA_EXPECT_OK(graph.finalise());

    EXPECT_FALSE(graph.value(mask).requires_grad);
    EXPECT_FALSE(graph.value(selected).requires_grad);
}

/** @brief Tests that constant() writes one operand-free CONSTANT node holding the value. */
TEST(Small_ElementwiseOps, ConstantWritesAnOperandFreeNodeHoldingItsValue)
{
    constexpr double K_VALUE = 1e-7;

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out,
                            builder.constant(Shape{4, 3}, K_VALUE, DType::Float64));

    expect_produced_by(graph, out, Ops::K_CONSTANT.id, {});
    EXPECT_EQ(graph.node(graph.value(out).producer).attributes.get(0), K_VALUE);
    EXPECT_EQ(graph.value(out).shape, (Shape{4, 3}));
    EXPECT_EQ(graph.value(out).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out).layout, Graph::TensorLayout::NHWC);
    EXPECT_EQ(graph.value(out).lifetime, Graph::Lifetime::Activation);
}

/** @brief Tests that constant() refuses an element type that is not real floating. */
TEST(Small_ElementwiseOps, ConstantRejectsAnElementTypeThatIsNotRealFloating)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const std::size_t nodes_before = graph.node_count();
    const std::size_t values_before = graph.value_count();

    SA_EXPECT_FAILURE(builder.constant(Shape{2}, 1.0, DType::Int64),
                      StatusCode::InvalidDescription);
    SA_EXPECT_FAILURE(builder.constant(Shape{2}, 1.0, DType::Complex32),
                      StatusCode::InvalidDescription);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
    EXPECT_EQ(graph.value_count(), values_before) << "a rejected call left a value behind";
}

/** @brief Tests that constant_like() copies the shape, dtype and layout of the value it names. */
TEST(Small_ElementwiseOps, ConstantLikeCopiesShapeDTypeAndLayoutWithoutReadingItsOperand)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, small_nchw_input(builder, "x", DType::Float64));

    SA_ASSERT_OK_AND_ASSIGN(const ValueID out, builder.constant_like(x, 0.5));

    expect_produced_by(graph, out, Ops::K_CONSTANT.id, {});
    expect_described_like(graph, out, x);
    EXPECT_EQ(graph.value(out).layout, Graph::TensorLayout::NCHW);
    EXPECT_EQ(graph.value(out).dtype, DType::Float64);
}

/** @brief Tests that constant_like() refuses to describe a value that is not real floating. */
TEST(Small_ElementwiseOps, ConstantLikeRejectsAValueThatIsNotRealFloating)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 3}, "x", DType::Int64);
    const std::size_t nodes_before = graph.node_count();
    const std::size_t values_before = graph.value_count();

    SA_EXPECT_FAILURE(builder.constant_like(x, 1.0), StatusCode::InvalidDescription);
    EXPECT_EQ(graph.node_count(), nodes_before) << "a rejected call left a node behind";
    EXPECT_EQ(graph.value_count(), values_before) << "a rejected call left a value behind";
}

/** @brief Tests that a CONSTANT joins a binary operation with the value it was shaped like. */
TEST(Small_ElementwiseOps, AConstantLikeAValueIsAnOperandThatFinaliseAccepts)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID x, small_nchw_input(builder, "x"));

    SA_ASSERT_OK_AND_ASSIGN(const ValueID shifted, builder.add(x, builder.constant_like(x, 1e-7)));

    SA_EXPECT_OK(graph.finalise());
    EXPECT_FALSE(graph.value(shifted).requires_grad);
}

/** @brief Tests that zeros() and zeros_like() still write ZERO and never CONSTANT. */
TEST(Small_ElementwiseOps, ZerosStillWritesTheZeroIdentity)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 3}, "x");

    expect_produced_by(graph, builder.zeros(Shape{4, 3}), Ops::K_ZERO.id, {});
    SA_ASSERT_OK_AND_ASSIGN(const ValueID zeroed, builder.zeros_like(x, Shape{4, 3}));
    expect_produced_by(graph, zeroed, Ops::K_ZERO.id, {});
}

namespace
{
    /** @brief Holds the row count of the table every case reads. */
    constexpr int64_t K_TABLE_ROWS = 7;

    /** @brief Holds the row length of the table every case reads. */
    constexpr int64_t K_ROW_LENGTH = 3;

}

/** @brief Tests that a gather over a rank-1 index writes GATHER_ROWS over [table, indices]. */
TEST(Small_GatherOps, GatherRowsWritesItsIdentityOverTheTableAndTheIndex)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);

    SA_ASSERT_OK_AND_ASSIGN(const Traced out, builder.gather_rows(table, indices));

    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());
    expect_produced_by(graph, out.value(), Ops::K_GATHER_ROWS.id, {table, indices});
    EXPECT_EQ(graph.value(out.value()).shape, (Shape{5, K_ROW_LENGTH}));
    EXPECT_EQ(graph.value(out.value()).dtype, DType::Float32);
    EXPECT_EQ(graph.value(out.value()).lifetime, Graph::Lifetime::Activation);
    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that the result keeps every extent of the index and the table's element type. */
TEST(Small_GatherOps, GatherRowsAppendsTheRowLengthToAnIndexOfAnyRank)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table =
        builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table", DType::Float64);
    const ValueID indices = builder.input(Shape{2, 4}, "indices", DType::Int32);
    const ValueID deep = builder.input(Shape{2, 1, 3, 1, 2}, "deep", DType::Int32);

    SA_ASSERT_OK_AND_ASSIGN(const Traced out, builder.gather_rows(table, indices));
    SA_ASSERT_OK_AND_ASSIGN(const Traced deepest, builder.gather_rows(table, deep));

    ASSERT_TRUE(out.has_value());
    ASSERT_TRUE(deepest.has_value());
    EXPECT_EQ(graph.value(out.value()).shape, (Shape{2, 4, K_ROW_LENGTH}));
    EXPECT_EQ(graph.value(out.value()).dtype, DType::Float64);
    EXPECT_EQ(graph.value(deepest.value()).shape, (Shape{2, 1, 3, 1, 2, K_ROW_LENGTH}));
}

/** @brief Tests that a table of rank 1 or 3 is refused and no node is added. */
TEST(Small_GatherOps, GatherRowsRefusesATableThatIsNotRankTwo)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID flat = builder.parameter(Shape{K_TABLE_ROWS}, "flat");
    const ValueID cube = builder.parameter(Shape{K_TABLE_ROWS, 2, K_ROW_LENGTH}, "cube");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.gather_rows(flat, indices), nodes, "rank-2 [V, D] table");
    expect_refused(graph, builder.gather_rows(cube, indices), nodes, "[7, 2, 3]");
}

/** @brief Tests that a FLOAT32 index and an INT64 index are both refused. */
TEST(Small_GatherOps, GatherRowsRefusesAnIndexThatIsNotInt32)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID real = builder.input(Shape{5}, "real", DType::Float32);
    const ValueID wide = builder.input(Shape{5}, "wide", DType::Int64);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.gather_rows(table, real), nodes, "INT32 index");
    expect_refused(graph, builder.gather_rows(table, wide), nodes, "INT32 index");
}

/** @brief Tests that an integer table is refused. */
TEST(Small_GatherOps, GatherRowsRefusesATableThatIsNotFloating)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.input(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table", DType::Int32);
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.gather_rows(table, indices), nodes,
                   "half, float32 or float64 table");
}

/** @brief Tests that an index of the highest rank is refused, as its result has no rank left. */
TEST(Small_GatherOps, GatherRowsRefusesAnIndexWhoseResultWouldPassTheHighestRank)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID indices = builder.input(Shape{1, 1, 1, 1, 1, 1}, "indices", DType::Int32);
    const std::size_t nodes = graph.node_count();

    ASSERT_EQ(graph.value(indices).shape.rank(), K_MAX_RANK);
    expect_refused(graph, builder.gather_rows(table, indices), nodes, "rank 1 to 5");
}

/** @brief Tests that a scatter-add writes SCATTER_ADD_ROWS over [rows, indices]. */
TEST(Small_GatherOps, ScatterAddRowsWritesItsIdentity)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.input(Shape{2, 4, K_ROW_LENGTH}, "rows", DType::Float64);
    const ValueID indices = builder.input(Shape{2, 4}, "indices", DType::Int32);

    SA_ASSERT_OK_AND_ASSIGN(const Traced out,
                            builder.scatter_add_rows(rows, indices, K_TABLE_ROWS));

    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());
    expect_produced_by(graph, out.value(), Ops::K_SCATTER_ADD_ROWS.id, {rows, indices});
}

/** @brief Tests that a scatter-add yields a [V, D] table in the rows' type, as an activation. */
TEST(Small_GatherOps, ScatterAddRowsYieldsTheTableShapeDTypeAndLifetime)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.input(Shape{2, 4, K_ROW_LENGTH}, "rows", DType::Float64);
    const ValueID indices = builder.input(Shape{2, 4}, "indices", DType::Int32);

    SA_ASSERT_OK_AND_ASSIGN(const Traced out,
                            builder.scatter_add_rows(rows, indices, K_TABLE_ROWS));

    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(graph.value(out.value()).shape, (Shape{K_TABLE_ROWS, K_ROW_LENGTH}));
    EXPECT_EQ(graph.value(out.value()).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out.value()).lifetime, Graph::Lifetime::Activation);
}

namespace
{
    /** @brief Holds one operand set a scatter-add refuses and the part its message names. */
    struct ScatterRefusal
    {
        std::string_view name; /**< The label of the case. */
        Shape rows;            /**< The shape of the rows. */
        DType rows_dtype;      /**< The element type of the rows. */
        Shape indices;         /**< The shape of the index. */
        DType indices_dtype;   /**< The element type of the index. */
        int64_t table_rows;    /**< The row count of the table. */
        const char* part;      /**< The part the refusal names. */
    };

    /** @brief Lists the operand sets a scatter-add refuses. */
    const std::array<ScatterRefusal, 5> K_SCATTER_REFUSALS = {{
        {"RankOneRows", Shape{5}, DType::Float32, Shape{5}, DType::Int32, K_TABLE_ROWS,
         "rank 2 or more"},
        {"IntegerRows", Shape{5, K_ROW_LENGTH}, DType::Int32, Shape{5}, DType::Int32,
         K_TABLE_ROWS, "half, float32 or float64 rows"},
        {"RealIndex", Shape{5, K_ROW_LENGTH}, DType::Float32, Shape{5}, DType::Float32,
         K_TABLE_ROWS, "INT32 index"},
        {"ShortIndex", Shape{5, K_ROW_LENGTH}, DType::Float32, Shape{4}, DType::Int32,
         K_TABLE_ROWS, "one index per row"},
        {"EmptyTable", Shape{5, K_ROW_LENGTH}, DType::Float32, Shape{5}, DType::Int32, 0,
         "a table of 0 rows"},
    }};

    /** @brief Runs once per operand set a scatter-add refuses. */
    class Small_ScatterAddRefusal : public ::testing::TestWithParam<ScatterRefusal>
    {

    };
}

/** @brief Tests each refusal of the scatter-add: its rows, its index and its row count. */
TEST_P(Small_ScatterAddRefusal, ScatterAddRowsRefusesOperandsThatDoNotFit)
{
    const ScatterRefusal& refusal = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows = builder.input(refusal.rows, "rows", refusal.rows_dtype);
    const ValueID indices = builder.input(refusal.indices, "indices", refusal.indices_dtype);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.scatter_add_rows(rows, indices, refusal.table_rows), nodes,
                   refusal.part);
}

INSTANTIATE_TEST_SUITE_P(, Small_ScatterAddRefusal, ::testing::ValuesIn(K_SCATTER_REFUSALS),
                         row_name<ScatterRefusal>);

/** @brief Tests that finalise leaves an INT32 input without a gradient and marks the gather. */
TEST(Small_GatherOps, AnInt32InputLeafRequiresNoGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
    const ValueID indices = builder.input(Shape{5}, "indices", DType::Int32);
    SA_ASSERT_OK_AND_ASSIGN(const Traced out, builder.gather_rows(table, indices));
    ASSERT_TRUE(out.has_value());

    SA_EXPECT_OK(graph.finalise());

    EXPECT_EQ(graph.value(indices).lifetime, Graph::Lifetime::Input);
    EXPECT_FALSE(graph.value(indices).requires_grad);
    EXPECT_TRUE(graph.value(out.value()).requires_grad);
}

namespace
{
    /**
     * @brief Builds a gather feeding a matmul and runs the cast pass under mixed precision.
     *
     * The leaves are an INT32 index, a float32 table and a float32 weight.
     */
    class Small_GatherCastPass : public ::testing::Test
    {
        public:
            /** @brief Builds the graph and runs the pass. */
            void SetUp() override
            {
                table = builder.parameter(Shape{K_TABLE_ROWS, K_ROW_LENGTH}, "table");
                indices = builder.input(Shape{5}, "indices", DType::Int32);
                const ValueID weight = builder.parameter(Shape{K_ROW_LENGTH, 2}, "weight");
                SA_ASSERT_OK_AND_ASSIGN(const Traced gather_out,
                                        builder.gather_rows(table, indices));
                ASSERT_TRUE(gather_out.has_value());
                gathered = gather_out.value();
                SA_ASSERT_OK_AND_ASSIGN(const ValueID product_out,
                                        builder.matmul(gathered, weight));
                product = product_out;

                Graph::CastOptions options;
                options.policy = &policy;
                SA_ASSERT_OK_AND_ASSIGN(const Graph::CastReport inserted,
                                        Graph::insert_casts(graph, options));
                report = inserted;
            }

            Graph::Graph graph;                 /**< The graph under test. */
            GraphBuilder builder{graph};        /**< The builder over the graph. */
            Graph::MixedPrecisionPolicy policy; /**< The policy the pass runs under. */
            Graph::CastReport report;           /**< What the pass inserted. */
            ValueID table{};                    /**< The float32 table. */
            ValueID indices{};                  /**< The INT32 index. */
            ValueID gathered{};                 /**< The gather's result. */
            ValueID product{};                  /**< The matmul that reads the gather. */
    };

    /** @brief Holds one value of the fixture and the element type it has after the pass. */
    struct DeclaredType
    {
        std::string_view name;                 /**< The label of the case. */
        ValueID Small_GatherCastPass::* value; /**< The value under test. */
        DType expected;                        /**< The element type the pass leaves it in. */
    };

    /** @brief Lists the values the pass leaves as declared and the one it narrows. */
    const std::array<DeclaredType, 4> K_DECLARED_TYPES = {{
        {"Index", &Small_GatherCastPass::indices, DType::Int32},
        {"Table", &Small_GatherCastPass::table, DType::Float32},
        {"Gathered", &Small_GatherCastPass::gathered, DType::Float32},
        {"Product", &Small_GatherCastPass::product, DType::Half},
    }};

    /** @brief Runs once per value the cast pass is checked on. */
    class Small_GatherCastPassType : public Small_GatherCastPass,
                                     public ::testing::WithParamInterface<DeclaredType>
    {

    };
}

/** @brief Tests that the cast pass narrows the product and retypes no other value of the gather. */
TEST_P(Small_GatherCastPassType, EachValueEndsInItsDeclaredTypeOrTheNarrowedOne)
{
    const DeclaredType& row = GetParam();

    EXPECT_EQ(graph.value(this->*row.value).dtype, row.expected);
}

INSTANTIATE_TEST_SUITE_P(, Small_GatherCastPassType, ::testing::ValuesIn(K_DECLARED_TYPES),
                         row_name<DeclaredType>);

/** @brief Tests that the gather still reads the table and the index after the cast pass. */
TEST_F(Small_GatherCastPass, TheGatherStillReadsTheTableAndTheIndex)
{
    const Graph::Node& gather = graph.node(graph.value(gathered).producer);

    EXPECT_EQ(gather.op, Ops::K_GATHER_ROWS.id);
    EXPECT_EQ(gather.inputs, (std::vector<ValueID>{table, indices}));
}

/** @brief Tests that the cast pass does not cast the index. */
TEST_F(Small_GatherCastPass, NoCastReadsTheIndex)
{
    const auto casts = report.casts();

    EXPECT_FALSE(std::any_of(casts.begin(), casts.end(), [this](const Graph::InsertedCast& cast)
                             { return cast.source == indices; }));
}

/** @brief Tests that the cast pass adds no cast at the gather. */
TEST_F(Small_GatherCastPass, NoCastIsAddedAtTheGather)
{
    const auto casts = report.casts();

    EXPECT_FALSE(std::any_of(casts.begin(), casts.end(), [](const Graph::InsertedCast& cast)
                             { return cast.op == Ops::K_GATHER_ROWS.id; }));
}

/** @brief Tests that no value disagrees with the policy after the cast pass. */
TEST_F(Small_GatherCastPass, NoValueDisagreesWithThePolicyAfterTheCastPass)
{
    EXPECT_TRUE(Graph::find_dtype_mismatches(graph, policy).empty());
}

namespace
{
    /** @brief Holds the last extent every case normalises over. */
    constexpr int64_t K_WIDTH = 4;

    /** @brief Holds the epsilon every case passes. */
    constexpr double K_EPSILON = 1e-5;

    /** @brief Holds the leaves of one graph: a [2, 3, K_WIDTH] input, a gain and a bias. */
    struct Leaves
    {
        ValueID x;     /**< The [2, 3, K_WIDTH] input. */
        ValueID gamma; /**< The [K_WIDTH] gain. */
        ValueID beta;  /**< The [K_WIDTH] bias. */
    };

    /** @brief Declares the three leaves in @p dtype. */
    Leaves declare_leaves(const GraphBuilder& builder, DType dtype = DType::Float32)
    {
        return Leaves{builder.input(Shape{2, 3, K_WIDTH}, "x", dtype),
                      builder.parameter(Shape{K_WIDTH}, "gamma", dtype),
                      builder.parameter(Shape{K_WIDTH}, "beta", dtype)};
    }

    /** @brief Expects @p refused to name @p part, and the graph to hold no node beyond @p nodes. */
    template<typename Outcome>
    void expect_refused(const Graph::Graph& graph, const Outcome& refused, std::size_t nodes,
                        const char* part)
    {
        SA_EXPECT_FAILURE(refused, StatusCode::InvalidDescription, part);
        EXPECT_EQ(graph.node_count(), nodes) << "a refused call left a node behind";
        EXPECT_TRUE(graph.has_refusal());
    }
}

/** @brief Tests that the node reads [x, gamma, beta] and writes the value and two [rows] saves. */
TEST(Small_LayerNormOps, LayerNormWritesItsValueAndTheTwoRowStatistics)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Leaves leaves = declare_leaves(builder, DType::Float64);

    const Result<LayerNormResult> out =
        builder.layer_norm(leaves.x, leaves.gamma, leaves.beta, K_EPSILON);

    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());
    const Graph::Node& node = graph.node(graph.value(out.value().output).producer);
    EXPECT_EQ(node.op, Ops::K_LAYER_NORM.id);
    EXPECT_EQ(node.inputs, (std::vector<ValueID>{leaves.x, leaves.gamma, leaves.beta}));
    EXPECT_EQ(node.outputs, (std::vector<ValueID>{out.value().output, out.value().saved_mean,
                                                  out.value().saved_inv_std}));
    EXPECT_EQ(node.attributes.values[0], K_EPSILON);

    EXPECT_EQ(graph.value(out.value().output).shape, (Shape{2, 3, K_WIDTH}));
    EXPECT_EQ(graph.value(out.value().saved_mean).shape, (Shape{6}));
    EXPECT_EQ(graph.value(out.value().saved_inv_std).shape, (Shape{6}));
    EXPECT_EQ(graph.value(out.value().output).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out.value().saved_mean).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out.value().output).lifetime, Graph::Lifetime::Activation);
    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that a rank-1 input is one row, so both saved statistics hold one element. */
TEST(Small_LayerNormOps, ARankOneInputIsOneRow)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{K_WIDTH}, "x");
    const ValueID gamma = builder.parameter(Shape{K_WIDTH}, "gamma");
    const ValueID beta = builder.parameter(Shape{K_WIDTH}, "beta");

    const Result<LayerNormResult> out = builder.layer_norm(x, gamma, beta, K_EPSILON);

    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(graph.value(out.value().output).shape, (Shape{K_WIDTH}));
    EXPECT_EQ(graph.value(out.value().saved_mean).shape, (Shape{1}));
}

namespace
{
    /** @brief Holds one operand set a layer norm refuses and the part its message names. */
    struct LayerNormRefusal
    {
        std::string_view name; /**< The label of the case. */
        Shape x;               /**< The shape of the input. */
        DType x_dtype;         /**< The element type of the input. */
        Shape gamma;           /**< The shape of the gain. */
        DType gamma_dtype;     /**< The element type of the gain. */
        Shape beta;            /**< The shape of the bias. */
        DType beta_dtype;      /**< The element type of the bias. */
        double epsilon;        /**< The epsilon passed. */
        const char* part;      /**< The part the refusal names. */
    };

    /** @brief Lists the operand sets a layer norm refuses. */
    const std::array<LayerNormRefusal, 4> K_LAYER_NORM_REFUSALS = {{
        {"IntegerInput", Shape{2, K_WIDTH}, DType::Int32, Shape{K_WIDTH}, DType::Float32,
         Shape{K_WIDTH}, DType::Float32, K_EPSILON, "half, float32 or float64 input"},
        {"NarrowGain", Shape{2, 3, K_WIDTH}, DType::Float32, Shape{K_WIDTH - 1}, DType::Float32,
         Shape{K_WIDTH}, DType::Float32, K_EPSILON, "gamma as [4]"},
        {"WideBias", Shape{2, 3, K_WIDTH}, DType::Float32, Shape{K_WIDTH}, DType::Float32,
         Shape{K_WIDTH}, DType::Float64, K_EPSILON, "beta as [4] of float32"},
        {"ZeroEpsilon", Shape{2, 3, K_WIDTH}, DType::Float32, Shape{K_WIDTH}, DType::Float32,
         Shape{K_WIDTH}, DType::Float32, 0.0, "epsilon must be positive"},
    }};

    /** @brief Runs once per operand set the forward layer norm refuses. */
    class Small_LayerNormRefusal : public ::testing::TestWithParam<LayerNormRefusal>
    {

    };
}

/** @brief Tests each refusal of the forward emitter: the input, the parameters, the epsilon. */
TEST_P(Small_LayerNormRefusal, LayerNormRefusesOperandsThatDoNotFit)
{
    const LayerNormRefusal& refusal = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(refusal.x, "x", refusal.x_dtype);
    const ValueID gamma = builder.parameter(refusal.gamma, "gamma", refusal.gamma_dtype);
    const ValueID beta = builder.parameter(refusal.beta, "beta", refusal.beta_dtype);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.layer_norm(x, gamma, beta, refusal.epsilon), nodes,
                   refusal.part);
}

INSTANTIATE_TEST_SUITE_P(, Small_LayerNormRefusal, ::testing::ValuesIn(K_LAYER_NORM_REFUSALS),
                         row_name<LayerNormRefusal>);

/** @brief Tests that the gradient node reads five operands and writes dx, dgamma and dbeta. */
TEST(Small_LayerNormOps, LayerNormBackwardWritesTheThreeGradients)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Leaves leaves = declare_leaves(builder);
    const Result<LayerNormResult> out =
        builder.layer_norm(leaves.x, leaves.gamma, leaves.beta, K_EPSILON);
    ASSERT_TRUE(out.has_value());
    const ValueID dy = builder.input(Shape{2, 3, K_WIDTH}, "dy");

    const Result<LayerNormGrads> grads = builder.layer_norm_backward(
        dy, leaves.x, leaves.gamma, out.value().saved_mean, out.value().saved_inv_std);

    SA_EXPECT_OK(grads);
    ASSERT_TRUE(grads.has_value());
    const Graph::Node& node = graph.node(graph.value(grads.value().dx).producer);
    EXPECT_EQ(node.op, Ops::K_LAYER_NORM_BACKWARD.id);
    EXPECT_EQ(node.inputs,
              (std::vector<ValueID>{dy, leaves.x, leaves.gamma, out.value().saved_mean,
                                    out.value().saved_inv_std}));
    EXPECT_EQ(node.outputs, (std::vector<ValueID>{grads.value().dx, grads.value().dgamma,
                                                  grads.value().dbeta}));
    EXPECT_EQ(graph.value(grads.value().dx).shape, (Shape{2, 3, K_WIDTH}));
    EXPECT_EQ(graph.value(grads.value().dgamma).shape, (Shape{K_WIDTH}));
    EXPECT_EQ(graph.value(grads.value().dbeta).shape, (Shape{K_WIDTH}));
}

/** @brief Tests that a gradient or a saved statistic of another shape is refused. */
TEST(Small_LayerNormOps, LayerNormBackwardRefusesOperandsThatDoNotFit)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const Leaves leaves = declare_leaves(builder);
    const Result<LayerNormResult> out =
        builder.layer_norm(leaves.x, leaves.gamma, leaves.beta, K_EPSILON);
    ASSERT_TRUE(out.has_value());
    const ValueID dy = builder.input(Shape{2, 3, K_WIDTH}, "dy");
    const ValueID flat = builder.input(Shape{6, K_WIDTH}, "flat");
    const ValueID per_column = builder.input(Shape{K_WIDTH}, "per_column");
    const std::size_t nodes = graph.node_count();

    expect_refused(graph,
                   builder.layer_norm_backward(flat, leaves.x, leaves.gamma,
                                               out.value().saved_mean,
                                               out.value().saved_inv_std),
                   nodes, "operand of [2, 3, 4]");
    expect_refused(graph,
                   builder.layer_norm_backward(dy, leaves.x, leaves.gamma, per_column,
                                               out.value().saved_inv_std),
                   nodes, "operand of [6]");
}

/** @brief Tests that the gradient is one node over [dy, log_probs]. */
TEST(Small_LogSoftmaxOps, TheGradientWritesItsIdentity)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 3}, "x", DType::Float64);
    const ValueID dy = builder.input(Shape{4, 3}, "dy", DType::Float64);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID log_probs, builder.log_softmax(x));

    SA_ASSERT_OK_AND_ASSIGN(const Traced out, builder.log_softmax_backward(dy, log_probs));

    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());
    const Graph::Node& node = graph.node(graph.value(out.value()).producer);
    EXPECT_EQ(node.op, Ops::K_LOG_SOFTMAX_BACKWARD.id);
    EXPECT_EQ(node.inputs, (std::vector<ValueID>{dy, log_probs}));
    EXPECT_EQ(node.outputs, (std::vector<ValueID>{out.value()}));
    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that the gradient keeps the shape, element type and lifetime of its operands. */
TEST(Small_LogSoftmaxOps, TheGradientKeepsShapeDTypeAndLifetime)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 3}, "x", DType::Float64);
    const ValueID dy = builder.input(Shape{4, 3}, "dy", DType::Float64);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID log_probs, builder.log_softmax(x));

    SA_ASSERT_OK_AND_ASSIGN(const Traced out, builder.log_softmax_backward(dy, log_probs));

    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(graph.value(out.value()).shape, (Shape{4, 3}));
    EXPECT_EQ(graph.value(out.value()).dtype, DType::Float64);
    EXPECT_EQ(graph.value(out.value()).lifetime, Graph::Lifetime::Activation);
}

namespace
{
    /** @brief Holds one gradient and log-probability pair that is refused, and the part named. */
    struct LogSoftmaxRefusal
    {
        std::string_view name; /**< The label of the case. */
        Shape gradient;        /**< The shape of the gradient. */
        DType gradient_dtype;  /**< The element type of the gradient. */
        Shape log_probs;       /**< The shape of the log-probabilities. */
        DType log_probs_dtype; /**< The element type of the log-probabilities. */
        const char* part;      /**< The part the refusal names. */
    };

    /** @brief Lists the operand sets the gradient refuses. */
    const std::array<LogSoftmaxRefusal, 4> K_LOG_SOFTMAX_REFUSALS = {{
        {"RankThree", Shape{2, 4, 3}, DType::Float32, Shape{2, 4, 3}, DType::Float32,
         "rank-2 log-probabilities"},
        {"IntegerOperands", Shape{4, 3}, DType::Int32, Shape{4, 3}, DType::Int32,
         "half, float32 or float64 log-probabilities"},
        {"GradientOfAnotherShape", Shape{4, 5}, DType::Float32, Shape{4, 3}, DType::Float32,
         "[4, 5]"},
        {"GradientOfAnotherDType", Shape{4, 3}, DType::Float64, Shape{4, 3}, DType::Float32,
         "a gradient like them"},
    }};

    /** @brief Runs once per operand set the log-softmax gradient refuses. */
    class Small_LogSoftmaxRefusal : public ::testing::TestWithParam<LogSoftmaxRefusal>
    {

    };
}

/** @brief Tests each refusal: a rank, an integer, a gradient of another shape or type. */
TEST_P(Small_LogSoftmaxRefusal, TheGradientRefusesOperandsThatDoNotFit)
{
    const LogSoftmaxRefusal& refusal = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID dy = builder.input(refusal.gradient, "dy", refusal.gradient_dtype);
    const ValueID log_probs = builder.input(refusal.log_probs, "log_probs",
                                            refusal.log_probs_dtype);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.log_softmax_backward(dy, log_probs), nodes, refusal.part);
}

INSTANTIATE_TEST_SUITE_P(, Small_LogSoftmaxRefusal, ::testing::ValuesIn(K_LOG_SOFTMAX_REFUSALS),
                         row_name<LogSoftmaxRefusal>);

TEST(Small_Diagnostics, MatmulContractionMismatchNamesBothInnerExtents)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{4, 5}, "a");
    const ValueID b = builder.input(Shape{7, 3}, "b");

    SA_EXPECT_FAILURE(builder.matmul(a, b), StatusCode::InvalidDescription,
                      "MATMUL contracts 5 against 7", "[4, 5]", "[7, 3]");
}

TEST(Small_Diagnostics, MatmulRejectsARank1OperandAndPrintsWhatItGot)
{
    // The message shows both operand shapes; see tests/core/graph/README.md.
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID vector = builder.input(Shape{5}, "v");
    const ValueID matrix = builder.input(Shape{5, 3}, "m");

    SA_EXPECT_FAILURE(builder.matmul(vector, matrix), StatusCode::InvalidDescription,
                      "MATMUL needs two rank-2 operands", "[5]", "[5, 3]");
    SA_EXPECT_FAILURE(builder.matmul(matrix, vector), StatusCode::InvalidDescription,
                      "MATMUL needs two rank-2 operands", "[5, 3]", "[5]");
}

TEST(Small_Diagnostics, MatmulNamesBothDtypesWhenTheyDisagree)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{4, 5}, "a", DType::Float32);
    const ValueID b = builder.input(Shape{5, 3}, "b", DType::Float64);

    SA_EXPECT_FAILURE(builder.matmul(a, b), StatusCode::InvalidDescription,
                      "MATMUL needs operands of one element type", "float32", "float64");
}

namespace
{
    /** @brief Holds one bias-add call that is refused and the parts its message names. */
    struct BiasAddRefusal
    {
        std::string_view name; /**< The label of the case. */
        Shape x;               /**< The shape of the left operand. */
        Shape bias;            /**< The shape of the bias. */
        const char* message;   /**< The refusal's message. */
        const char* shape;     /**< The shape the message prints. */
        const char* detail;    /**< A second part the message names. */
    };

    /** @brief Lists the bias-add calls that are refused. */
    const std::array<BiasAddRefusal, 3> K_BIAS_ADD_REFUSALS = {{
        {"BiasOfAnotherLength", Shape{4, 3}, Shape{5}, "BIAS_ADD bias has 5 entries", "[4, 3]",
         "3 columns"},
        {"RankTwoBias", Shape{4, 3}, Shape{1, 3}, "BIAS_ADD needs a rank-1 bias", "[1, 3]",
         "[1, 3]"},
        {"RankOneLeftOperand", Shape{3}, Shape{3}, "BIAS_ADD needs a rank-2 left operand", "[3]",
         "[3]"},
    }};

    /** @brief Runs once per bias-add call that is refused. */
    class Small_BiasAddDiagnostic : public ::testing::TestWithParam<BiasAddRefusal>
    {

    };
}

/** @brief Tests that bias_add names the operand that does not line up and prints its shape. */
TEST_P(Small_BiasAddDiagnostic, BiasAddNamesTheAxisThatDoesNotLineUp)
{
    const BiasAddRefusal& refusal = GetParam();

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(refusal.x, "x");
    const ValueID bias = builder.input(refusal.bias, "bias");

    SA_EXPECT_FAILURE(builder.bias_add(x, bias), StatusCode::InvalidDescription, refusal.message,
                      refusal.shape, refusal.detail);
}

INSTANTIATE_TEST_SUITE_P(, Small_BiasAddDiagnostic, ::testing::ValuesIn(K_BIAS_ADD_REFUSALS),
                         row_name<BiasAddRefusal>);

TEST(Small_Diagnostics, SumRowsAndLogSoftmaxRejectARank1Operand)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID vector = builder.input(Shape{6}, "v");

    SA_EXPECT_FAILURE(builder.sum_rows(vector), StatusCode::InvalidDescription,
                      "SUM_ROWS needs a rank-2 operand", "[6]");
    SA_EXPECT_FAILURE(builder.log_softmax(vector), StatusCode::InvalidDescription,
                      "LOG_SOFTMAX needs a rank-2 operand", "[6]");
}

TEST(Small_Diagnostics, ElementwiseOperationsNameBothShapesAndBothDtypes)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID a = builder.input(Shape{4, 3}, "a");
    const ValueID b = builder.input(Shape{4, 2}, "b");
    const ValueID wide = builder.input(Shape{4, 3}, "wide", DType::Float64);

    SA_EXPECT_FAILURE(builder.add(a, b), StatusCode::InvalidDescription,
                      "ADD needs equally shaped operands", "[4, 3]", "[4, 2]");
    SA_EXPECT_FAILURE(builder.sub(a, b), StatusCode::InvalidDescription,
                      "SUB needs equally shaped operands");
    SA_EXPECT_FAILURE(builder.mul(a, b), StatusCode::InvalidDescription,
                      "MUL needs equally shaped operands");
    SA_EXPECT_FAILURE(builder.mul(a, wide), StatusCode::InvalidDescription,
                      "MUL needs operands of one element type", "float32", "float64");
    SA_EXPECT_FAILURE(builder.accumulate(a, b), StatusCode::InvalidDescription,
                      "ACCUMULATE needs equally shaped operands");
}

TEST(Small_Diagnostics, NLLLossNamesTheShapeItCannotServe)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID log_probs = builder.input(Shape{4, 3}, "logp");
    const ValueID mismatched = builder.input(Shape{4, 5}, "targets");
    const ValueID vector = builder.input(Shape{4}, "v");

    SA_EXPECT_FAILURE(builder.nll_loss(log_probs, mismatched), StatusCode::InvalidDescription,
                      "NLL_LOSS needs equally shaped operands", "[4, 3]", "[4, 5]");
    SA_EXPECT_FAILURE(builder.nll_loss(vector, vector), StatusCode::InvalidDescription,
                      "NLL_LOSS needs rank-2 operands", "[4]");
}

/** @brief Tests that activation_backward stops the process for an identity it does not know. */
TEST(Medium_DiagnosticsDeathTest, ActivationBackwardRefusesAnOperationThatIsNotOne)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID dy = builder.input(Shape{4, 3}, "dy");

    SA_EXPECT_FATAL((void)builder.activation_backward(Ops::K_RELU.id, dy, dy), "sushiai.relu",
                    "is not an activation gradient");
}

namespace
{
    /** @brief Holds a fresh graph and a builder over it, for the refusal-family tests. */
    struct RefusalBench
    {
        Graph::Graph graph;              /**< @brief The graph the builder appends to. */
        GraphBuilder builder{graph};     /**< @brief The builder under test. */

        /** @brief Declares an NHWC [2,4,4,3] float32 input. */
        ValueID image() const
        {
            return builder.input(Shape{2, 4, 4, 3}, "image");
        }
    };

    /** @brief Returns a 2x2 window of unit stride and dilation and no padding. */
    Graph::ConvGeometry window_2x2()
    {
        Graph::ConvGeometry g;
        g.kernel_h = 2;
        g.kernel_w = 2;
        return g;
    }

    /** @brief Returns a window that no 4x4 input holds. */
    Graph::ConvGeometry window_9x9()
    {
        Graph::ConvGeometry g;
        g.kernel_h = 9;
        g.kernel_w = 9;
        return g;
    }
}

/** @brief Tests that the free window helpers return InvalidDescription for a bad geometry. */
TEST(Small_RefusalFamilies, WindowHelpersReturnTheirRefusal)
{
    Graph::ConvGeometry zero = window_2x2();
    zero.stride_h = 0;
    SA_EXPECT_FAILURE(Graph::window_extents(4, 4, zero), StatusCode::InvalidDescription,
                      "positive stride");
    SA_EXPECT_FAILURE(Graph::window_extents(4, 4, window_9x9()), StatusCode::InvalidDescription,
                      "fit its padded input");
    SA_EXPECT_FAILURE(Graph::upsampled_extents(4, 4, 0, 2), StatusCode::InvalidDescription,
                      "scales of at least one");
    SA_EXPECT_FAILURE(Graph::upsampled_extents(std::numeric_limits<int64_t>::max(), 4, 2, 2),
                      StatusCode::InvalidDescription, "fit in 64 bits");

    SA_ASSERT_OK_AND_ASSIGN(const Graph::SpatialExtents fitted,
                            Graph::window_extents(4, 4, window_2x2()));
    EXPECT_EQ(fitted.height, 3);
    EXPECT_EQ(fitted.width, 3);
}

/** @brief Tests that an emitter handed a failed operand returns it and emits nothing. */
TEST(Small_RefusalFamilies, AFailedOperandPassesThroughWithoutANode)
{
    RefusalBench bench;
    const ValueID x = bench.builder.input(Shape{4, 3}, "x");
    const std::size_t nodes = bench.graph.node_count();

    const Traced refused = bench.builder.reshape(x, Shape{5});
    const Traced chained = bench.builder.relu(bench.builder.matmul(refused, x, false, true));

    SA_EXPECT_FAILURE(chained, StatusCode::InvalidDescription, "element count");
    EXPECT_EQ(chained.status().message().data(), refused.status().message().data());
    EXPECT_EQ(bench.graph.node_count(), nodes);
    EXPECT_TRUE(bench.graph.has_refusal());
}

/** @brief Tests the refusals of the scaling, activation, cast, layout and reshape emitters. */
TEST(Small_RefusalFamilies, ScaleActivationCastAndLayoutEmittersRefuse)
{
    RefusalBench bench;
    const GraphBuilder& builder = bench.builder;
    const ValueID x = builder.input(Shape{4, 3}, "x");
    const ValueID pair = builder.input(Shape{2}, "pair");
    const ValueID index = builder.input(Shape{4, 3}, "index", DType::Int32);
    const ValueID target = builder.input(Shape{4, 5}, "target", DType::Float64);
    const std::size_t nodes = bench.graph.node_count();

    expect_refused(bench.graph, builder.dynamic_scale(x, pair), nodes, "single element");
    expect_refused(bench.graph, builder.leaky_relu(x, 1.0), nodes, "slope must be in (0, 1)");
    expect_refused(bench.graph, builder.cast(index, DType::Float32), nodes,
                   "real floating types only");
    expect_refused(bench.graph, builder.cast_into(x, target), nodes, "one shape");
    expect_refused(bench.graph, builder.layout_convert(x, Graph::TensorLayout::NCHW), nodes,
                   "rank-4 operand");
    expect_refused(bench.graph, builder.reshape(x, Shape{5}), nodes, "element count");
}

/** @brief Tests the refusals of the window emitters and of their gradients. */
TEST(Small_RefusalFamilies, WindowEmittersRefuse)
{
    RefusalBench bench;
    const GraphBuilder& builder = bench.builder;
    const ValueID image = bench.image();
    const ValueID flat = builder.input(Shape{4, 3}, "flat");
    const ValueID weight = builder.parameter(Shape{5, 2, 2, 4}, "weight");
    const ValueID dy = builder.input(Shape{2, 3, 3, 3}, "dy");
    const ValueID floats = builder.input(Shape{2, 3, 3, 3}, "floats");
    const Shape image_shape{2, 4, 4, 3};
    const std::size_t nodes = bench.graph.node_count();

    expect_refused(bench.graph, builder.im2col(image, window_9x9()), nodes, "fit its padded");
    expect_refused(bench.graph, builder.im2col_with_extents(image, window_2x2(), 0, 3), nodes,
                   "non-positive output extents");
    expect_refused(bench.graph, builder.col2im(flat, Shape{2, 4, 4}, window_2x2()), nodes,
                   "rank-4 [N,H,W,C] image");
    expect_refused(bench.graph,
                   builder.col2im_with_extents(flat, image_shape, window_2x2(), 3, 3), nodes,
                   "column matrix");
    expect_refused(bench.graph, builder.conv2d(image, weight, Traced(), window_2x2()), nodes,
                   "conv2d with a 2x2 window");
    expect_refused(bench.graph, builder.maxpool2d(flat, window_2x2()), nodes, "rank-4");
    expect_refused(bench.graph, builder.maxpool2d_with_extents(image, window_2x2(), 3, 0), nodes,
                   "non-positive output extents");
    expect_refused(bench.graph, builder.avgpool2d(image, window_9x9()), nodes, "fit its padded");
    expect_refused(bench.graph, builder.upsample2d(image, 0, 1), nodes, "at least one");
    expect_refused(bench.graph, builder.spatial_mean(flat), nodes, "rank-4");
    expect_refused(bench.graph,
                   builder.maxpool2d_backward(dy, floats, image_shape, window_2x2()), nodes,
                   "INT64");
    expect_refused(bench.graph, builder.avgpool2d_backward(flat, image_shape, window_2x2()),
                   nodes, "rank-4 [N,H,W,C] output gradient");
    expect_refused(bench.graph, builder.upsample2d_backward(dy, image_shape, 1, 0), nodes,
                   "at least one");
    expect_refused(bench.graph, builder.spatial_mean_backward(dy, image_shape), nodes,
                   "rank-2 [N,C]");
}

/** @brief Tests the refusals of the channel emitters and of batch normalisation. */
TEST(Small_RefusalFamilies, ChannelAndBatchNormEmittersRefuse)
{
    RefusalBench bench;
    const GraphBuilder& builder = bench.builder;
    const ValueID image = bench.image();
    const ValueID shallow = builder.input(Shape{2, 3, 4, 3}, "shallow");
    const ValueID flat = builder.input(Shape{4, 3}, "flat");
    const ValueID gamma = builder.parameter(Shape{3}, "gamma");
    const ValueID wide = builder.parameter(Shape{4}, "wide");
    const ValueID doubled = builder.parameter(Shape{3}, "doubled", DType::Float64);
    const std::array<ValueID, 2> mismatched = {image, shallow};
    const std::size_t nodes = bench.graph.node_count();

    expect_refused(bench.graph, builder.concat_channels(SushiRuntime::span<const ValueID>()),
                   nodes, "at least one operand");
    expect_refused(bench.graph,
                   builder.concat_channels(SushiRuntime::span<const ValueID>(
                       mismatched.data(), mismatched.size())),
                   nodes, "agree on N, H and W");
    expect_refused(bench.graph, builder.slice_channels(image, 2, 2), nodes, "does not fit");
    expect_refused(bench.graph, builder.channel_affine(image, gamma, wide), nodes,
                   "per-channel parameters must be [3]");
    expect_refused(bench.graph, builder.sum_channels(flat), nodes, "rank-4");
    expect_refused(bench.graph, builder.batchnorm2d(image, gamma, gamma, 0.0), nodes,
                   "epsilon must be positive");
    expect_refused(bench.graph, builder.batchnorm2d(image, gamma, doubled, 1e-5), nodes,
                   "element type");
    expect_refused(bench.graph, builder.batchnorm2d_backward(flat, image, gamma, gamma, gamma),
                   nodes, "rank-4 [N,H,W,C] output gradient");
    expect_refused(bench.graph, builder.batchnorm_fold(gamma, wide, gamma, gamma, 1e-5), nodes,
                   "beta shaped like gamma");
}
