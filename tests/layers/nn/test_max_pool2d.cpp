/**************************************************************************/
/* test_max_pool2d.cpp                                                    */
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
 * @file test_max_pool2d.cpp
 * @brief Tests the 2-D max pooling layer: its trace and the values of a window that runs past the
 *        edge.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/max_pool2d.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using NN::MaxPool2d;
// SaTest stays reachable inside a TEST_F body; see tests/layers/nn/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::write_tensor;

static_assert(NN::Module<MaxPool2d>, "MaxPool2d must satisfy the Module contract");
static_assert(!NN::ModuleContainer<MaxPool2d>, "MaxPool2d must not look like a container");

namespace
{
    /** @brief Builds a layer whose window has the given taps, steps and leading pads. */
    MaxPool2d make_pool(int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                        int64_t pad_top = 0, int64_t pad_left = 0)
    {
        MaxPool2d layer;
        layer.geometry.kernel_h = kernel_h;
        layer.geometry.kernel_w = kernel_w;
        layer.geometry.stride_h = stride_h;
        layer.geometry.stride_w = stride_w;
        layer.geometry.pad_top = pad_top;
        layer.geometry.pad_left = pad_left;
        return layer;
    }

    /** @brief Traces @p layer over a batch of @p batch samples and returns the result's shape. */
    Shape traced_shape(const MaxPool2d& layer, int64_t batch, const Shape& sample)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);

        const Graph::ValueID x =
            builder.input(Shape{batch, sample[0], sample[1], sample[2]}, "x");
        // The input fits, so a refusal is a defect of the layer: value() stops the test.
        return graph.value(layer.forward(builder, x).value()).shape;
    }

    /** @brief Prepends a batch extent to a per-sample [H, W, C] shape. */
    Shape with_batch(int64_t batch, const Shape& sample)
    {
        return Shape{batch, sample[0], sample[1], sample[2]};
    }
} // namespace

/** @brief Tests that a 2x2 stride-2 pool halves both spatial extents and keeps N and C. */
TEST(Small_MaxPool2d, ForwardHalvesTheSpatialExtentsOfAnEvenInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 3, 3}));
}

/** @brief Tests that the traced rank-4 result carries the NHWC layout tag. */
TEST(Small_MaxPool2d, ForwardTagsItsResultNHWC)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    ASSERT_EQ(graph.value(y).shape.rank(), 4);
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

/** @brief Tests that forward() returns the pooled value and not the INT64 argmax beside it. */
TEST(Small_MaxPool2d, ForwardReturnsThePooledValueNotTheArgmax)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const MaxPool2d layer = make_pool(2, 2, 2, 2);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).dtype, graph.value(x).dtype);
    EXPECT_NE(graph.value(y).dtype, DType::Int64);
}

/** @brief Tests that output_shape() returns [out_h, out_w, C] for an unpadded square window. */
TEST(Small_MaxPool2d, OutputShapeOfAnUnpaddedSquareWindow)
{
    const MaxPool2d layer = make_pool(2, 2, 2, 2);

    EXPECT_EQ(layer.output_shape(Shape{8, 6, 3}).value(), (Shape{4, 3, 3}));
}

namespace
{
    /** @brief Holds a layer, a batch, a per-sample input and the per-sample shape it must give. */
    struct ExtentRow
    {
        std::string name;  /**< The row's name. */
        MaxPool2d layer;   /**< The layer under test. */
        int64_t batch;     /**< The batch extent the trace runs over. */
        Shape sample;      /**< The per-sample [H, W, C] input. */
        Shape expected;    /**< The per-sample shape output_shape() must return. */
    };

    /** @brief Returns @p layer with a trailing pad of @p height rows and @p width columns. */
    MaxPool2d with_trailing(MaxPool2d layer, int64_t height, int64_t width)
    {
        layer.trailing_pad_h = height;
        layer.trailing_pad_w = width;
        return layer;
    }

    /** @brief Returns @p layer with the dilation @p height by @p width. */
    MaxPool2d with_dilation(MaxPool2d layer, int64_t height, int64_t width)
    {
        layer.geometry.dilation_h = height;
        layer.geometry.dilation_w = width;
        return layer;
    }
}

/** @brief Runs the extent case over each window geometry. */
class Small_MaxPool2dExtent : public ::testing::TestWithParam<ExtentRow>
{
};

/** @brief Tests that output_shape() gives the extent of the row and agrees with the traced shape. */
TEST_P(Small_MaxPool2dExtent, OutputShapeAgreesWithForward)
{
    const ExtentRow& row = GetParam();

    EXPECT_EQ(row.layer.output_shape(row.sample).value(), row.expected);
    EXPECT_EQ(traced_shape(row.layer, row.batch, row.sample), with_batch(row.batch, row.expected));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_MaxPool2dExtent,
    ::testing::Values(
        ExtentRow{"ASquareWindow", make_pool(2, 2, 2, 2), 2, Shape{8, 6, 3}, Shape{4, 3, 3}},
        ExtentRow{"APaddedAnisotropicWindow", make_pool(3, 2, 2, 1, 1, 0), 3, Shape{5, 7, 4},
                  Shape{3, 6, 4}},
        ExtentRow{"AStrideThatLeavesARemainder", make_pool(3, 3, 2, 2), 1, Shape{8, 10, 1},
                  Shape{3, 4, 1}},
        ExtentRow{"ADilatedAnisotropicWindow", with_dilation(make_pool(2, 2, 1, 1), 2, 3), 2,
                  Shape{7, 9, 2}, Shape{5, 6, 2}},
        ExtentRow{"WithoutATrailingRowAtStrideOne", make_pool(2, 2, 1, 1), 2, Shape{13, 13, 4},
                  Shape{12, 12, 4}},
        ExtentRow{"ATrailingRowKeepsTheExtentAtStrideOne",
                  with_trailing(make_pool(2, 2, 1, 1), 1, 1), 2, Shape{13, 13, 4},
                  Shape{13, 13, 4}},
        ExtentRow{"ATrailingPadAtStrideTwoGivesSevenFromThirteen",
                  with_trailing(make_pool(2, 2, 2, 2), 1, 1), 1, Shape{13, 13, 4},
                  Shape{7, 7, 4}},
        ExtentRow{"ATrailingPadGivesTwoHundredEightFromFourHundredSixteen",
                  with_trailing(make_pool(2, 2, 2, 2), 1, 1), 1, Shape{416, 416, 16},
                  Shape{208, 208, 16}},
        ExtentRow{"ATrailingPadOnOneAxisOnly", with_trailing(make_pool(2, 2, 1, 1), 1, 0), 1,
                  Shape{3, 3, 2}, Shape{3, 2, 2}},
        ExtentRow{"ATrailingPadLetsAWindowFitAShortInput",
                  with_trailing(make_pool(2, 2, 1, 1), 1, 1), 1, Shape{1, 4, 3},
                  Shape{1, 4, 3}}),
    [](const ::testing::TestParamInfo<ExtentRow>& info) { return info.param.name; });

namespace
{
    /** @brief Holds a layer, an input shape forward() refuses and the words its message names. */
    struct ForwardRefusalRow
    {
        std::string name;   /**< The row's name. */
        MaxPool2d layer;    /**< The layer under test. */
        Shape input;        /**< The shape of the batched input. */
        bool nchw;          /**< Whether the input is converted to NCHW first. */
        std::string first;  /**< The first substring the message names; empty names none. */
        std::string second; /**< The second substring the message names. */
        std::string third;  /**< The third substring the message names. */
    };
}

/** @brief Runs the forward refusal over each input and geometry the layer does not take. */
class Small_MaxPool2dForwardRefusal : public ::testing::TestWithParam<ForwardRefusalRow>
{
};

/** @brief Tests that forward() refuses the row, names the reason and emits no node. */
TEST_P(Small_MaxPool2dForwardRefusal, ForwardRefuses)
{
    const ForwardRefusalRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Graph::ValueID x = builder.input(row.input, "x");
    if (row.nchw)
    {
        SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID converted,
                                builder.layout_convert(x, Graph::TensorLayout::NCHW));
        x = converted;
    }
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(row.layer.forward(builder, x), StatusCode::InvalidDescription, row.first,
                      row.second, row.third);
    EXPECT_EQ(graph.node_count(), nodes_before);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_MaxPool2dForwardRefusal,
    ::testing::Values(
        ForwardRefusalRow{"AnInputThatIsNotRankFour", make_pool(2, 2, 2, 2), Shape{2, 16}, false,
                          "", "", ""},
        ForwardRefusalRow{"AWindowThatDoesNotFit", make_pool(5, 5, 1, 1), Shape{2, 4, 4, 3}, false,
                          "", "", ""},
        ForwardRefusalRow{"AnNCHWInput", make_pool(2, 2, 2, 2), Shape{2, 8, 8, 3}, true, "", "",
                          ""},
        ForwardRefusalRow{"AnNCHWInputNarrowerThanTheWindowNamesTheLayout", make_pool(3, 3, 1, 1),
                          Shape{1, 8, 8, 2}, true, "NHWC", "", ""},
        ForwardRefusalRow{"AZeroStride", make_pool(2, 2, 0, 2), Shape{2, 8, 8, 3}, false,
                          "MaxPool2d", "stride 0,2", ""},
        ForwardRefusalRow{"ANegativePadding", make_pool(2, 2, 2, 2, 0, -1), Shape{2, 8, 8, 3},
                          false, "MaxPool2d", "padding 0,-1", ""},
        ForwardRefusalRow{"ANegativeTrailingPad", with_trailing(make_pool(2, 2, 1, 1), -2, 0),
                          Shape{2, 8, 8, 3}, false, "MaxPool2d", "non-negative trailing padding",
                          "-2,0"},
        ForwardRefusalRow{"ATrailingPadThatLeavesAWindowNoTap",
                          with_trailing(make_pool(2, 2, 2, 2), 0, 3), Shape{1, 4, 4, 2}, false,
                          "MaxPool2d", "every window a tap inside the input", "0,3"}),
    [](const ::testing::TestParamInfo<ForwardRefusalRow>& info) { return info.param.name; });

namespace
{
    /** @brief Holds a layer, a per-sample input output_shape() refuses and the words it names. */
    struct ShapeRefusalRow
    {
        std::string name;   /**< The row's name. */
        MaxPool2d layer;    /**< The layer under test. */
        Shape sample;       /**< The per-sample input. */
        std::string first;  /**< The first substring the message names; empty names none. */
        std::string second; /**< The second substring the message names. */
        std::string third;  /**< The third substring the message names. */
    };
}

/** @brief Runs the output_shape refusal over each geometry and shape the layer does not take. */
class Small_MaxPool2dShapeRefusal : public ::testing::TestWithParam<ShapeRefusalRow>
{
};

/** @brief Tests that output_shape() refuses the row and names the reason. */
TEST_P(Small_MaxPool2dShapeRefusal, OutputShapeRefuses)
{
    const ShapeRefusalRow& row = GetParam();

    SA_EXPECT_FAILURE(row.layer.output_shape(row.sample), StatusCode::InvalidDescription,
                      row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_MaxPool2dShapeRefusal,
    ::testing::Values(
        ShapeRefusalRow{"AZeroStride", make_pool(2, 2, 0, 2), Shape{8, 8, 3}, "MaxPool2d",
                        "stride 0,2", ""},
        ShapeRefusalRow{"ANegativePadding", make_pool(2, 2, 2, 2, 0, -1), Shape{8, 8, 3},
                        "MaxPool2d", "padding 0,-1", ""},
        ShapeRefusalRow{"AFeatureVector", make_pool(2, 2, 2, 2), Shape{16}, "", "", ""},
        ShapeRefusalRow{"ABatchedShape", make_pool(2, 2, 2, 2), Shape{2, 8, 8, 3}, "", "", ""},
        ShapeRefusalRow{"AWindowTallerThanTheInput", make_pool(5, 2, 1, 1), Shape{4, 8, 3}, "", "",
                        ""},
        ShapeRefusalRow{"AWindowWiderThanTheInput", make_pool(2, 5, 1, 1), Shape{8, 4, 3}, "", "",
                        ""},
        ShapeRefusalRow{"AShortInputWithoutATrailingPad", make_pool(2, 2, 1, 1), Shape{1, 4, 3},
                        "", "", ""},
        ShapeRefusalRow{"ANegativeTrailingPad", with_trailing(make_pool(2, 2, 1, 1), 0, -1),
                        Shape{8, 8, 3}, "MaxPool2d", "non-negative trailing padding", "0,-1"},
        ShapeRefusalRow{"ATrailingPadThatOverflows",
                        with_trailing(make_pool(2, 2, 1, 1), std::numeric_limits<int64_t>::max(),
                                      0),
                        Shape{8, 8, 3}, "MaxPool2d", "within 64 bits", ""},
        ShapeRefusalRow{"ATrailingPadThatLeavesAWindowNoTap",
                        with_trailing(make_pool(2, 2, 1, 1), 2, 0), Shape{3, 3, 2}, "MaxPool2d",
                        "every window a tap inside the input", "2,0"}),
    [](const ::testing::TestParamInfo<ShapeRefusalRow>& info) { return info.param.name; });

namespace
{
    /** @brief Holds the input of the pooled 3x3 value, one element of its result and that value. */
    struct TrailingElementRow
    {
        std::string name;            /**< The row's name. */
        std::vector<double> input;   /**< The 9 input values, row-major. */
        std::size_t element;         /**< The index of the pooled element compared. */
        double expected;             /**< The maximum of the truncated window at that element. */
    };

    /** @brief Provides the Engine every trailing-window case records on. */
    class Medium_MaxPoolTrailing : public SaTest::EngineTest,
                                   public ::testing::WithParamInterface<TrailingElementRow>
    {
    };

    /** @brief Builds a 2x2 window at stride one with one trailing row and one trailing column. */
    NN::MaxPool2d trailing_window()
    {
        return with_trailing(make_pool(2, 2, 1, 1), 1, 1);
    }

    /**
     * @brief Pools a 3x3 single-channel input with trailing_window() and reads the 3x3 result.
     * @param input The 9 input values, row-major.
     * @param pooled Receives the 9 pooled values, row-major.
     */
    void pool_three_by_three(SushiAI::ExecutionBackend& backend, const std::vector<double>& input,
                             std::vector<double>& pooled)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueID x = builder.input(Shape{1, 3, 3, 1}, "x");
        SA_ASSERT_OK_AND_ASSIGN(const ValueID y, trailing_window().forward(builder, x));

        TensorPool pool(backend.engine());
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        write_tensor(pool.tensor(x), input);

        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(backend, graph, pool));
        SA_EXPECT_OK(step->run());

        pooled = read_tensor(pool.tensor(y));
    }

    /**
     * @brief Lists one row per element of the pooled result of @p input.
     * @param prefix The name every row starts with.
     * @param input The 9 input values, row-major.
     * @param expected The 9 maxima of the truncated windows, row-major.
     */
    std::vector<TrailingElementRow> element_rows(const std::string& prefix,
                                                 const std::vector<double>& input,
                                                 const std::vector<double>& expected)
    {
        std::vector<TrailingElementRow> rows;
        for (std::size_t i = 0; i < expected.size(); ++i)
            rows.push_back({prefix + std::to_string(i), input, i, expected[i]});

        return rows;
    }

    /** @brief Lists the rows of both inputs: mixed signs, then all negative. */
    std::vector<TrailingElementRow> trailing_rows()
    {
        // Input [[5, -1, 2], [-3, 8, 0], [4, -7, 6]]; expected [[8, 8, 2], [8, 8, 6], [4, 6, 6]].
        std::vector<TrailingElementRow> rows =
            element_rows("TheMixedInputAtElement", {5, -1, 2, -3, 8, 0, 4, -7, 6},
                         {8, 8, 2, 8, 8, 6, 4, 6, 6});
        // Input runs from -18 to -10; a pad that took part in the maximum as zero would show as 0.
        const std::vector<TrailingElementRow> negative =
            element_rows("TheNegativeInputAtElement", {-18, -17, -16, -15, -14, -13, -12, -11, -10},
                         {-14, -13, -13, -11, -10, -10, -11, -10, -10});
        rows.insert(rows.end(), negative.begin(), negative.end());

        return rows;
    }
} // namespace

/**
 * @brief Tests that the pooled element of the row is the true maximum of its truncated window.
 *
 * A maximum selects an input value, so the result is exact and the bound is zero.
 */
TEST_P(Medium_MaxPoolTrailing, TheElementIsTheMaximumOfItsTruncatedTaps)
{
    const TrailingElementRow& row = GetParam();
    std::vector<double> pooled;
    ASSERT_NO_FATAL_FAILURE(pool_three_by_three(backend(), row.input, pooled));
    ASSERT_EQ(pooled.size(), row.input.size());

    EXPECT_TRUE(TestSupport::is_within(
        TestSupport::measure_error(pooled[row.element], row.expected, 0.0)));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_MaxPoolTrailing, ::testing::ValuesIn(trailing_rows()),
    [](const ::testing::TestParamInfo<TrailingElementRow>& info) { return info.param.name; });
