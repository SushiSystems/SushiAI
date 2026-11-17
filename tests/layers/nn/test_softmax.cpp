/**************************************************************************/
/* test_softmax.cpp                                                       */
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
 * @file test_softmax.cpp
 * @brief Tests the softmax row: its axis rule, its factory, its values and its gradient.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/softmax.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/gradient_probe.hpp>
#include <support/host_row_values.hpp>
#include <support/test_names.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using NN::Softmax;
using NN::Trace;
namespace SaTest = SushiAI::Test;

namespace
{
    using SaTest::require_value;

    /** @brief Builds the node of a softmax row over @p input with @p attributes as JSON. */
    Result<std::unique_ptr<NN::Node>> make_row(const Shape& input, const std::string& attributes)
    {
        const Config::Value row = require_value(Config::parse_json(attributes, "<test>"));
        const Shape shapes[] = {input};

        return NN::make_softmax_node(NN::ShapeSpan(shapes, 1), row, "layer 2 'softmax'");
    }

    /** @brief Counts the nodes of @p graph that apply @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                ++count;
        }

        return count;
    }

    /** @brief Holds a graph a softmax was traced into, with what the trace declared. */
    struct TracedSoftmax
    {
        Graph::Graph graph;                        /**< The graph the layer was traced into. */
        Graph::ValueID input = Graph::K_NO_VALUE;  /**< The input leaf. */
        Graph::ValueID output = Graph::K_NO_VALUE; /**< The traced result. */
        Status status;                             /**< The status of the trace. */
    };

    /** @brief Fills @p traced by tracing a softmax over @p axis of a float64 leaf of @p extents. */
    void trace_softmax(TracedSoftmax& traced, const Shape& extents, int64_t axis)
    {
        const Graph::GraphBuilder builder(traced.graph);
        Trace trace = Trace::evaluation(builder);
        traced.input = builder.input(extents, "x", DType::Float64);

        Softmax layer{axis};
        traced.output = layer.forward(trace, traced.input);
        traced.status = trace.status();
    }

    /**
     * @brief Returns the softmax of @p values over the axis @p axis of @p extents.
     *
     * The elements that share every index but the one along @p axis form a group; each element is
     * exp(x) over the sum of exp over its group.
     */
    std::vector<double> softmax_over(const std::vector<double>& values,
                                     const std::vector<int64_t>& extents, std::size_t axis)
    {
        int64_t inner = 1;
        for (std::size_t position = axis + 1; position < extents.size(); ++position)
            inner *= extents[position];
        const int64_t length = extents[axis];
        const int64_t outer = static_cast<int64_t>(values.size()) / (length * inner);

        std::vector<double> result(values.size());
        for (int64_t high = 0; high < outer; ++high)
        {
            for (int64_t low = 0; low < inner; ++low)
            {
                double total = 0.0;
                for (int64_t at = 0; at < length; ++at)
                {
                    const std::size_t flat =
                        static_cast<std::size_t>((high * length + at) * inner + low);
                    total += std::exp(values[flat]);
                }

                for (int64_t at = 0; at < length; ++at)
                {
                    const std::size_t flat =
                        static_cast<std::size_t>((high * length + at) * inner + low);
                    result[flat] = std::exp(values[flat]) / total;
                }
            }
        }

        return result;
    }

    /** @brief Returns @p count inputs that differ and span a few units. */
    std::vector<double> spread(std::size_t count)
    {
        std::vector<double> values(count);
        for (std::size_t index = 0; index < count; ++index)
            values[index] = static_cast<double>((index * 7) % 11) / 4.0 - 1.0;

        return values;
    }

    /** @brief Returns @p extents as a shape. */
    Shape shape_of(const std::vector<int64_t>& extents)
    {
        return require_value(
            Shape::make(SushiRuntime::span<const int64_t>(extents.data(), extents.size())));
    }

    /** @brief Returns the number of elements of a value of @p extents. */
    std::size_t element_count(const std::vector<int64_t>& extents)
    {
        return static_cast<std::size_t>(shape_of(extents).num_elements());
    }

    /** @brief Holds one softmax over an axis of a value. */
    struct AxisRow
    {
        std::string name;             /**< The row's name. */
        std::vector<int64_t> extents; /**< The extents of the value, batch axis first. */
        int64_t axis;                 /**< The per-sample axis, negative from the end. */
        std::size_t position;         /**< The axis of the value the softmax runs over. */
    };

    /** @brief Names an AxisRow for gtest. */
    std::string axis_name(const ::testing::TestParamInfo<AxisRow>& info)
    {
        return info.param.name;
    }

    /** @brief Runs each axis row on the host. */
    class Small_SoftmaxValues : public ::testing::TestWithParam<AxisRow>
    {

    };

    /**
     * @brief Bounds the rounding of one element: the shift by the maximum, the exponential, the
     *        sum over a group of at most three, the logarithm of the sum, the difference and the
     *        final exponential, each within one unit roundoff.
     */
    constexpr std::size_t K_SOFTMAX_OPERATIONS = 12;
} // namespace

/** @brief Tests that the softmax of each axis matches the exponentials over their group. */
TEST_P(Small_SoftmaxValues, TheValuesAreTheExponentialsOverTheirGroup)
{
    const AxisRow& row = GetParam();
    TracedSoftmax traced;
    trace_softmax(traced, shape_of(row.extents), row.axis);
    ASSERT_TRUE(traced.status.is_ok()) << traced.status.message();

    const std::vector<double> input = spread(element_count(row.extents));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<double> got,
        SaTest::evaluate_on_host(traced.graph, {{traced.input, input}}, traced.output));

    const std::vector<double> expected = softmax_over(input, row.extents, row.position);
    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
        EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
            got[index], expected[index], K_SOFTMAX_OPERATIONS)))
            << "element " << index;
}

INSTANTIATE_TEST_SUITE_P(
    Axes, Small_SoftmaxValues,
    ::testing::Values(AxisRow{"LastOfAVector", {2, 3}, -1, 1},
                      AxisRow{"FirstOfAMatrix", {2, 3, 2}, 0, 1},
                      AxisRow{"FirstOfAMatrixCountedFromTheEnd", {2, 3, 2}, -2, 1},
                      AxisRow{"LastOfAMatrix", {2, 3, 2}, 1, 2},
                      AxisRow{"MiddleOfThreeAxes", {2, 2, 3, 2}, 1, 2},
                      AxisRow{"FirstOfThreeAxes", {2, 2, 3, 2}, 0, 1}),
    axis_name);

/** @brief Tests that the last axis needs no transpose: the softmax of a row is the chain alone. */
TEST(Small_Softmax, TheLastAxisEmitsNoTranspose)
{
    TracedSoftmax traced;
    trace_softmax(traced, Shape{2, 3, 4}, -1);

    ASSERT_TRUE(traced.status.is_ok()) << traced.status.message();
    EXPECT_EQ(count_nodes(traced.graph, Ops::K_LOG_SOFTMAX.id), 1u);
    EXPECT_EQ(count_nodes(traced.graph, Ops::K_TRANSPOSE.id), 0u);
    EXPECT_EQ(traced.graph.value(traced.output).shape, (Shape{2, 3, 4}));
}

/** @brief Tests that another axis is moved last and back, so its result keeps the input shape. */
TEST(Small_Softmax, AnotherAxisIsMovedLastAndBack)
{
    TracedSoftmax traced;
    trace_softmax(traced, Shape{2, 3, 4}, 0);

    ASSERT_TRUE(traced.status.is_ok()) << traced.status.message();
    EXPECT_EQ(count_nodes(traced.graph, Ops::K_LOG_SOFTMAX.id), 1u);
    EXPECT_EQ(count_nodes(traced.graph, Ops::K_TRANSPOSE.id), 2u);
    EXPECT_EQ(count_nodes(traced.graph, Ops::K_MATMUL.id), 0u);
    EXPECT_EQ(traced.graph.value(traced.output).shape, (Shape{2, 3, 4}));
}

/** @brief Tests that an axis outside the per-sample rank fails the trace and keeps the input. */
TEST(Small_Softmax, AnAxisOutsideTheRankFailsTheTrace)
{
    TracedSoftmax traced;
    trace_softmax(traced, Shape{2, 3, 4}, 2);

    EXPECT_EQ(traced.output, traced.input);
    SA_EXPECT_FAILURE(traced.status, StatusCode::InvalidDescription, "axis 2");
}

namespace
{
    /** @brief Holds one description the factory refuses and the words the message holds. */
    struct RefusalRow
    {
        std::string name;       /**< The row's name. */
        Shape input;            /**< The per-sample input shape. */
        std::string attributes; /**< The row's keys as JSON. */
        std::string fragment;   /**< What the message holds besides the context. */
    };

    /** @brief Names a RefusalRow for gtest. */
    std::string refusal_name(const ::testing::TestParamInfo<RefusalRow>& info)
    {
        return info.param.name;
    }

    /** @brief Runs each refusal row. */
    class Small_SoftmaxRefusal : public ::testing::TestWithParam<RefusalRow>
    {

    };
} // namespace

/** @brief Tests that the factory refuses a description and names the row and the fault. */
TEST_P(Small_SoftmaxRefusal, TheFactoryRefusesTheRow)
{
    const RefusalRow& row = GetParam();

    SA_EXPECT_FAILURE(make_row(row.input, row.attributes), StatusCode::InvalidDescription,
                      "layer 2 'softmax'", row.fragment);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_SoftmaxRefusal,
    ::testing::Values(RefusalRow{"AnAxisPastTheRank", Shape{3, 2}, R"({"axis": 2})", "axis 2"},
                      RefusalRow{"AnAxisBeforeTheFirst", Shape{3, 2}, R"({"axis": -3})", "axis -3"},
                      RefusalRow{"AnAxisThatIsNoInteger", Shape{3, 2}, R"({"axis": "last"})",
                                 "axis"},
                      RefusalRow{"AnUnknownKey", Shape{3, 2}, R"({"dim": 1})", "'dim'"},
                      RefusalRow{"AnInputOfRankZero", Shape{}, "{}", "rank 1 or more"}),
    refusal_name);

/** @brief Tests that a row with no key softmaxes the last axis and keeps the input shape. */
TEST(Small_Softmax, TheDefaultAxisIsTheLastAndThePortKeepsTheShape)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node, make_row(Shape{4, 3}, "{}"));

    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].name, "output");
    EXPECT_EQ(node->ports()[0].shape, (Shape{4, 3}));
    EXPECT_EQ(node->input_count(), 1u);
}

/** @brief Tests that the output shape is the input shape for the first and the last axis. */
TEST(Small_Softmax, TheOutputShapeIsTheInputShapeForAnyAxisInRange)
{
    const Softmax first{0};
    const Softmax last{-1};

    EXPECT_EQ(first.output_shape(Shape{5, 3}).value(), (Shape{5, 3}));
    EXPECT_EQ(last.output_shape(Shape{5, 3}).value(), (Shape{5, 3}));
}

namespace
{
    /** @brief Provides the Engine every gradient case records on. */
    class Medium_SoftmaxGradient : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that the gradient of a softmax over the first sample axis matches central differences. */
TEST_F(Medium_SoftmaxGradient, TheGradientOverAnInnerAxisMatchesCentralDifferences)
{
    SaTest::GradientProbe probe(backend(), Shape{2, 3, 2},
                                [](const Graph::GraphBuilder& builder, Graph::ValueID x)
    {
        Trace trace = Trace::evaluation(builder);
        Softmax layer{0};
        const Graph::ValueID normalised = layer.forward(trace, x);
        return builder.reshape(normalised, Shape{2, 6});
    });

    std::vector<double> point = spread(12);
    SaTest::expect_gradient_matches(probe, point);
}
