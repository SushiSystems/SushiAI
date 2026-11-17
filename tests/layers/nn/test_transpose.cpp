/**************************************************************************/
/* test_transpose.cpp                                                     */
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
 * @file test_transpose.cpp
 * @brief Tests the transpose row: its shape rule, its factory, its values and its gradient.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
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
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/nn/transpose.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/gradient_probe.hpp>
#include <support/host_row_values.hpp>
#include <support/test_names.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using NN::Trace;
using NN::Transpose;
namespace SaTest = SushiAI::Test;

namespace
{
    using SaTest::require_value;

    /** @brief Holds a permutation of the per-sample axes, written as a vector. */
    using Permutation = std::vector<int64_t>;

    /** @brief Builds the node of a transpose row over @p input with the key "perm" as JSON. */
    Result<std::unique_ptr<NN::Node>> make_row(const Shape& input, const std::string& attributes)
    {
        const Config::Value row = require_value(Config::parse_json(attributes, "<test>"));
        const Shape shapes[] = {input};

        return NN::make_transpose_node(NN::ShapeSpan(shapes, 1), row, "layer 4 'transpose'");
    }

    /**
     * @brief Returns the elements of @p source permuted: output axis i takes input axis
     *        @p permutation[i] over the axes of @p extents, the last axis fastest.
     *
     * The output element at the multi-index o reads the input element whose index along axis
     * permutation[i] is o[i], so the stride of that axis in the input is the product of the
     * extents behind it.
     */
    std::vector<double> permuted(const std::vector<double>& source,
                                 const std::vector<int64_t>& extents,
                                 const Permutation& permutation)
    {
        const std::size_t rank = extents.size();
        std::vector<int64_t> strides(rank, 1);
        for (std::size_t axis = rank - 1; axis > 0; --axis)
            strides[axis - 1] = strides[axis] * extents[axis];

        std::vector<int64_t> output_extents;
        for (std::size_t position = 0; position < rank; ++position)
            output_extents.push_back(extents[static_cast<std::size_t>(permutation[position])]);

        std::vector<double> result(source.size());
        std::vector<int64_t> index(rank, 0);
        for (std::size_t flat = 0; flat < source.size(); ++flat)
        {
            int64_t from = 0;
            for (std::size_t axis = 0; axis < rank; ++axis)
                from += index[axis] * strides[static_cast<std::size_t>(permutation[axis])];

            result[flat] = source[static_cast<std::size_t>(from)];
            for (std::size_t axis = rank; axis > 0; --axis)
            {
                if (++index[axis - 1] < output_extents[axis - 1])
                    break;

                index[axis - 1] = 0;
            }
        }

        return result;
    }

    /** @brief Returns 0, 1, 2 and so on, @p count of them. */
    std::vector<double> counted(std::size_t count)
    {
        std::vector<double> values(count);
        for (std::size_t index = 0; index < count; ++index)
            values[index] = static_cast<double>(index);

        return values;
    }

    /** @brief Holds one permutation with the shape it gives. */
    struct ShapeRow
    {
        std::string name;    /**< The row's name. */
        Shape input;         /**< The per-sample input shape. */
        std::string perm;    /**< The "perm" array as JSON. */
        Shape expected;      /**< The per-sample result shape. */
    };

    /** @brief Names a ShapeRow for gtest. */
    std::string shape_name(const ::testing::TestParamInfo<ShapeRow>& info)
    {
        return info.param.name;
    }

    /** @brief Runs each shape row. */
    class Small_TransposeShape : public ::testing::TestWithParam<ShapeRow>
    {

    };
} // namespace

/** @brief Tests that the port of the node holds the input extents in the order of the permutation. */
TEST_P(Small_TransposeShape, TheFactoryGivesThePermutedShapeAsThePort)
{
    const ShapeRow& row = GetParam();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            make_row(row.input, R"({"perm": )" + row.perm + "}"));

    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].name, "output");
    EXPECT_EQ(node->ports()[0].shape, row.expected);
    EXPECT_EQ(node->input_count(), 1u);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_TransposeShape,
    ::testing::Values(ShapeRow{"SwapOfAMatrix", Shape{5, 3}, "[1, 0]", Shape{3, 5}},
                      ShapeRow{"IdentityOfAVector", Shape{7}, "[0]", Shape{7}},
                      ShapeRow{"RotationOfThreeAxes", Shape{2, 3, 4}, "[2, 0, 1]", Shape{4, 2, 3}},
                      ShapeRow{"SwapOfTheInnerAxes", Shape{2, 3, 4}, "[0, 2, 1]", Shape{2, 4, 3}},
                      ShapeRow{"ReverseOfThreeAxes", Shape{2, 3, 4}, "[2, 1, 0]", Shape{4, 3, 2}}),
    shape_name);

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
    class Small_TransposeRefusal : public ::testing::TestWithParam<RefusalRow>
    {

    };
} // namespace

/** @brief Tests that the factory refuses a row that gives no permutation of the input axes. */
TEST_P(Small_TransposeRefusal, TheFactoryRefusesTheRow)
{
    const RefusalRow& row = GetParam();

    SA_EXPECT_FAILURE(make_row(row.input, row.attributes), StatusCode::InvalidDescription,
                      "layer 4 'transpose'", row.fragment);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_TransposeRefusal,
    ::testing::Values(
        RefusalRow{"TooFewAxes", Shape{2, 3, 4}, R"({"perm": [1, 0]})", "no permutation"},
        RefusalRow{"ARepeatedAxis", Shape{2, 3}, R"({"perm": [1, 1]})", "no permutation"},
        RefusalRow{"AnAxisOutOfRange", Shape{2, 3}, R"({"perm": [0, 2]})", "no permutation"},
        RefusalRow{"ANegativeAxis", Shape{2, 3}, R"({"perm": [0, -1]})", "no permutation"},
        RefusalRow{"NoPermKey", Shape{2, 3}, "{}", "perm"},
        RefusalRow{"APermThatIsNoArray", Shape{2, 3}, R"({"perm": 1})", "perm"},
        RefusalRow{"AnEntryThatIsNoInteger", Shape{2, 3}, R"({"perm": [0, "x"]})", "perm[1]"},
        RefusalRow{"AnUnknownKey", Shape{2, 3}, R"({"perm": [1, 0], "axis": 1})", "'axis'"}),
    refusal_name);

/** @brief Tests that a row over a rank-0 shape is refused. */
TEST(Small_Transpose, TheFactoryRefusesAnInputOfRankZero)
{
    SA_EXPECT_FAILURE(make_row(Shape{}, R"({"perm": []})"), StatusCode::InvalidDescription,
                      "layer 4 'transpose'", "rank 1 or more");
}

namespace
{
    /** @brief Holds one permutation of a [2, 3, 4] sample. */
    struct ValueRow
    {
        std::string name;  /**< The row's name. */
        Permutation perm;  /**< The permutation of the three per-sample axes. */
    };

    /** @brief Names a ValueRow for gtest. */
    std::string value_name(const ::testing::TestParamInfo<ValueRow>& info)
    {
        return info.param.name;
    }

    /** @brief Runs each permutation over the host. */
    class Small_TransposeValues : public ::testing::TestWithParam<ValueRow>
    {

    };

    /** @brief Holds a graph a transpose was traced into, with what the trace declared. */
    struct TracedTranspose
    {
        Graph::Graph graph;       /**< The graph the layer was traced into. */
        Graph::ValueID input = Graph::K_NO_VALUE;  /**< The input leaf. */
        Graph::ValueID output = Graph::K_NO_VALUE; /**< The traced result. */
        Status status;            /**< The status of the trace. */
    };

    /** @brief Fills @p traced by tracing a transpose of @p permutation over a leaf of @p extents. */
    void trace_transpose(TracedTranspose& traced, const Shape& extents, const Permutation& permutation,
                         DType dtype = DType::Float64)
    {
        const Graph::GraphBuilder builder(traced.graph);
        Trace trace = Trace::evaluation(builder);
        traced.input = builder.input(extents, "x", dtype);

        Transpose layer{permutation};
        traced.output = layer.forward(trace, traced.input);
        traced.status = trace.status();
    }
} // namespace

/** @brief Tests the values of each permutation of [2, 3, 4] against index arithmetic. */
TEST_P(Small_TransposeValues, TheValuesFollowThePermutation)
{
    const ValueRow& row = GetParam();
    TracedTranspose traced;
    trace_transpose(traced, Shape{2, 2, 3, 4}, row.perm);
    ASSERT_TRUE(traced.status.is_ok()) << traced.status.message();

    Permutation whole{0};
    for (const int64_t axis : row.perm)
        whole.push_back(axis + 1);
    const std::vector<double> input = counted(48);
    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<double> got,
        SaTest::evaluate_on_host(traced.graph, {{traced.input, input}}, traced.output));

    const std::vector<double> expected = permuted(input, {2, 2, 3, 4}, whole);
    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
        EXPECT_TRUE(TestSupport::is_within(
            TestSupport::measure_error(got[index], expected[index], 0.0)))
            << "element " << index;
}

INSTANTIATE_TEST_SUITE_P(
    Permutations, Small_TransposeValues,
    ::testing::Values(ValueRow{"SwapFirstTwo", {1, 0, 2}}, ValueRow{"SwapLastTwo", {0, 2, 1}},
                      ValueRow{"RotateLeft", {1, 2, 0}}, ValueRow{"RotateRight", {2, 0, 1}},
                      ValueRow{"Reverse", {2, 1, 0}}),
    value_name);

/** @brief Tests the values of a swap over an axis of extent one, which moves no element. */
TEST(Small_Transpose, AnAxisOfExtentOneKeepsTheElementsInOrder)
{
    TracedTranspose traced;
    trace_transpose(traced, Shape{2, 1, 4}, {1, 0});
    ASSERT_TRUE(traced.status.is_ok()) << traced.status.message();

    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<double> got,
        SaTest::evaluate_on_host(traced.graph, {{traced.input, counted(8)}}, traced.output));

    EXPECT_EQ(got, counted(8));
    EXPECT_EQ(traced.graph.value(traced.output).shape, (Shape{2, 4, 1}));
}

/** @brief Tests that the identity permutation emits no node and returns its input. */
TEST(Small_Transpose, TheIdentityPermutationEmitsNothing)
{
    TracedTranspose traced;
    trace_transpose(traced, Shape{2, 3, 4}, {0, 1});

    EXPECT_EQ(traced.output, traced.input);
    for (const Graph::Node& node : traced.graph.nodes())
        EXPECT_NE(node.op, Ops::K_TRANSPOSE.id);
    SA_EXPECT_OK(traced.status);
}

/** @brief Tests that a permutation is one K_TRANSPOSE node holding it, and no product. */
TEST(Small_Transpose, APermutationIsOneTransposeNodeHoldingIt)
{
    TracedTranspose traced;
    trace_transpose(traced, Shape{2, 2, 3, 2}, {1, 0, 2});
    ASSERT_TRUE(traced.status.is_ok()) << traced.status.message();

    std::size_t transposes = 0;
    std::size_t products = 0;
    for (const Graph::Node& node : traced.graph.nodes())
    {
        if (node.op == Ops::K_TRANSPOSE.id)
        {
            ++transposes;
            EXPECT_EQ(node.attributes.get(0), 1.0);
            EXPECT_EQ(node.attributes.get(1), 0.0);
            EXPECT_EQ(node.attributes.get(2), 2.0);
        }

        if (node.op == Ops::K_MATMUL.id)
            ++products;
    }
    EXPECT_EQ(transposes, 1u);
    EXPECT_EQ(products, 0u);
}

/** @brief Tests that a half-precision input is traced as one K_TRANSPOSE node. */
TEST(Small_Transpose, AHalfPrecisionInputIsTraced)
{
    TracedTranspose traced;
    trace_transpose(traced, Shape{2, 3, 4}, {1, 0}, DType::Half);

    SA_EXPECT_OK(traced.status);
    EXPECT_EQ(traced.graph.value(traced.output).dtype, DType::Half);
    EXPECT_EQ(traced.graph.value(traced.output).shape, (Shape{2, 4, 3}));
}

/** @brief Tests that a permutation of the wrong length records the refusal and keeps the input. */
TEST(Small_Transpose, AWrongLengthPermutationFailsTheTrace)
{
    TracedTranspose traced;
    trace_transpose(traced, Shape{2, 3, 4}, {0, 1, 2});

    EXPECT_EQ(traced.output, traced.input);
    SA_EXPECT_FAILURE(traced.status, StatusCode::InvalidDescription, "no permutation");
}

namespace
{
    /** @brief Provides the Engine every gradient case records on. */
    class Medium_TransposeGradient : public SaTest::EngineTest
    {

    };

    /** @brief Holds the elements of the input of the gradient case: [2, 2, 3, 2]. */
    constexpr std::size_t K_GRADIENT_ELEMENTS = 24;

    /** @brief Returns a point with no two elements equal. */
    std::vector<double> gradient_point()
    {
        std::vector<double> point(K_GRADIENT_ELEMENTS);
        for (std::size_t index = 0; index < point.size(); ++index)
            point[index] = 0.37 * static_cast<double>(index) - 3.1 +
                           0.05 * static_cast<double>((index * 7) % 5);

        return point;
    }
} // namespace

/** @brief Tests that the gradient of a swap before a trailing axis matches central differences. */
TEST_F(Medium_TransposeGradient, TheGradientOfTheSwapMatchesCentralDifferences)
{
    SaTest::GradientProbe probe(backend(), Shape{2, 2, 3, 2},
                                [](const Graph::GraphBuilder& builder, Graph::ValueID x)
    {
        Trace trace = Trace::evaluation(builder);
        Transpose layer{{1, 0, 2}};
        const Graph::ValueID swapped = layer.forward(trace, x);
        return builder.reshape(swapped, Shape{2, 12});
    });

    std::vector<double> point = gradient_point();
    SaTest::expect_gradient_matches(probe, point);
}
