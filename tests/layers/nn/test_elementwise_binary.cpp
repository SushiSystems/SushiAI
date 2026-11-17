/**************************************************************************/
/* test_elementwise_binary.cpp                                            */
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
 * @file test_elementwise_binary.cpp
 * @brief Tests the mul, maximum, sub and div rows: the shape rule, the nodes, the values and the gradients.
 * @author Mustafa Garip
 */

#include <cstddef>
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
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/elementwise_binary.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/gradient_probe.hpp>
#include <support/host_row_values.hpp>
#include <support/test_names.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using NN::BinaryOperation;
using NN::ElementwiseBinary;
using NN::Trace;
namespace SaTest = SushiAI::Test;

namespace
{
    using SaTest::require_value;

    /** @brief Holds one operation with the node it emits over equal shapes and over a prefix. */
    struct OperationRow
    {
        std::string name;        /**< The row's name. */
        BinaryOperation operation; /**< The operation under test. */
        std::string row_type;    /**< The type a model file spells. */
        OpID equal_op;      /**< The node over two inputs of one shape. */
        OpID broadcast_op;  /**< The node over a whole input and a prefix; 0 for none. */
    };

    /** @brief Lists the four operations. */
    std::vector<OperationRow> operation_rows()
    {
        return {
            {"Mul", BinaryOperation::Mul, "mul", Ops::K_MUL.id, Ops::K_BROADCAST_MUL.id},
            {"Maximum", BinaryOperation::Maximum, "maximum", Ops::K_MAXIMUM.id, OpID{}},
            {"Sub", BinaryOperation::Sub, "sub", Ops::K_SUB.id, Ops::K_BROADCAST_SUB.id},
            {"Div", BinaryOperation::Div, "div", Ops::K_DIV.id, Ops::K_BROADCAST_DIV.id},
        };
    }

    /** @brief Builds a node of @p operation over @p shapes. */
    Result<std::unique_ptr<ElementwiseBinary>> create_over(BinaryOperation operation,
                                                           const std::vector<Shape>& shapes)
    {
        return ElementwiseBinary::create(operation, NN::ShapeSpan(shapes.data(), shapes.size()));
    }

    /** @brief Views @p ids as the inputs of one forward call. */
    NN::ValueSpan span_of(const std::vector<Graph::ValueID>& ids)
    {
        return NN::ValueSpan(ids.data(), ids.size());
    }

    /** @brief Traces @p node over @p inputs and returns the values it produced. */
    Result<std::vector<Graph::ValueID>> trace_node(ElementwiseBinary& node,
                                                   const Graph::GraphBuilder& builder,
                                                   const std::vector<Graph::ValueID>& inputs)
    {
        std::vector<Graph::ValueID> outputs;
        SA_TRY(node.forward(Trace::evaluation(builder), span_of(inputs), outputs));

        return outputs;
    }

    /** @brief Names a row of OperationRow for gtest. */
    std::string operation_name(const ::testing::TestParamInfo<OperationRow>& info)
    {
        return info.param.name;
    }

    /** @brief Runs each operation row. */
    class Small_ElementwiseBinary : public ::testing::TestWithParam<OperationRow>
    {

    };

    /** @brief Runs the broadcasting operations: every row but the maximum. */
    class Small_ElementwiseBinaryBroadcast : public ::testing::TestWithParam<OperationRow>
    {

    };

    /** @brief Runs the operations that have a broadcast form and do not commute. */
    class Small_ElementwiseBinaryNoncommuting : public ::testing::TestWithParam<OperationRow>
    {

    };

    /** @brief Lists sub and div, the rows that have a broadcast form and do not commute. */
    std::vector<OperationRow> noncommuting_rows()
    {
        std::vector<OperationRow> rows;
        for (const OperationRow& row : operation_rows())
        {
            if (row.operation == BinaryOperation::Sub || row.operation == BinaryOperation::Div)
                rows.push_back(row);
        }

        return rows;
    }

    /** @brief Lists the rows that have a broadcast form. */
    std::vector<OperationRow> broadcasting_rows()
    {
        std::vector<OperationRow> rows;
        for (const OperationRow& row : operation_rows())
        {
            if (row.operation != BinaryOperation::Maximum)
                rows.push_back(row);
        }

        return rows;
    }
} // namespace

/** @brief Tests that two equal shapes give one port of that shape. */
TEST_P(Small_ElementwiseBinary, CreateKeepsTheSharedShapeAsThePort)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ElementwiseBinary> node,
                            create_over(GetParam().operation, {Shape{8, 6, 3}, Shape{8, 6, 3}}));

    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].name, "output");
    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 3}));
    EXPECT_EQ(node->input_count(), 2u);
}

/** @brief Tests that an input count other than two is refused and the count is named. */
TEST_P(Small_ElementwiseBinary, CreateRejectsAnInputCountOtherThanTwo)
{
    const OperationRow& row = GetParam();

    SA_EXPECT_FAILURE(create_over(row.operation, {Shape{4}}), StatusCode::InvalidDescription,
                      row.row_type, "takes 2 inputs", "was given 1");
    SA_EXPECT_FAILURE(create_over(row.operation, {Shape{4}, Shape{4}, Shape{4}}),
                      StatusCode::InvalidDescription, "was given 3");
}

/** @brief Tests that an input with no per-sample axis is refused and named by its index. */
TEST_P(Small_ElementwiseBinary, CreateRejectsAnInputOfRankZero)
{
    SA_EXPECT_FAILURE(create_over(GetParam().operation, {Shape{4}, Shape{}}),
                      StatusCode::InvalidDescription, "input 1", "rank 1 or more");
}

/** @brief Tests that two shapes with no prefix relation are refused and both are named. */
TEST_P(Small_ElementwiseBinary, CreateRejectsShapesThatShareNoPrefix)
{
    SA_EXPECT_FAILURE(create_over(GetParam().operation, {Shape{8, 6, 3}, Shape{6, 3}}),
                      StatusCode::InvalidDescription, "[8, 6, 3]", "[6, 3]", "leading prefix");
}

/** @brief Tests that emitting over equal shapes gives one node of the operation, inputs in order. */
TEST_P(Small_ElementwiseBinary, ForwardEmitsOneNodeReadingBothInputsInOrder)
{
    const OperationRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 8, 6, 3}, "a");
    const Graph::ValueID b = builder.input(Shape{2, 8, 6, 3}, "b");
    const std::size_t nodes_before = graph.node_count();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ElementwiseBinary> node,
                            create_over(row.operation, {Shape{8, 6, 3}, Shape{8, 6, 3}}));
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({b, a}), outputs));

    ASSERT_EQ(graph.node_count(), nodes_before + 1);
    const Graph::Node& emitted = graph.nodes().back();
    EXPECT_EQ(emitted.op, row.equal_op);
    EXPECT_EQ(emitted.inputs, (std::vector<Graph::ValueID>{b, a}));
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(emitted.outputs.front(), outputs[0]);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{2, 8, 6, 3}));
}

/** @brief Tests that forward refuses an input count it was not built over and emits nothing. */
TEST_P(Small_ElementwiseBinary, ForwardRejectsAnInputCountItWasNotBuiltOver)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 4}, "a");
    const std::size_t nodes_before = graph.node_count();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ElementwiseBinary> node,
                            create_over(GetParam().operation, {Shape{4}, Shape{4}}));
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_FAILURE(node->forward(Trace::evaluation(builder), span_of({a}), outputs),
                      StatusCode::InvalidDescription, "input count is 2", "given is 1");
    EXPECT_TRUE(outputs.empty());
    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that each factory builds its node from a row with no key and refuses a key. */
TEST_P(Small_ElementwiseBinary, TheFactoryTakesNoKey)
{
    const OperationRow& row = GetParam();
    const Shape shapes[] = {Shape{4, 3}, Shape{4, 3}};
    const Config::Value none = require_value(Config::parse_json("{}", "<test>"));
    const Config::Value stray = require_value(Config::parse_json(R"({"alpha": 2})", "<test>"));
    const std::string context = "layer 3 '" + row.row_type + "'";

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        NN::make_binary_node(row.operation, NN::ShapeSpan(shapes, 2), none, context));
    EXPECT_NE(dynamic_cast<const ElementwiseBinary*>(node.get()), nullptr);

    SA_EXPECT_FAILURE(NN::make_binary_node(row.operation, NN::ShapeSpan(shapes, 2), stray, context),
                      StatusCode::InvalidDescription, context, "'alpha'", "no key");
}

/** @brief Tests that the row type of each operation is the one a model file spells. */
TEST_P(Small_ElementwiseBinary, TheRowNameIsTheModelFileType)
{
    EXPECT_EQ(NN::binary_row_name(GetParam().operation), GetParam().row_type);
}

INSTANTIATE_TEST_SUITE_P(Operations, Small_ElementwiseBinary, ::testing::ValuesIn(operation_rows()),
                         operation_name);

/** @brief Tests that a second input that is a leading prefix is broadcast, and the port is the first. */
TEST_P(Small_ElementwiseBinaryBroadcast, TheSecondInputMayBeALeadingPrefix)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ElementwiseBinary> node,
                            create_over(GetParam().operation, {Shape{8, 6, 3}, Shape{8, 6}}));

    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 3}));
}

/** @brief Tests that forward over a prefix emits the broadcast node with the whole input first. */
TEST_P(Small_ElementwiseBinaryBroadcast, ForwardEmitsOneLeadingBroadcastNode)
{
    const OperationRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID whole = builder.input(Shape{2, 8, 6, 3}, "whole");
    const Graph::ValueID vector = builder.input(Shape{2, 8}, "vector");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ElementwiseBinary> node,
                            create_over(row.operation, {Shape{8, 6, 3}, Shape{8}}));
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({whole, vector}), outputs));

    const Graph::Node& emitted = graph.nodes().back();
    EXPECT_EQ(emitted.op, row.broadcast_op);
    EXPECT_EQ(emitted.inputs, (std::vector<Graph::ValueID>{whole, vector}));
    EXPECT_EQ(Graph::broadcast_side(emitted.attributes), Graph::BroadcastSide::Leading);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{2, 8, 6, 3}));
}

/** @brief Tests that a first input that is a prefix is refused when the operation does not commute. */
TEST_P(Small_ElementwiseBinaryNoncommuting, ThePrefixFirstIsRefused)
{
    SA_EXPECT_FAILURE(create_over(GetParam().operation, {Shape{8}, Shape{8, 6, 3}}),
                      StatusCode::InvalidDescription, "leading prefix");
}

/** @brief Tests that a product accepts the prefix input first, and the port is the whole input. */
TEST(Small_ElementwiseBinaryMul, ThePrefixMayComeFirst)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ElementwiseBinary> node,
                            create_over(BinaryOperation::Mul, {Shape{8}, Shape{8, 6, 3}}));

    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 6, 3}));
}

INSTANTIATE_TEST_SUITE_P(Operations, Small_ElementwiseBinaryNoncommuting,
                         ::testing::ValuesIn(noncommuting_rows()), operation_name);

INSTANTIATE_TEST_SUITE_P(Operations, Small_ElementwiseBinaryBroadcast,
                         ::testing::ValuesIn(broadcasting_rows()), operation_name);

/** @brief Tests that a product with the prefix input first swaps the operands of the node. */
TEST(Small_ElementwiseBinaryMul, APrefixFirstIsEmittedWithTheWholeInputFirst)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID vector = builder.input(Shape{2, 8}, "vector");
    const Graph::ValueID whole = builder.input(Shape{2, 8, 6, 3}, "whole");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ElementwiseBinary> node,
                            create_over(BinaryOperation::Mul, {Shape{8}, Shape{8, 6, 3}}));
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), span_of({vector, whole}), outputs));

    const Graph::Node& emitted = graph.nodes().back();
    EXPECT_EQ(emitted.op, Ops::K_BROADCAST_MUL.id);
    EXPECT_EQ(emitted.inputs, (std::vector<Graph::ValueID>{whole, vector}));
}

/** @brief Tests that a maximum over a prefix is refused, since the graph has no broadcast maximum. */
TEST(Small_ElementwiseBinaryMaximum, AShorterSecondInputIsRefused)
{
    SA_EXPECT_FAILURE(create_over(BinaryOperation::Maximum, {Shape{8, 6, 3}, Shape{8}}),
                      StatusCode::InvalidDescription, "maximum takes equal shapes only");
}

/** @brief Tests that the state walk of a binary node visits no parameter and no buffer. */
TEST(Small_ElementwiseBinaryMul, TheStateWalkVisitsNothing)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ElementwiseBinary> node,
                            create_over(BinaryOperation::Mul, {Shape{4}, Shape{4}}));

    std::size_t visits = 0;
    const auto on_parameter = [&visits](const std::string&, NN::Parameter&)
    {
        ++visits;
    };
    const auto on_buffer = [&visits](const std::string&, NN::Buffer&)
    {
        ++visits;
    };
    const NN::StateVisitor visitor(on_parameter, on_buffer);
    node->walk_state("product", visitor);

    EXPECT_EQ(visits, 0u);
}

namespace
{
    /** @brief Holds one operation with the values it gives on the hand-worked inputs. */
    struct ValueRow
    {
        std::string name;             /**< The row's name. */
        BinaryOperation operation;    /**< The operation under test. */
        std::vector<double> equal;    /**< The result over two inputs of one shape. */
        std::vector<double> broadcast; /**< The result over a whole input and a prefix. */
    };

    /**
     * @brief Lists the hand-worked results.
     *
     * Equal shapes: a = {7, -2, 5, 8, 3, -6} and b = {2, 4, -5, 8, -1, 3} over [2, 3]. Prefix: a =
     * {1..8} over [2, 2, 2] and v = {2, 4, 1, -2} over [2, 2], so element [n, i, j] meets v[n, i].
     * Every quotient and product is a dyadic rational, so double holds each exactly.
     */
    std::vector<ValueRow> value_rows()
    {
        return {
            {"Mul", BinaryOperation::Mul, {14, -8, -25, 64, -3, -18},
             {2, 4, 12, 16, 5, 6, -14, -16}},
            {"Maximum", BinaryOperation::Maximum, {7, 4, 5, 8, 3, 3}, {}},
            {"Sub", BinaryOperation::Sub, {5, -6, 10, 0, 4, -9}, {-1, 0, -1, 0, 4, 5, 9, 10}},
            {"Div", BinaryOperation::Div, {3.5, -0.5, -1, 1, -3, -2},
             {0.5, 1, 0.75, 1, 5, 6, -3.5, -4}},
        };
    }

    /** @brief Runs each hand-worked row over inputs of one shape on the host. */
    class Small_ElementwiseBinaryValues : public ::testing::TestWithParam<ValueRow>
    {

    };

    /** @brief Runs each hand-worked row over a whole input and a prefix on the host. */
    class Small_ElementwiseBinaryPrefixValues : public ::testing::TestWithParam<ValueRow>
    {

    };

    /** @brief Lists the rows that have a result over a prefix. */
    std::vector<ValueRow> prefix_value_rows()
    {
        std::vector<ValueRow> rows;
        for (const ValueRow& row : value_rows())
        {
            if (!row.broadcast.empty())
                rows.push_back(row);
        }

        return rows;
    }

    /** @brief Names a ValueRow for gtest. */
    std::string value_name(const ::testing::TestParamInfo<ValueRow>& info)
    {
        return info.param.name;
    }

    /** @brief Expects every element of @p got to equal @p expected exactly. */
    void expect_exact(const std::vector<double>& got, const std::vector<double>& expected)
    {
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t index = 0; index < expected.size(); ++index)
            EXPECT_TRUE(TestSupport::is_within(
                TestSupport::measure_error(got[index], expected[index], 0.0)))
                << "element " << index;
    }
} // namespace

/** @brief Tests the values of an operation over two inputs of one shape against the hand results. */
TEST_P(Small_ElementwiseBinaryValues, TheValuesOverEqualShapesMatchTheHandResults)
{
    const ValueRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 3}, "a", DType::Float64);
    const Graph::ValueID b = builder.input(Shape{2, 3}, "b", DType::Float64);

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ElementwiseBinary> node,
                            create_over(row.operation, {Shape{3}, Shape{3}}));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Graph::ValueID> outputs,
                            trace_node(*node, builder, {a, b}));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<double> got,
        SaTest::evaluate_on_host(graph,
                                 {{a, {7, -2, 5, 8, 3, -6}}, {b, {2, 4, -5, 8, -1, 3}}},
                                 NN::HostTableList(), outputs[0]));

    expect_exact(got, row.equal);
}

/** @brief Tests the values of an operation over a whole input and a prefix against the hand results. */
TEST_P(Small_ElementwiseBinaryPrefixValues, TheValuesOverAPrefixMatchTheHandResults)
{
    const ValueRow& row = GetParam();

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID a = builder.input(Shape{2, 2, 2}, "a", DType::Float64);
    const Graph::ValueID v = builder.input(Shape{2, 2}, "v", DType::Float64);

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ElementwiseBinary> node,
                            create_over(row.operation, {Shape{2, 2}, Shape{2}}));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Graph::ValueID> outputs,
                            trace_node(*node, builder, {a, v}));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<double> got,
        SaTest::evaluate_on_host(graph, {{a, {1, 2, 3, 4, 5, 6, 7, 8}}, {v, {2, 4, 1, -2}}},
                                 NN::HostTableList(), outputs[0]));

    expect_exact(got, row.broadcast);
}

INSTANTIATE_TEST_SUITE_P(Operations, Small_ElementwiseBinaryValues,
                         ::testing::ValuesIn(value_rows()), value_name);

INSTANTIATE_TEST_SUITE_P(Operations, Small_ElementwiseBinaryPrefixValues,
                         ::testing::ValuesIn(prefix_value_rows()), value_name);

namespace
{
    /** @brief Provides the Engine every gradient case records on. */
    class Medium_ElementwiseBinaryGradient : public SaTest::EngineTest,
                                             public ::testing::WithParamInterface<OperationRow>
    {

    };

    /** @brief Returns the second operand, away from zero for a quotient and from the first for a maximum. */
    std::vector<double> second_operand()
    {
        return {2.0, 4.0, -5.0, 8.0, -1.5, 3.0};
    }

    /** @brief Returns the point of the first operand, 0.3 or more from the second operand. */
    std::vector<double> first_point()
    {
        return {7.3, -2.2, 5.1, 8.4, 3.2, -6.1};
    }

    /**
     * @brief Traces @p operation over the [2, 3] input @p x and the leaf @p other, and returns the
     *        value.
     */
    Graph::ValueID trace_operation(const Graph::GraphBuilder& builder, BinaryOperation operation,
                                   Graph::ValueID first, Graph::ValueID second)
    {
        const Shape shapes[] = {Shape{3}, Shape{3}};
        const std::unique_ptr<ElementwiseBinary> node =
            ElementwiseBinary::create(operation, NN::ShapeSpan(shapes, 2)).value();

        return trace_node(*node, builder, {first, second}).value().at(0);
    }
} // namespace

/** @brief Tests the gradient of the first input against central differences, in FLOAT64. */
TEST_P(Medium_ElementwiseBinaryGradient, TheFirstInputGradientMatchesCentralDifferences)
{
    const BinaryOperation operation = GetParam().operation;
    Graph::ValueID other = Graph::K_NO_VALUE;

    SaTest::GradientProbe probe(backend(), Shape{2, 3},
                                [&](const Graph::GraphBuilder& builder, Graph::ValueID x)
    {
        other = builder.parameter(Shape{2, 3}, "other", DType::Float64);
        return trace_operation(builder, operation, x, other);
    });
    probe.fill(other, second_operand());

    std::vector<double> point = first_point();
    SaTest::expect_gradient_matches(probe, point);
}

/** @brief Tests the gradient of the second input against central differences, in FLOAT64. */
TEST_P(Medium_ElementwiseBinaryGradient, TheSecondInputGradientMatchesCentralDifferences)
{
    const BinaryOperation operation = GetParam().operation;
    Graph::ValueID first = Graph::K_NO_VALUE;

    SaTest::GradientProbe probe(backend(), Shape{2, 3},
                                [&](const Graph::GraphBuilder& builder, Graph::ValueID x)
    {
        first = builder.parameter(Shape{2, 3}, "first", DType::Float64);
        return trace_operation(builder, operation, first, x);
    });
    probe.fill(first, first_point());

    std::vector<double> point = second_operand();
    SaTest::expect_gradient_matches(probe, point);
}

INSTANTIATE_TEST_SUITE_P(Operations, Medium_ElementwiseBinaryGradient,
                         ::testing::ValuesIn(operation_rows()), operation_name);
