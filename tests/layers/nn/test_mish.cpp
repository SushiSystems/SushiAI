/**************************************************************************/
/* test_mish.cpp                                                          */
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
 * @file test_mish.cpp
 * @brief Tests the Mish layer: its trace and its values on a device.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/mish.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
using NN::Mish;
// SaTest stays reachable inside a TEST_F body; see tests/layers/nn/README.md.
namespace SaTest = SushiAI::Test;
using SaTest::read_tensor;
using SaTest::write_tensor;

static_assert(NN::Module<Mish>, "Mish must satisfy the Module contract");
static_assert(!NN::ModuleContainer<Mish>, "Mish is a leaf, not a container");
static_assert(std::tuple_size<decltype(Mish::fields())>::value == 0,
              "Mish must declare no parameter field");

/** @brief Tests that tracing emits SOFTPLUS, TANH and MUL in order over the input. */
TEST(Medium_Mish, ForwardEmitsSoftplusTanhThenMulReadingTheInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    const std::size_t nodes_before = graph.node_count();

    const Mish layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    ASSERT_EQ(graph.node_count(), nodes_before + 3);
    const std::vector<Graph::Node>& nodes = graph.nodes();
    const Graph::Node& softplus_node = nodes[nodes_before];
    const Graph::Node& tanh_node = nodes[nodes_before + 1];
    const Graph::Node& product_node = nodes[nodes_before + 2];

    EXPECT_EQ(softplus_node.op, Ops::K_SOFTPLUS.id);
    ASSERT_EQ(softplus_node.inputs.size(), 1u);
    EXPECT_EQ(softplus_node.inputs.front(), x);
    ASSERT_EQ(softplus_node.outputs.size(), 1u);

    EXPECT_EQ(tanh_node.op, Ops::K_TANH.id);
    ASSERT_EQ(tanh_node.inputs.size(), 1u);
    EXPECT_EQ(tanh_node.inputs.front(), softplus_node.outputs.front());
    ASSERT_EQ(tanh_node.outputs.size(), 1u);

    EXPECT_EQ(product_node.op, Ops::K_MUL.id);
    ASSERT_EQ(product_node.inputs.size(), 2u);
    EXPECT_EQ(product_node.inputs[0], x);
    EXPECT_EQ(product_node.inputs[1], tanh_node.outputs.front());
    ASSERT_EQ(product_node.outputs.size(), 1u);
    EXPECT_EQ(product_node.outputs.front(), y);
}

/** @brief Tests that a feature-vector batch keeps its shape and element type. */
TEST(Medium_Mish, ForwardKeepsTheShapeAndTypeOfAFeatureBatch)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    const Mish layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4}));
    EXPECT_EQ(graph.value(y).dtype, graph.value(x).dtype);
}

/** @brief Tests that an image batch keeps its shape and carries the NHWC layout tag. */
TEST(Medium_Mish, ForwardKeepsTheShapeAndNHWCTagOfAnImageBatch)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");

    const Mish layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 8, 6, 3}));
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

/** @brief Tests that the result takes the layout of its input when that is NCHW. */
TEST(Medium_Mish, ForwardCarriesTheLayoutOfItsInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID nchw,
                            builder.layout_convert(x, Graph::TensorLayout::NCHW));

    const Mish layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, nchw));

    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NCHW);
    EXPECT_EQ(graph.value(y).shape, graph.value(nchw).shape);
}

/** @brief Tests that output_shape() returns its input for a feature vector and for an image. */
TEST(Medium_Mish, OutputShapeReturnsItsInput)
{
    const Mish layer{};

    EXPECT_EQ(layer.output_shape(Shape{7}).value(), (Shape{7}));
    EXPECT_EQ(layer.output_shape(Shape{8, 6, 4}).value(), (Shape{8, 6, 4}));
}

/** @brief Tests that the parameter walk visits nothing after a trace. */
TEST(Medium_Mish, TheParameterWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    Mish layer{};
    SA_EXPECT_OK(layer.forward(builder, x));

    std::size_t visits = 0;
    NN::for_each_parameter(layer, [&visits](const std::string&, NN::Parameter&)
    {
        ++visits;
    });
    EXPECT_EQ(visits, 0u);
}

namespace
{
    /** @brief Provides the Engine every Mish case records on. */
    class Medium_MishValues : public SaTest::EngineTest
    {
    };

    /** @brief Bounds an engine result by one part in a million of the exact value's magnitude. */
    constexpr double K_RELATIVE_TOLERANCE = 1e-6;

    /** @brief Holds the sweep's half range: the sweep runs from -30 to 30. */
    constexpr double K_SWEEP_HALF_RANGE = 30.0;

    /** @brief Holds the sweep's spacing; 0.5 is exact in FLOAT32. */
    constexpr double K_SWEEP_STEP = 0.5;

    /** @brief Holds the magnitude past which Darknet's softplus leaves log(exp(x) + 1). */
    constexpr double K_DARKNET_SOFTPLUS_CUTOFF = 20.0;

    /** @brief Bounds the largest absolute difference between Darknet's form and the exact form. */
    constexpr double K_DARKNET_DIFFERENCE_BOUND = 1e-12;

    /** @brief Computes x * tanh(log1p(exp(x))) in double; the exact value the engine is held to. */
    double exact_mish(double x)
    {
        return x * std::tanh(std::log1p(std::exp(x)));
    }

    /**
     * @brief Computes Darknet's softplus in double: x above the cut-off, exp(x) below its
     * negative, and log(exp(x) + 1) between.
     */
    double darknet_softplus(double x)
    {
        if (x > K_DARKNET_SOFTPLUS_CUTOFF)
            return x;
        if (x < -K_DARKNET_SOFTPLUS_CUTOFF)
            return std::exp(x);
        return std::log(std::exp(x) + 1.0);
    }

    /** @brief Computes Darknet's tanh in double as 2 / (1 + exp(-2 v)) - 1. */
    double darknet_tanh(double v)
    {
        return 2.0 / (1.0 + std::exp(-2.0 * v)) - 1.0;
    }

    /** @brief Computes Darknet's mish in double, x * tanh(softplus(x)) with the cut-off. */
    double darknet_mish(double x)
    {
        return x * darknet_tanh(darknet_softplus(x));
    }

    /**
     * @brief Compares every element of @p got with @p expected to K_RELATIVE_TOLERANCE.
     * @pre @p expected holds exact values; a zero expected element must be matched exactly.
     */
    void expect_relative(const std::vector<double>& got, const std::vector<double>& expected)
    {
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            const double bound = K_RELATIVE_TOLERANCE * std::abs(expected[i]);
            EXPECT_NEAR(got[i], expected[i], bound) << "at element " << i;
        }
    }

    /**
     * @brief Runs the Mish layer over one FLOAT32 row on the engine and reads the result.
     * @param input The row's elements; the row is as long as this vector.
     * @param result Receives the row's elements after the layer.
     */
    void run_mish(SushiBLAS::Engine& engine, const std::vector<double>& input,
                  std::vector<double>& result)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const std::int64_t columns = static_cast<std::int64_t>(input.size());
        const Graph::ValueID x = builder.input(Shape{1, columns}, "x");
        const NN::Mish layer{};
        SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

        Graph::TensorPool pool(engine);
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        write_tensor(pool.tensor(x), input);
        std::unique_ptr<Graph::CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, Graph::CompiledStep::compile(engine, graph, pool));
        SA_EXPECT_OK(step->run());
        result = read_tensor(pool.tensor(y));
    }
} // namespace

/** @brief Tests that Mish matches the exact form at -20, -1, 0, 1 and 20. */
TEST_F(Medium_MishValues, MatchesTheExactFormAtTheAcceptancePoints)
{
    const std::vector<double> input = {-20.0, -1.0, 0.0, 1.0, 20.0};

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_mish(*engine, input, result));

    std::vector<double> expected(input.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = exact_mish(input[i]);

    ASSERT_NO_FATAL_FAILURE(expect_relative(result, expected));
}

/**
 * @brief Tests Mish from -30 to 30 against the exact form, and records how far Darknet's
 * cut-off form lies from the exact form.
 */
TEST_F(Medium_MishValues, SweepMatchesTheExactFormAndStatesTheDarknetDifference)
{
    const std::size_t count =
        static_cast<std::size_t>(2.0 * K_SWEEP_HALF_RANGE / K_SWEEP_STEP) + 1;
    std::vector<double> input(count);
    for (std::size_t i = 0; i < count; ++i)
        input[i] = -K_SWEEP_HALF_RANGE + static_cast<double>(i) * K_SWEEP_STEP;

    std::vector<double> result;
    ASSERT_NO_FATAL_FAILURE(run_mish(*engine, input, result));

    std::vector<double> expected(count);
    double largest_difference = 0.0;
    double largest_at = input.front();
    for (std::size_t i = 0; i < count; ++i)
    {
        expected[i] = exact_mish(input[i]);
        const double difference = std::abs(darknet_mish(input[i]) - expected[i]);
        if (difference > largest_difference)
        {
            largest_difference = difference;
            largest_at = input[i];
        }
    }

    ASSERT_NO_FATAL_FAILURE(expect_relative(result, expected));

    std::ostringstream stated;
    stated << std::scientific << largest_difference;
    RecordProperty("darknet_cut_off_largest_difference", stated.str());
    EXPECT_LE(largest_difference, K_DARKNET_DIFFERENCE_BOUND) << "at x = " << largest_at;
}
