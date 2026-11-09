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
 * @file test_mish.cpp
 * @brief Tests the Mish layer: its trace and its values on a device.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
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
#include <support/tolerance.hpp>

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

/** @brief Tests that a feature-vector batch keeps its shape and element type. */
TEST(Small_Mish, ForwardKeepsTheShapeAndTypeOfAFeatureBatch)
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
TEST(Small_Mish, ForwardKeepsTheShapeAndNHWCTagOfAnImageBatch)
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
TEST(Small_Mish, ForwardCarriesTheLayoutOfItsInput)
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
TEST(Small_Mish, OutputShapeReturnsItsInput)
{
    const Mish layer{};

    EXPECT_EQ(layer.output_shape(Shape{7}).value(), (Shape{7}));
    EXPECT_EQ(layer.output_shape(Shape{8, 6, 4}).value(), (Shape{8, 6, 4}));
}

/** @brief Tests that the parameter walk visits nothing after a trace. */
TEST(Small_Mish, TheParameterWalkVisitsNothing)
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

    /**
     * @brief Bounds the relative error of an engine FLOAT32 result against the exact value; 32 unit
     * roundoffs of float, the smallest power of two that covers one part in a million.
     */
    constexpr double K_RELATIVE_TOLERANCE = 32.0 * TestSupport::unit_roundoff<float>();

    /** @brief Holds the sweep's half range: the sweep runs from -30 to 30. */
    constexpr double K_SWEEP_HALF_RANGE = 30.0;

    /** @brief Holds the sweep's spacing; 0.5 is exact in FLOAT32. */
    constexpr double K_SWEEP_STEP = 0.5;

    /** @brief Computes x * tanh(log1p(exp(x))) in double; the exact value the engine is held to. */
    double exact_mish(double x)
    {
        return x * std::tanh(std::log1p(std::exp(x)));
    }

    /** @brief Returns the sweep's inputs, from -K_SWEEP_HALF_RANGE to K_SWEEP_HALF_RANGE. */
    std::vector<double> sweep_input()
    {
        const std::size_t count =
            static_cast<std::size_t>(2.0 * K_SWEEP_HALF_RANGE / K_SWEEP_STEP) + 1;
        std::vector<double> input(count);
        std::generate(input.begin(), input.end(),
                      [index = std::size_t{0}]() mutable
                      {
                          return -K_SWEEP_HALF_RANGE + static_cast<double>(index++) * K_SWEEP_STEP;
                      });

        return input;
    }

    /** @brief Returns exact_mish of each element of @p input. */
    std::vector<double> exact_values(const std::vector<double>& input)
    {
        std::vector<double> values(input.size());
        std::transform(input.begin(), input.end(), values.begin(), exact_mish);

        return values;
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
    void run_mish(SushiAI::ExecutionBackend& backend, const std::vector<double>& input,
                  std::vector<double>& result)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const std::int64_t columns = static_cast<std::int64_t>(input.size());
        const Graph::ValueID x = builder.input(Shape{1, columns}, "x");
        const NN::Mish layer{};
        SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

        Graph::TensorPool pool(backend.engine());
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        write_tensor(pool.tensor(x), input);
        std::unique_ptr<Graph::CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, Graph::CompiledStep::compile(backend, graph, pool));
        SA_EXPECT_OK(step->run());
        result = read_tensor(pool.tensor(y));
    }
} // namespace

/** @brief Tests that Mish matches the exact form from -30 to 30. */
TEST_F(Medium_MishValues, SweepMatchesTheExactForm)
{
    const std::vector<double> input = sweep_input();
    std::vector<double> result;

    ASSERT_NO_FATAL_FAILURE(run_mish(backend(), input, result));

    ASSERT_NO_FATAL_FAILURE(expect_relative(result, exact_values(input)));
}
