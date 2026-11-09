/**************************************************************************/
/* test_silu.cpp                                                          */
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
 * @file test_silu.cpp
 * @brief Tests the SiLU layer's trace, shape inference and empty state.
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
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/silu.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using NN::SiLU;

// SaTest stays reachable inside a TEST_F body; see tests/layers/nn/README.md.
namespace SaTest = SushiAI::Test;
using SaTest::read_tensor;
using SaTest::write_tensor;

static_assert(NN::Module<SiLU>, "SiLU must satisfy the Module contract");
static_assert(!NN::ModuleContainer<SiLU>, "SiLU is a leaf, not a container");
static_assert(std::tuple_size<decltype(SiLU::fields())>::value == 0,
              "SiLU must declare no parameter field");

/** @brief Tests that a feature-vector batch keeps its shape and element type. */
TEST(Small_SiLU, ForwardKeepsTheShapeAndTypeOfAFeatureBatch)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    const SiLU layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4}));
    EXPECT_EQ(graph.value(y).dtype, graph.value(x).dtype);
}

/** @brief Tests that an image batch keeps its shape and carries the NHWC layout tag. */
TEST(Small_SiLU, ForwardKeepsTheShapeAndNHWCTagOfAnImageBatch)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");

    const SiLU layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 8, 6, 3}));
    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NHWC);
}

/** @brief Tests that the result takes the layout of its input when that is NCHW. */
TEST(Small_SiLU, ForwardCarriesTheLayoutOfItsInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 6, 3}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID nchw,
                            builder.layout_convert(x, Graph::TensorLayout::NCHW));

    const SiLU layer{};
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID y, layer.forward(builder, nchw));

    EXPECT_EQ(graph.value(y).layout, Graph::TensorLayout::NCHW);
    EXPECT_EQ(graph.value(y).shape, graph.value(nchw).shape);
}

/** @brief Tests that output_shape() returns its input for a feature vector and for an image. */
TEST(Small_SiLU, OutputShapeReturnsItsInput)
{
    const SiLU layer{};

    EXPECT_EQ(layer.output_shape(Shape{7}).value(), (Shape{7}));
    EXPECT_EQ(layer.output_shape(Shape{8, 6, 4}).value(), (Shape{8, 6, 4}));
}

/** @brief Tests that the parameter walk visits nothing after a trace. */
TEST(Small_SiLU, TheParameterWalkVisitsNothing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    SiLU layer{};
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
    /** @brief Provides the Engine every SiLU value case records on. */
    class Medium_SiLUValues : public SaTest::EngineTest
    {
    };

    /**
     * @brief Bounds the relative error of an engine FLOAT32 result against the exact value; 32 unit
     * roundoffs of float, the bound the Mish values are held to.
     */
    constexpr double K_RELATIVE_TOLERANCE = 32.0 * TestSupport::unit_roundoff<float>();

    /** @brief Computes x / (1 + exp(-x)) in double; the exact value the engine is held to. */
    double exact_silu(double x)
    {
        return x / (1.0 + std::exp(-x));
    }

    /** @brief Compares every element of @p got with @p expected to K_RELATIVE_TOLERANCE. */
    void expect_relative(const std::vector<double>& got, const std::vector<double>& expected)
    {
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            const double bound = K_RELATIVE_TOLERANCE * std::abs(expected[i]);
            EXPECT_NEAR(got[i], expected[i], bound) << "at element " << i;
        }
    }

    /** @brief Runs the SiLU layer over one FLOAT32 row on the engine and reads the result. */
    void run_silu(SushiAI::ExecutionBackend& backend, const std::vector<double>& input,
                  std::vector<double>& result)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const std::int64_t columns = static_cast<std::int64_t>(input.size());
        const Graph::ValueID x = builder.input(Shape{1, columns}, "x");
        const SiLU layer{};
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

/** @brief Tests that SiLU matches x * sigmoid(x) at -20, -1, 0, 1 and 20. */
TEST_F(Medium_SiLUValues, MatchesTheExactFormAtTheAcceptancePoints)
{
    const std::vector<double> input = {-20.0, -1.0, 0.0, 1.0, 20.0};
    std::vector<double> expected(input.size());
    std::transform(input.begin(), input.end(), expected.begin(), exact_silu);
    std::vector<double> result;

    ASSERT_NO_FATAL_FAILURE(run_silu(backend(), input, result));

    ASSERT_NO_FATAL_FAILURE(expect_relative(result, expected));
}
