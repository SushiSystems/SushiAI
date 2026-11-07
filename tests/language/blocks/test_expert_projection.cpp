/**************************************************************************/
/* test_expert_projection.cpp                                             */
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
 * @file test_expert_projection.cpp
 * @brief Tests the parameters, the result and the refusals of ExpertProjection::mix.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/blocks/expert_projection.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Language::Blocks::ExpertProjection;

namespace
{
    /** @brief Holds the width of the result. */
    constexpr int64_t K_DIM = 6;

    /** @brief Holds the hidden width of one expert. */
    constexpr int64_t K_HIDDEN = 8;

    /** @brief Holds the number of experts. */
    constexpr int64_t K_EXPERTS = 3;

    /** @brief Holds the rows every case mixes. */
    constexpr int64_t K_ROWS = 5;

    /** @brief Returns a projection of the extents above, with no parameter declared. */
    ExpertProjection make_projection()
    {
        ExpertProjection projection;
        projection.dim = K_DIM;
        projection.hidden = K_HIDDEN;
        projection.experts = K_EXPERTS;
        return projection;
    }

    /** @brief Provides the engine the numeric case records on. */
    class Medium_BlockExpertProjection : public SaTest::EngineTest
    {

    };

    /** @brief Returns @p count small integers that repeat with period @p period. */
    std::vector<double> counting(int64_t count, int64_t period)
    {
        std::vector<double> values(static_cast<std::size_t>(count));
        std::iota(values.begin(), values.end(), 0.0);
        std::transform(values.begin(), values.end(), values.begin(),
                       [period](double v) { return std::fmod(v, static_cast<double>(period)) - 1.0; });
        return values;
    }

    /** @brief Returns gated * weight^T + gate * bias, row by row, summed in double. */
    std::vector<double> mixed_on_host(const std::vector<double>& gated,
                                      const std::vector<double>& gate,
                                      const std::vector<double>& weight,
                                      const std::vector<double>& bias)
    {
        const std::size_t width = static_cast<std::size_t>(K_EXPERTS * K_HIDDEN);
        std::vector<double> mixed;
        for (std::size_t row = 0; row < static_cast<std::size_t>(K_ROWS); ++row)
        {
            for (std::size_t d = 0; d < static_cast<std::size_t>(K_DIM); ++d)
            {
                double sum = 0.0;
                for (std::size_t j = 0; j < width; ++j)
                    sum += gated[row * width + j] * weight[d * width + j];
                for (std::size_t e = 0; e < static_cast<std::size_t>(K_EXPERTS); ++e)
                    sum += gate[row * static_cast<std::size_t>(K_EXPERTS) + e] *
                           bias[e * static_cast<std::size_t>(K_DIM) + d];
                mixed.push_back(sum);
            }
        }

        return mixed;
    }
} // namespace

/** @brief Tests that a walk under a prefix yields the weight, then the bias, under it. */
TEST(Small_BlockExpertProjection, AWalkYieldsThePrefixedWeightThenTheBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID gated = builder.input(Shape{K_ROWS, K_EXPERTS * K_HIDDEN}, "gated");
    const Graph::ValueID gate = builder.input(Shape{K_ROWS, K_EXPERTS}, "gate");
    ExpertProjection projection = make_projection();
    SA_EXPECT_OK(projection.mix(builder, gated, gate));

    std::vector<std::string> names;
    const auto on_parameter = [&names](const std::string& name, NN::Parameter&)
    {
        names.push_back(name);
    };
    const auto on_buffer = [&names](const std::string& name, NN::Buffer&)
    {
        names.push_back(name);
    };
    const NN::StateVisitor visitor(on_parameter, on_buffer);
    NN::walk_state(projection, "ep", visitor);

    EXPECT_EQ(names, (std::vector<std::string>{"ep.weight", "ep.bias"}));
}

/** @brief Tests that a mix declares both parameters and returns a [R, dim] value. */
TEST(Small_BlockExpertProjection, AMixDeclaresTheWeightAndTheBiasAndReturnsRowsOfDim)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID gated = builder.input(Shape{K_ROWS, K_EXPERTS * K_HIDDEN}, "gated");
    const Graph::ValueID gate = builder.input(Shape{K_ROWS, K_EXPERTS}, "gate");
    ExpertProjection projection = make_projection();

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID mixed, projection.mix(builder, gated, gate));

    EXPECT_EQ(graph.value(mixed).shape, (Shape{K_ROWS, K_DIM}));
    EXPECT_EQ(projection.weight.shape(), (Shape{K_DIM, K_EXPERTS * K_HIDDEN}));
    EXPECT_EQ(projection.bias.shape(), (Shape{K_EXPERTS, K_DIM}));
    EXPECT_EQ(projection.weight.role(), NN::ParameterRole::Weight);
    EXPECT_EQ(projection.bias.role(), NN::ParameterRole::Bias);
}

/**
 * @brief Tests that a mix emits gated times the transposed weight plus gate times the bias.
 *
 * Every value is a small integer, so each product and sum is exact in float32 and the result is
 * compared with the host sum bit for bit.
 */
TEST_F(Medium_BlockExpertProjection, AMixSumsTheGatedProjectionAndTheGatedBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID gated = builder.input(Shape{K_ROWS, K_EXPERTS * K_HIDDEN}, "gated");
    const Graph::ValueID gate = builder.input(Shape{K_ROWS, K_EXPERTS}, "gate");
    ExpertProjection projection = make_projection();
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID mixed, projection.mix(builder, gated, gate));
    Graph::TensorPool pool(backend().engine());
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    const std::vector<double> gated_values = counting(K_ROWS * K_EXPERTS * K_HIDDEN, 3);
    const std::vector<double> gate_values = counting(K_ROWS * K_EXPERTS, 2);
    const std::vector<double> weight_values = counting(K_DIM * K_EXPERTS * K_HIDDEN, 5);
    const std::vector<double> bias_values = counting(K_EXPERTS * K_DIM, 7);
    SaTest::write_tensor(pool.tensor(gated), gated_values);
    SaTest::write_tensor(pool.tensor(gate), gate_values);
    SaTest::write_tensor(pool.tensor(projection.weight.id_in(graph)), weight_values);
    SaTest::write_tensor(pool.tensor(projection.bias.id_in(graph)), bias_values);
    std::unique_ptr<Graph::CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, Graph::CompiledStep::compile(backend(), graph, pool));

    SA_EXPECT_OK(step->run());

    EXPECT_EQ(SaTest::read_tensor(pool.tensor(mixed)),
              mixed_on_host(gated_values, gate_values, weight_values, bias_values));
}

/** @brief Tests that a projection with an extent of zero is refused before any node. */
TEST(Small_BlockExpertProjection, AnEmptyExtentIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID gated = builder.input(Shape{K_ROWS, K_EXPERTS * K_HIDDEN}, "gated");
    const Graph::ValueID gate = builder.input(Shape{K_ROWS, K_EXPERTS}, "gate");
    ExpertProjection projection = make_projection();
    projection.experts = 0;
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(projection.mix(builder, gated, gate), StatusCode::InvalidDescription,
                      "ExpertProjection", "experts=0");
    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_FALSE(projection.weight.declared());
}

/** @brief Tests that a gated value narrower than experts * hidden is refused. */
TEST(Small_BlockExpertProjection, AGatedValueOfAnotherWidthIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID gated = builder.input(Shape{K_ROWS, K_HIDDEN}, "gated");
    const Graph::ValueID gate = builder.input(Shape{K_ROWS, K_EXPERTS}, "gate");
    ExpertProjection projection = make_projection();

    SA_EXPECT_FAILURE(projection.mix(builder, gated, gate), StatusCode::InvalidDescription);
}
