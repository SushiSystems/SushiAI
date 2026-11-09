/**************************************************************************/
/* test_layers.cpp                                                        */
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
 * @file test_layers.cpp
 * @brief Tests the built-in layers: their backward topology, the loss root they feed and the shape
 *        errors they name.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using Graph::GraphBuilder;
using Graph::K_NO_VALUE;
using Graph::ValueID;
using Graph::NodeID;

namespace
{
    /** @brief How many nodes of @p op the whole graph holds. */
    std::size_t count_op(const Graph::Graph& graph, OpID op)
    {
        return static_cast<std::size_t>(
            std::count_if(graph.nodes().begin(), graph.nodes().end(),
                          [op](const Graph::Node& node) { return node.op == op; }));
    }

} // namespace

namespace
{

    /** @brief Differentiates @p loss with respect to @p wrt. */
    Result<std::vector<ValueID>> differentiate(Graph::Graph& graph, ValueID loss,
                                       const std::vector<ValueID>& wrt,
                                       ValueID seed_factor = K_NO_VALUE,
                                       Autograd::Provenance* provenance = nullptr)
    {
        return Autograd::differentiate(graph, loss,
                                       SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()),
                                       seed_factor, provenance);
    }

} // namespace

/** @brief Tests that every parameter of a two-layer MLP receives a gradient of its own shape. */
TEST(Small_Layers, EveryParameterOfATwoLayerMLPReceivesAGradient)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{8, 6}, "x");
    const ValueID targets = builder.input(Shape{8, 3}, "targets");

    NN::Sequential model{NN::Linear{6, 4}, NN::ReLU{}, NN::Linear{4, 3}};
    SA_ASSERT_OK_AND_ASSIGN(const ValueID logits, model.forward(builder, x));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            NN::CrossEntropyLoss{}.forward(builder, logits, targets));

    std::vector<ValueID> wrt;
    NN::for_each_parameter(model, [&wrt, &graph](const std::string&, NN::Parameter& parameter)
    {
        wrt.push_back(parameter.id_in(graph));
    });
    ASSERT_EQ(wrt.size(), 4u);

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<ValueID> grads, Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())));

    ASSERT_EQ(grads.size(), 4u);
    for (std::size_t i = 0; i < grads.size(); ++i)
    {
        ASSERT_NE(grads[i], K_NO_VALUE) << "parameter " << i << " received no gradient";
        EXPECT_EQ(graph.value(grads[i]).shape, graph.value(wrt[i]).shape);
        EXPECT_EQ(graph.value(grads[i]).lifetime, Graph::Lifetime::Gradient);
    }
}

/** @brief Tests that an input that does not require a gradient receives none. */
TEST(Small_Layers, TheInputIsNotDifferentiatedWhenItDoesNotRequireOne)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{8, 6}, "x");
    const ValueID targets = builder.input(Shape{8, 3}, "targets");

    NN::Linear layer{6, 3};
    SA_ASSERT_OK_AND_ASSIGN(const ValueID logits, layer.forward(builder, x));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            NN::CrossEntropyLoss{}.forward(builder, logits, targets));

    const std::vector<ValueID> wrt = {x, layer.weight.id_in(graph)};
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<ValueID> grads, Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())));

    ASSERT_EQ(grads.size(), 2u);
    EXPECT_EQ(grads[0], K_NO_VALUE);
    EXPECT_NE(grads[1], K_NO_VALUE);
}

/** @brief Tests that a fused cross-entropy root emits no seed node. */
TEST(Small_Layers, AFusedCrossEntropyRootEmitsNoSeedNode)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueID x = builder.input(Shape{8, 6}, "x");
    const ValueID targets = builder.input(Shape{8, 3}, "targets");

    NN::Sequential model{NN::Linear{6, 4}, NN::ReLU{}, NN::Linear{4, 3}};
    SA_ASSERT_OK_AND_ASSIGN(const ValueID logits, model.forward(builder, x));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            NN::CrossEntropyLoss{}.forward(builder, logits, targets));

    std::vector<ValueID> wrt;
    NN::for_each_parameter(model, [&wrt, &graph](const std::string&, NN::Parameter& parameter)
    {
        wrt.push_back(parameter.id_in(graph));
    });

    SA_EXPECT_OK(differentiate(graph, loss, wrt));

    EXPECT_EQ(count_op(graph, Ops::K_ZERO.id), 0u);
    EXPECT_EQ(count_op(graph, Ops::K_CONSTANT.id), 0u);
}

namespace
{
    /** @brief Holds a Linear layer, an input shape it refuses and the words the refusal names. */
    struct LinearRefusalRow
    {
        std::string name;           /**< The row's name. */
        NN::Linear layer;           /**< The layer under test. */
        Shape input;                /**< The shape of the value the layer is traced over. */
        std::string first;          /**< The first substring the message names. */
        std::string second;         /**< The second substring the message names. */
        std::string third;          /**< The third substring the message names. */
    };
}

/** @brief Runs the Linear refusal over each input the layer does not fit. */
class Small_LayersLinearRefusal : public ::testing::TestWithParam<LinearRefusalRow>
{
};

/** @brief Tests that Linear names its own dimensions when the input of the row does not fit. */
TEST_P(Small_LayersLinearRefusal, LinearNamesItsOwnDimensions)
{
    LinearRefusalRow row = GetParam();
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(row.input, "x");

    SA_EXPECT_FAILURE(row.layer.forward(builder, x), StatusCode::InvalidDescription, row.first,
                      row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LayersLinearRefusal,
    ::testing::Values(
        LinearRefusalRow{"AnInputExtentThatIsNotTheFeatureCount", NN::Linear{6, 4}, Shape{8, 5},
                         "Linear(6, 4)", "[8, 5]", "inner extent must be 6"},
        LinearRefusalRow{"AnInputOfTheWrongRank", NN::Linear{5, 4}, Shape{5},
                         "Linear needs a rank-2 input", "[5]", ""},
        LinearRefusalRow{"AZeroFeatureCount", NN::Linear{0, 4}, Shape{8, 5},
                         "Linear needs positive feature counts", "in=0", "out=4"}),
    [](const ::testing::TestParamInfo<LinearRefusalRow>& info) { return info.param.name; });
