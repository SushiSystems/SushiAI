/**************************************************************************/
/* test_layer_norm.cpp                                                    */
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
 * @file test_layer_norm.cpp
 * @brief Tests the trace, the state, the refusals and the node factory of the LayerNorm layer.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using NN::LayerNorm;
using NN::Trace;

namespace
{
    /** @brief Holds the last extent every case normalises over. */
    constexpr int64_t K_DIM = 4;

    /** @brief Returns a layer over K_DIM features with the default epsilon. */
    LayerNorm make_layer()
    {
        LayerNorm layer{};
        layer.dim = K_DIM;
        return layer;
    }

    /** @brief Traces @p layer over @p x, expects @p x back, and returns what was recorded. */
    Status refused(LayerNorm& layer, const Graph::GraphBuilder& builder, Graph::ValueID x)
    {
        const Trace trace = Trace::evaluation(builder);
        EXPECT_EQ(layer.forward(trace, x), x);
        return trace.status();
    }

    /** @brief Traces a layer over a value id the one-value graph does not hold. */
    void trace_over_a_value_the_graph_lacks()
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        static_cast<void>(builder.input(Shape{2, K_DIM}, "x"));
        LayerNorm layer = make_layer();
        static_cast<void>(layer.forward(Trace::evaluation(builder), Graph::ValueID{7}));
    }
}

static_assert(NN::TraceModule<LayerNorm> && !NN::Module<LayerNorm> && !NN::ModalModule<LayerNorm>,
              "LayerNorm has the trace form alone");

/** @brief Tests that a trace adds two leaves and one LAYER_NORM over [x, weight, bias]. */
TEST(Unit_LayerNorm, ATraceDeclaresTheGainAndTheBiasAndEmitsOneNode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3, K_DIM}, "x");
    LayerNorm layer = make_layer();
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::training(builder, NN::discard_updates());
    const Graph::ValueID y = layer.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.node_count(), nodes_before + 3u);

    const Graph::Node& node = graph.node(graph.value(y).producer);
    EXPECT_EQ(node.op, Ops::K_LAYER_NORM.id);
    ASSERT_EQ(node.inputs.size(), 3u);
    EXPECT_EQ(node.inputs[0], x);
    EXPECT_EQ(node.inputs[1], layer.weight.id_in(graph));
    EXPECT_EQ(node.inputs[2], layer.bias.id_in(graph));
    EXPECT_EQ(node.attributes.values[0], NN::K_LAYER_NORM_EPSILON);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 3, K_DIM}));
    EXPECT_EQ(graph.value(y).dtype, DType::Float32);
}

/** @brief Tests that the gain is a [dim] scale, the bias a [dim] bias, both of the input's type. */
TEST(Unit_LayerNorm, TheGainIsAScaleAndTheBiasABias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    LayerNorm layer = make_layer();
    layer.epsilon = 1e-3;
    const Graph::ValueID y = layer.forward(
        Trace::evaluation(builder), builder.input(Shape{5, K_DIM}, "x", DType::Float64));

    ASSERT_TRUE(layer.weight.declared());
    ASSERT_TRUE(layer.bias.declared());
    EXPECT_EQ(layer.weight.shape(), (Shape{K_DIM}));
    EXPECT_EQ(layer.bias.shape(), (Shape{K_DIM}));
    EXPECT_EQ(layer.weight.role(), NN::ParameterRole::Scale);
    EXPECT_EQ(layer.bias.role(), NN::ParameterRole::Bias);
    EXPECT_EQ(layer.weight.dtype(), DType::Float64);
    EXPECT_EQ(layer.bias.dtype(), DType::Float64);
    EXPECT_EQ(graph.node(graph.value(y).producer).attributes.values[0], 1e-3);
}

/** @brief Tests that the state walk yields the two parameters, weight before bias. */
TEST(Unit_LayerNorm, TheWalkYieldsTheGainThenTheBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    LayerNorm layer = make_layer();
    static_cast<void>(layer.forward(Trace::evaluation(builder),
                                    builder.input(Shape{5, K_DIM}, "x")));

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
    NN::walk_state(layer, "norm", visitor);

    EXPECT_EQ(names, (std::vector<std::string>{"norm.weight", "norm.bias"}));
}

/** @brief Tests that an input of another width and an integer input fail and declare nothing. */
TEST(Unit_LayerNorm, AnInputThatDoesNotFitFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID wide = builder.input(Shape{2, K_DIM + 1}, "wide");
    const Graph::ValueID whole = builder.input(Shape{2, K_DIM}, "whole", DType::Int32);
    LayerNorm layer = make_layer();
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(refused(layer, builder, wide), StatusCode::InvalidDescription,
                      "LayerNorm(4)", "[..., 4] input", "[2, 5]");
    SA_EXPECT_FAILURE(refused(layer, builder, whole), StatusCode::InvalidDescription,
                      "LayerNorm(4)", "half, float32 or float64 input", "int32");

    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_FALSE(layer.weight.declared());
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that a dim that is not positive and an epsilon that is not positive each fail. */
TEST(Unit_LayerNorm, ALayerDescribedWrongFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, K_DIM}, "x");

    LayerNorm no_dim{};
    LayerNorm no_epsilon = make_layer();
    no_epsilon.epsilon = 0.0;

    SA_EXPECT_FAILURE(refused(no_dim, builder, x), StatusCode::InvalidDescription,
                      "positive dim", "got 0");
    SA_EXPECT_FAILURE(refused(no_epsilon, builder, x), StatusCode::InvalidDescription,
                      "LayerNorm(4)", "positive epsilon");
    EXPECT_FALSE(no_epsilon.weight.declared());
}

/** @brief Tests that output_shape returns the per-sample shape it is given. */
TEST(Unit_LayerNorm, OutputShapeIsTheInputShape)
{
    const LayerNorm layer = make_layer();

    EXPECT_EQ(layer.output_shape(Shape{K_DIM}), (Shape{K_DIM}));
    EXPECT_EQ(layer.output_shape(Shape{6, K_DIM}), (Shape{6, K_DIM}));
}

/** @brief Tests that the factory takes dim from the last extent and reads "ln_epsilon". */
TEST(Unit_LayerNorm, TheFactoryReadsTheWidthAndTheEpsilon)
{
    const Shape input{6, K_DIM};
    const Config::Value plain = Config::parse_json("{}", "<test>");
    const Config::Value stated = Config::parse_json(R"({"ln_epsilon": 0.001})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        NN::make_layer_norm_node(NN::ShapeSpan(&input, 1), plain, "layer 0"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> tuned,
        NN::make_layer_norm_node(NN::ShapeSpan(&input, 1), stated, "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<LayerNorm>*>(node.get());
    const auto* wrapped_tuned = dynamic_cast<const NN::ModuleNode<LayerNorm>*>(tuned.get());
    ASSERT_NE(wrapped, nullptr);
    ASSERT_NE(wrapped_tuned, nullptr);
    EXPECT_EQ(wrapped->layer().dim, K_DIM);
    EXPECT_EQ(wrapped->layer().epsilon, NN::K_LAYER_NORM_EPSILON);
    EXPECT_EQ(wrapped_tuned->layer().epsilon, 0.001);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{6, K_DIM}));
}

/** @brief Tests that the factory refuses a stray key, a bad epsilon and an input that is off. */
TEST(Unit_LayerNorm, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{6, K_DIM};
    const Shape pair[] = {Shape{K_DIM}, Shape{K_DIM}};
    const Shape deep{1, 1, 1, 1, 1, K_DIM};
    const Config::Value plain = Config::parse_json("{}", "<test>");
    const Config::Value stray = Config::parse_json(R"({"momentum": 0.1})", "<test>");
    const Config::Value negative = Config::parse_json(R"({"ln_epsilon": -1.0})", "<test>");
    const Config::Value text = Config::parse_json(R"({"ln_epsilon": "small"})", "<test>");

    SA_EXPECT_FAILURE(NN::make_layer_norm_node(NN::ShapeSpan(&input, 1), stray, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'momentum'");
    SA_EXPECT_FAILURE(NN::make_layer_norm_node(NN::ShapeSpan(&input, 1), negative, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'ln_epsilon'", "positive");
    SA_EXPECT_FAILURE(NN::make_layer_norm_node(NN::ShapeSpan(&input, 1), text, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'ln_epsilon'", "a number");
    SA_EXPECT_FAILURE(NN::make_layer_norm_node(NN::ShapeSpan(pair, 2), plain, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "reads one input", "given 2");
    SA_EXPECT_FAILURE(NN::make_layer_norm_node(NN::ShapeSpan(&deep, 1), plain, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "rank 1 to 5");
}

/** @brief Tests that a trace over a value the graph does not hold stops the process. */
TEST(Unit_LayerNormDeathTest, ATraceOverAValueTheGraphLacksIsFatal)
{
    SA_EXPECT_FATAL(trace_over_a_value_the_graph_lacks(), "LayerNorm(4)", "value 7",
                    "holds 1 values");
}
