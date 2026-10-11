/**************************************************************************/
/* test_embedding.cpp                                                     */
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
 * @file test_embedding.cpp
 * @brief Tests the trace, the state, the refusals and the node factory of the Embedding layer.
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
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using NN::Embedding;
using NN::Trace;

namespace
{
    /** @brief Holds the row count of the table every case builds. */
    constexpr int64_t K_ROWS = 7;

    /** @brief Holds the row length of the table every case builds. */
    constexpr int64_t K_LENGTH = 3;

    /** @brief Returns a layer over a [K_ROWS, K_LENGTH] table of @p dtype. */
    Embedding make_layer(DType dtype = DType::Float32)
    {
        Embedding layer{};
        layer.num_embeddings = K_ROWS;
        layer.embedding_dim = K_LENGTH;
        layer.dtype = dtype;
        return layer;
    }

    /** @brief Counts the nodes of @p graph that carry @p op. */
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

    /** @brief Traces @p layer over @p x, expects @p x back, and returns what was recorded. */
    Status refused(Embedding& layer, const Graph::GraphBuilder& builder, Graph::ValueID x)
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
        static_cast<void>(builder.input(Shape{2, 5}, "indices", DType::Int32));
        Embedding layer = make_layer();
        static_cast<void>(layer.forward(Trace::evaluation(builder), Graph::ValueID{7}));
    }

    /** @brief Asks for the result shape of a per-sample index of rank five. */
    void shape_of_a_rank_five_sample()
    {
        const Embedding layer = make_layer();
        static_cast<void>(layer.output_shape(Shape{1, 1, 1, 1, 1}));
    }
}

static_assert(NN::TraceModule<Embedding> && !NN::Module<Embedding> && !NN::ModalModule<Embedding>,
              "Embedding has the trace form alone");

/** @brief Tests that a trace adds the table leaf and one GATHER_ROWS over [table, indices]. */
TEST(Unit_Embedding, ATraceDeclaresTheTableAndEmitsOneGather)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID indices = builder.input(Shape{2, 5}, "indices", DType::Int32);
    Embedding layer = make_layer();
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = layer.forward(trace, indices);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.node_count(), nodes_before + 2u);
    ASSERT_EQ(count_nodes(graph, Ops::K_GATHER_ROWS.id), 1u);

    const Graph::Node& node = graph.node(graph.value(y).producer);
    EXPECT_EQ(node.op, Ops::K_GATHER_ROWS.id);
    ASSERT_EQ(node.inputs.size(), 2u);
    EXPECT_EQ(node.inputs[0], layer.weight.id_in(graph));
    EXPECT_EQ(node.inputs[1], indices);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_LENGTH}));
    EXPECT_EQ(graph.value(y).dtype, DType::Float32);
}

/** @brief Tests that the table is a [V, D] Embedding parameter drawn from a normal of 0.02. */
TEST(Unit_Embedding, TheTableIsAnEmbeddingDrawnFromANormalOfFixedDeviation)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Embedding layer = make_layer(DType::Float64);
    const Graph::ValueID y = layer.forward(
        Trace::evaluation(builder), builder.input(Shape{4}, "indices", DType::Int32));

    ASSERT_TRUE(layer.weight.declared());
    EXPECT_EQ(layer.weight.shape(), (Shape{K_ROWS, K_LENGTH}));
    EXPECT_EQ(layer.weight.dtype(), DType::Float64);
    EXPECT_EQ(layer.weight.role(), NN::ParameterRole::Embedding);
    EXPECT_EQ(layer.weight.init().kind, NN::ParameterInit::Kind::Normal);
    EXPECT_NEAR(layer.weight.init().deviation, NN::K_EMBEDDING_DEVIATION, 0.0);
    EXPECT_NEAR(NN::K_EMBEDDING_DEVIATION, 0.02, 0.0);
    EXPECT_EQ(layer.weight.fan_in(), K_LENGTH);
    EXPECT_EQ(layer.weight.fan_out(), K_LENGTH);
    EXPECT_EQ(graph.value(y).dtype, DType::Float64);
    EXPECT_EQ(graph.value(layer.weight.id_in(graph)).lifetime, Graph::Lifetime::Parameter);
    EXPECT_TRUE(graph.value(layer.weight.id_in(graph)).requires_grad);
}

/** @brief Tests that the state walk yields the one parameter, named weight. */
TEST(Unit_Embedding, TheWalkYieldsTheTable)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Embedding layer = make_layer();
    static_cast<void>(layer.forward(Trace::evaluation(builder),
                                    builder.input(Shape{4}, "indices", DType::Int32)));

    std::vector<std::string> names;
    const auto on_parameter = [&names](const std::string& name, NN::Parameter& parameter)
    {
        names.push_back(name);
        EXPECT_EQ(parameter.shape(), (Shape{K_ROWS, K_LENGTH}));
    };
    const auto on_buffer = [&names](const std::string& name, NN::Buffer&)
    {
        names.push_back(name);
    };
    const NN::StateVisitor visitor(on_parameter, on_buffer);
    NN::walk_state(layer, "tokens", visitor);

    EXPECT_EQ(names, (std::vector<std::string>{"tokens.weight"}));
}

/** @brief Tests that traces into two graphs name one state and declare one leaf in each. */
TEST(Unit_Embedding, TwoGraphsNameTheSameTable)
{
    Embedding layer = make_layer();

    Graph::Graph first;
    const Graph::GraphBuilder first_builder(first);
    static_cast<void>(layer.forward(Trace::evaluation(first_builder),
                                    first_builder.input(Shape{2, 5}, "indices", DType::Int32)));
    const Graph::StateID state = layer.weight.state();

    Graph::Graph second;
    const Graph::GraphBuilder second_builder(second);
    static_cast<void>(layer.forward(Trace::evaluation(second_builder),
                                    second_builder.input(Shape{3, 2}, "indices", DType::Int32)));

    EXPECT_NE(state, Graph::K_NO_STATE);
    EXPECT_EQ(layer.weight.state(), state);
    EXPECT_NE(layer.weight.id_in(first), Graph::K_NO_VALUE);
    EXPECT_NE(layer.weight.id_in(second), Graph::K_NO_VALUE);
}

/** @brief Tests that a FLOAT32 input and an INT64 input fail and declare nothing. */
TEST(Unit_Embedding, AnInputThatIsNotInt32Fails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID real = builder.input(Shape{2, 5}, "real");
    const Graph::ValueID wide = builder.input(Shape{2, 5}, "wide", DType::Int64);
    Embedding layer = make_layer();
    const std::size_t values_before = graph.value_count();
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(refused(layer, builder, real), StatusCode::InvalidDescription,
                      "Embedding(7, 3)", "int32 input of indices", "float32");
    SA_EXPECT_FAILURE(refused(layer, builder, wide), StatusCode::InvalidDescription,
                      "Embedding(7, 3)", "int32 input of indices", "int64");

    EXPECT_EQ(graph.value_count(), values_before);
    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_FALSE(layer.weight.declared());
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that an index of the highest rank fails, as the result has no rank left. */
TEST(Unit_Embedding, AnInputOfTheHighestRankFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID indices =
        builder.input(Shape{1, 1, 1, 1, 1, 1}, "indices", DType::Int32);
    Embedding layer = make_layer();

    SA_EXPECT_FAILURE(refused(layer, builder, indices), StatusCode::InvalidDescription,
                      "Embedding(7, 3)", "rank 1 to 5", "[1, 1, 1, 1, 1, 1]");
    EXPECT_FALSE(layer.weight.declared());
}

/** @brief Tests that an extent that is not positive and an integer table each fail. */
TEST(Unit_Embedding, ALayerDescribedWrongFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID indices = builder.input(Shape{2, 5}, "indices", DType::Int32);

    Embedding no_rows = make_layer();
    no_rows.num_embeddings = 0;
    Embedding no_length = make_layer();
    no_length.embedding_dim = -1;
    Embedding integer_table = make_layer(DType::Int32);

    SA_EXPECT_FAILURE(refused(no_rows, builder, indices), StatusCode::InvalidDescription,
                      "positive extents", "num_embeddings=0");
    SA_EXPECT_FAILURE(refused(no_length, builder, indices), StatusCode::InvalidDescription,
                      "positive extents", "embedding_dim=-1");
    SA_EXPECT_FAILURE(refused(integer_table, builder, indices), StatusCode::InvalidDescription,
                      "Embedding(7, 3)", "half, float32 or float64 table", "int32");
    EXPECT_FALSE(no_rows.weight.declared());
    EXPECT_FALSE(integer_table.weight.declared());
}

/** @brief Tests that output_shape appends the row length to a per-sample shape of any rank. */
TEST(Unit_Embedding, OutputShapeAppendsTheRowLength)
{
    const Embedding layer = make_layer();

    EXPECT_EQ(layer.output_shape(Shape{5}), (Shape{5, K_LENGTH}));
    EXPECT_EQ(layer.output_shape(Shape{4, 2}), (Shape{4, 2, K_LENGTH}));
    EXPECT_EQ(layer.output_shape(Shape{1, 1, 1, 1}), (Shape{1, 1, 1, 1, K_LENGTH}));
}

/** @brief Tests that the factory reads both keys and builds a node over [T] into [T, D]. */
TEST(Unit_Embedding, TheFactoryReadsTheTwoExtents)
{
    const Shape input{5};
    const Config::Value attributes =
        Config::parse_json(R"({"num_embeddings": 11, "embedding_dim": 4})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        NN::make_embedding_node(NN::ShapeSpan(&input, 1), attributes, "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<Embedding>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().num_embeddings, 11);
    EXPECT_EQ(wrapped->layer().embedding_dim, 4);
    EXPECT_EQ(wrapped->layer().dtype, DType::Float32);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, 4}));
    EXPECT_EQ(node->input_count(), 1u);
}

/** @brief Tests that the factory refuses a stray key, an absent key and an input that is off. */
TEST(Unit_Embedding, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{5};
    const Shape pair[] = {Shape{5}, Shape{5}};
    const Shape deep{1, 1, 1, 1, 1};
    const Config::Value both =
        Config::parse_json(R"({"num_embeddings": 11, "embedding_dim": 4})", "<test>");
    const Config::Value stray = Config::parse_json(
        R"({"num_embeddings": 11, "embedding_dim": 4, "padding": 0})", "<test>");
    const Config::Value no_length = Config::parse_json(R"({"num_embeddings": 11})", "<test>");
    const Config::Value zero_rows =
        Config::parse_json(R"({"num_embeddings": 0, "embedding_dim": 4})", "<test>");

    SA_EXPECT_FAILURE(NN::make_embedding_node(NN::ShapeSpan(&input, 1), stray, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'padding'");
    SA_EXPECT_FAILURE(NN::make_embedding_node(NN::ShapeSpan(&input, 1), no_length, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'embedding_dim'", "absent");
    SA_EXPECT_FAILURE(NN::make_embedding_node(NN::ShapeSpan(&input, 1), zero_rows, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'num_embeddings'", "positive");
    SA_EXPECT_FAILURE(NN::make_embedding_node(NN::ShapeSpan(pair, 2), both, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "reads one input", "given 2");
    SA_EXPECT_FAILURE(NN::make_embedding_node(NN::ShapeSpan(&deep, 1), both, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "[1, 1, 1, 1, 1]", "rank 1 to 4");
}

/** @brief Tests that a trace over a value the graph does not hold stops the process. */
TEST(Unit_EmbeddingDeathTest, ATraceOverAValueTheGraphLacksIsFatal)
{
    SA_EXPECT_FATAL(trace_over_a_value_the_graph_lacks(), "Embedding(7, 3)", "value 7",
                    "holds 1 values");
}

/** @brief Tests that output_shape over a per-sample index of rank five stops the process. */
TEST(Unit_EmbeddingDeathTest, OutputShapeOverARankFiveSampleIsFatal)
{
    SA_EXPECT_FATAL(shape_of_a_rank_five_sample(), "Embedding::output_shape", "at most 4");
}
