/**************************************************************************/
/* test_block_language_head.cpp                                           */
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
 * @file test_block_language_head.cpp
 * @brief Tests the trace, the state, the refusals and the node factory of LanguageHead.
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
#include <SushiAI/language/blocks/language_head.hpp>
#include <SushiAI/nn/built_rows.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using Language::Blocks::LanguageHead;
using NN::Trace;

namespace
{
    /** @brief Holds the width of the sequence every case reads. */
    constexpr int64_t K_DIM = 6;

    /** @brief Holds the vocabulary size every case maps to. */
    constexpr int64_t K_VOCABULARY = 11;

    /** @brief Holds the nodes one trace adds: three leaves and four others. */
    constexpr std::size_t K_TRACE_NODES = 7;

    /** @brief Asks for the result shape of a per-sample shape that is not [T, C]. */
    void shape_of_a_rank_one_sample()
    {
        const LanguageHead head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
        static_cast<void>(head.output_shape(Shape{K_DIM}));
    }
}

/** @brief Tests that a trace normalises, then maps the rows to [N, T, vocabulary] with no bias. */
TEST(Unit_BlockLanguageHead, ATraceNormalisesThenMapsToTheVocabulary)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    LanguageHead head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = head.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_VOCABULARY}));
    EXPECT_EQ(graph.node_count(), nodes_before + K_TRACE_NODES);

    const Graph::Node& restored = graph.node(graph.value(y).producer);
    ASSERT_EQ(restored.op, Ops::K_RESHAPE.id);
    const Graph::Node& product = graph.node(graph.value(restored.inputs[0]).producer);
    ASSERT_EQ(product.op, Ops::K_MATMUL.id);
    const Graph::Node& rows = graph.node(graph.value(product.inputs[0]).producer);
    ASSERT_EQ(rows.op, Ops::K_RESHAPE.id);
    const Graph::Node& normalised = graph.node(graph.value(rows.inputs[0]).producer);
    ASSERT_EQ(normalised.op, Ops::K_LAYER_NORM.id);
    EXPECT_EQ(normalised.inputs[0], x);
    EXPECT_FALSE(head.linear.bias.declared());
}

/** @brief Tests that the walk names the normalisation's two parameters and the one weight. */
TEST(Unit_BlockLanguageHead, TheWalkYieldsTheNormalisationAndTheWeight)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    LanguageHead head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    static_cast<void>(head.forward(Trace::evaluation(builder),
                                   builder.input(Shape{2, 5, K_DIM}, "x")));

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
    NN::walk_state(head, "head", visitor);

    EXPECT_EQ(names, (std::vector<std::string>{"head.norm.weight", "head.norm.bias",
                                               "head.linear.weight"}));
    EXPECT_EQ(head.linear.weight.shape(), (Shape{K_VOCABULARY, K_DIM}));
}

/** @brief Tests that an input of another width, and a head of no width, each fail. */
TEST(Unit_BlockLanguageHead, AnInputOrADescriptionThatDoesNotFitFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID wide = builder.input(Shape{2, 5, K_DIM + 1}, "wide");
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    LanguageHead head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    LanguageHead empty{};
    const std::size_t nodes_before = graph.node_count();

    const Trace first = Trace::evaluation(builder);
    EXPECT_EQ(head.forward(first, wide), wide);
    SA_EXPECT_FAILURE(first.status(), StatusCode::InvalidDescription, "LanguageHead",
                      "rank-3 [N, T, 6] input", "[2, 5, 7]");

    const Trace second = Trace::evaluation(builder);
    EXPECT_EQ(empty.forward(second, x), x);
    SA_EXPECT_FAILURE(second.status(), StatusCode::InvalidDescription, "positive widths",
                      "dim=0 vocabulary=0");

    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that a tied head scores with the embedding's table and declares no weight. */
TEST(Unit_BlockLanguageHead, ATiedHeadScoresWithTheEmbeddingsTable)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Trace trace = Trace::evaluation(builder);

    NN::Embedding embedding{};
    embedding.num_embeddings = K_VOCABULARY;
    embedding.embedding_dim = K_DIM;
    const Graph::ValueID x =
        embedding.forward(trace, builder.input(Shape{2, 5}, "ids", DType::Int32));

    LanguageHead head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    head.tie_to(embedding.weight);
    const std::size_t nodes_before = graph.node_count();
    const Graph::ValueID y = head.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_VOCABULARY}));

    // The untied trace adds a weight leaf; the tied one reuses the table's.
    EXPECT_EQ(graph.node_count(), nodes_before + K_TRACE_NODES - 1);
    const Graph::Node& restored = graph.node(graph.value(y).producer);
    const Graph::Node& product = graph.node(graph.value(restored.inputs[0]).producer);
    ASSERT_EQ(product.op, Ops::K_MATMUL.id);
    EXPECT_EQ(product.inputs[1], embedding.weight.id_in(graph));

    EXPECT_FALSE(head.linear.weight.declared());
    EXPECT_EQ(NN::parameter_count(head), 2u);
    EXPECT_EQ(NN::parameter_count(embedding), 1u);
}

/** @brief Tests that a table no layer has declared, or of another shape, fails the trace. */
TEST(Unit_BlockLanguageHead, ATiedTableThatDoesNotFitFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");

    NN::Parameter undeclared;
    LanguageHead early(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    early.tie_to(undeclared);
    const Trace first = Trace::evaluation(builder);
    EXPECT_EQ(early.forward(first, x), x);
    SA_EXPECT_FAILURE(first.status(), StatusCode::InvalidDescription,
                      "tied to a table no layer has declared");

    NN::Parameter narrow;
    static_cast<void>(narrow.declare(builder, Shape{K_VOCABULARY, K_DIM - 1},
                                     NN::ParameterRole::Embedding, K_DIM - 1, K_DIM - 1,
                                     "table"));
    LanguageHead mismatched(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    mismatched.tie_to(narrow);
    const Trace second = Trace::evaluation(builder);
    EXPECT_EQ(mismatched.forward(second, x), x);
    SA_EXPECT_FAILURE(second.status(), StatusCode::InvalidDescription, "table of [11, 6]",
                      "tied to a float32 table of [11, 5]");
}

/** @brief Tests that output_shape replaces the width of a [T, C] shape by the vocabulary. */
TEST(Unit_BlockLanguageHead, OutputShapeReplacesTheWidthByTheVocabulary)
{
    const LanguageHead head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);

    EXPECT_EQ(head.output_shape(Shape{5, K_DIM}), (Shape{5, K_VOCABULARY}));
}

/** @brief Tests that the factory reads "vocabulary" and "ln_epsilon" and refuses a bad row. */
TEST(Unit_BlockLanguageHead, TheFactoryReadsTheVocabulary)
{
    const Shape input{5, K_DIM};
    const Shape flat{K_DIM};
    const Config::Value good =
        Config::parse_json(R"({"vocabulary": 11, "ln_epsilon": 0.001})", "<test>");
    const Config::Value absent = Config::parse_json("{}", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_language_head_node(NN::ShapeSpan(&input, 1), good, "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<LanguageHead>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().dim, K_DIM);
    EXPECT_EQ(wrapped->layer().vocabulary, K_VOCABULARY);
    EXPECT_EQ(wrapped->layer().norm.epsilon, 0.001);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, K_VOCABULARY}));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_language_head_node(NN::ShapeSpan(&input, 1), absent, "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "'vocabulary'", "absent");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_language_head_node(NN::ShapeSpan(&flat, 1), good, "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "rank-2 [T, C] input");
}

/** @brief Tests that "tie" binds the head to the table of the embedding row it names. */
TEST(Unit_BlockLanguageHead, TheFactoryTiesTheHeadToAnEmbeddingRow)
{
    NN::ModelGraph model(Shape{5});
    NN::Embedding table{};
    table.num_embeddings = K_VOCABULARY;
    table.embedding_dim = K_DIM;
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRef tokens,
                            model.add("0", std::move(table), model.input()));
    NN::LayerNorm norm{};
    norm.dim = K_DIM;
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRef normalised, model.add("1", norm, tokens));
    static_cast<void>(normalised);

    const NN::BuiltRows above(model);
    const Shape input{5, K_DIM};
    const Config::Value tied = Config::parse_json(R"({"tie": 0})", "<test>");
    const Config::Value before = Config::parse_json(R"({"tie": -1})", "<test>");
    const Config::Value resized = Config::parse_json(R"({"tie": 0, "vocabulary": 12})", "<test>");
    const Config::Value below = Config::parse_json(R"({"tie": 2})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_language_head_node(NN::ShapeSpan(&input, 1), tied, "layer 2",
                                                  above));
    const auto* wrapped = dynamic_cast<const NN::ModuleNode<LanguageHead>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().vocabulary, K_VOCABULARY);
    EXPECT_EQ(wrapped->layer().tied_weight, &NN::find_embedding(model.node(0))->weight);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, K_VOCABULARY}));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_language_head_node(NN::ShapeSpan(&input, 1), before, "layer 2",
                                                  above),
        StatusCode::InvalidDescription, "layer 2", "'tie'", "row 1", "no 'embedding' row");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_language_head_node(NN::ShapeSpan(&input, 1), resized, "layer 2",
                                                  above),
        StatusCode::InvalidDescription, "layer 2", "12 tokens", "[11, 6] table");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_language_head_node(NN::ShapeSpan(&input, 1), below, "layer 2",
                                                  above),
        StatusCode::InvalidDescription, "'tie'", "row 2", "not above");
    SA_EXPECT_FAILURE(
        Language::Blocks::make_language_head_node(NN::ShapeSpan(&input, 1), tied, "layer 0"),
        StatusCode::InvalidDescription, "layer 0", "not above");
}

/** @brief Tests that output_shape over a per-sample shape of rank one stops the process. */
TEST(Unit_BlockLanguageHeadDeathTest, OutputShapeOverARankOneSampleIsFatal)
{
    SA_EXPECT_FATAL(shape_of_a_rank_one_sample(), "LanguageHead::output_shape", "[6]");
}

/** @brief Tests that the incremental form of one position is the ordinary trace of it. */
TEST(Unit_BlockLanguageHead, TheIncrementalFormIsTheOrdinaryTraceOfOnePosition)
{
    LanguageHead whole(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    Graph::Graph whole_graph;
    const Graph::GraphBuilder whole_builder(whole_graph);
    static_cast<void>(whole.forward(Trace::evaluation(whole_builder),
                                    whole_builder.input(Shape{1, 1, K_DIM}, "x")));

    LanguageHead sliced(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    Graph::Graph sliced_graph;
    const Graph::GraphBuilder sliced_builder(sliced_graph);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(sliced_builder, fed, carried);
    const Graph::ValueID y = sliced.forward_incremental(
        trace, sliced_builder.input(Shape{1, 1, K_DIM}, "x"), Shape{5, K_DIM});

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(sliced_graph.value(y).shape, (Shape{1, 1, K_VOCABULARY}));
    ASSERT_EQ(sliced_graph.node_count(), whole_graph.node_count());
    for (std::size_t at = 0; at < whole_graph.node_count(); ++at)
        EXPECT_EQ(sliced_graph.nodes()[at].op, whole_graph.nodes()[at].op) << "node " << at;

    EXPECT_TRUE(fed.leaves().empty());
    EXPECT_TRUE(carried.values().empty());
}
