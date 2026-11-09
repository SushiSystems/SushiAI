/**************************************************************************/
/* test_language_head.cpp                                                 */
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
 * @file test_language_head.cpp
 * @brief Tests the trace, the tied table, the state and the node factory of LanguageHead.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/language_head.hpp>
#include <SushiAI/nn/built_rows.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include "blocks_test_support.hpp"

using namespace SushiAI;
using Language::Blocks::LanguageHead;
using NN::Trace;
namespace SaTest = SushiAI::Test;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Holds the width of the sequence every case reads. */
    constexpr int64_t K_DIM = 6;

    /** @brief Holds the vocabulary size every case maps to. */
    constexpr int64_t K_VOCABULARY = 11;

    /** @brief Holds the positions of the sequence the incremental case compares against. */
    constexpr int64_t K_POSITIONS = 3;

    /** @brief Holds the position the incremental case traces alone. */
    constexpr std::size_t K_LAST_POSITION = 2;

    /**
     * @brief Holds an epsilon that makes a row of mean 0 and variance 1 normalise to half
     *        of itself: 1 / sqrt(1 + 3) = 0.5.
     */
    constexpr double K_HALVING_EPSILON = 3.0;

    /** @brief Asks for the result shape of a per-sample shape that is not [T, C]. */
    void shape_of_a_rank_one_sample()
    {
        const LanguageHead head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
        static_cast<void>(head.output_shape(Shape{K_DIM}));
    }

    /** @brief Returns half of each row of @p rows dotted with each row of @p table. */
    std::vector<double> halved_scores_on_host(const std::vector<double>& rows,
                                              const std::vector<double>& table)
    {
        const std::size_t width = static_cast<std::size_t>(K_DIM);
        std::vector<double> scores;
        for (std::size_t row = 0; row < rows.size() / width; ++row)
        {
            for (std::size_t token = 0; token < table.size() / width; ++token)
            {
                double sum = 0.0;
                for (std::size_t c = 0; c < width; ++c)
                    sum += 0.5 * rows[row * width + c] * table[token * width + c];

                scores.push_back(sum);
            }
        }

        return scores;
    }

    /** @brief Leaves @p table undeclared. */
    Status leave_undeclared(const Graph::GraphBuilder&, NN::Parameter&)
    {
        return Status();
    }

    /** @brief Declares @p table as a [vocabulary, dim - 1] embedding table. */
    Status declare_narrow(const Graph::GraphBuilder& builder, NN::Parameter& table)
    {
        return table.declare(builder, Shape{K_VOCABULARY, K_DIM - 1},
                             NN::ParameterRole::Embedding, K_DIM - 1, K_DIM - 1, "table")
            .status();
    }

    /** @brief Holds one tied table the head cannot score with and two words of its message. */
    struct TableRow
    {
        std::string name;                                            /**< The row's name. */
        Status (*arrange)(const Graph::GraphBuilder&, NN::Parameter&); /**< Readies the table. */
        std::string first;                                           /**< A word of the message. */
        std::string second;                                          /**< Another, or empty. */
    };

    /** @brief Runs one refused table. */
    class Small_BlockLanguageHeadTable : public ::testing::TestWithParam<TableRow>
    {

    };

    /** @brief Holds one factory input the factory refuses and two words of its message. */
    struct FactoryRow
    {
        std::string name;   /**< The row's name. */
        Shape input;        /**< The per-sample shape. */
        std::string json;   /**< The row's attributes. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A second word of the message, or empty. */
    };

    /** @brief Runs one refused factory row. */
    class Small_BlockLanguageHeadFactory : public ::testing::TestWithParam<FactoryRow>
    {

    };

    /** @brief Holds one "tie" the factory refuses over the rows above and its message. */
    struct TieRow
    {
        std::string name;   /**< The row's name. */
        std::string json;   /**< The row's attributes. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A second word of the message. */
        std::string third;  /**< A third word of the message, or empty. */
    };

    /** @brief Builds an embedding row and a layer norm row for the head to tie to. */
    class Small_BlockLanguageHeadTie : public ::testing::TestWithParam<TieRow>
    {
        protected:
            /** @brief Adds the embedding as row 0 and a layer norm as row 1. */
            Small_BlockLanguageHeadTie()
            {
                NN::Embedding table{};
                table.num_embeddings = K_VOCABULARY;
                table.embedding_dim = K_DIM;
                const NN::NodeRef tokens =
                    require_value(model_.add("0", std::move(table), model_.input()));
                NN::LayerNorm norm{};
                norm.dim = K_DIM;
                static_cast<void>(require_value(model_.add("1", norm, tokens)));
            }

            /// @brief Holds the two rows above the head.
            NN::ModelGraph model_{Shape{5}};
    };

    /** @brief Names a row by its name field. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that a trace maps [N, T, dim] to [N, T, vocabulary] and declares no bias. */
TEST(Small_BlockLanguageHead, ATraceNormalisesThenMapsToTheVocabulary)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    LanguageHead head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = head.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_VOCABULARY}));
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_FALSE(head.linear.bias.declared());
}

/** @brief Tests that the walk names the normalisation's two parameters and the one weight. */
TEST(Small_BlockLanguageHead, TheWalkYieldsTheNormalisationAndTheWeight)
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

/** @brief Tests that an input of another width fails and leaves the graph as it was. */
TEST(Small_BlockLanguageHead, AnInputOfAnotherWidthIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID wide = builder.input(Shape{2, 5, K_DIM + 1}, "wide");
    LanguageHead head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = head.forward(trace, wide);

    EXPECT_EQ(y, wide);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "LanguageHead",
                      "rank-3 [N, T, 6] input", "[2, 5, 7]");
    EXPECT_EQ(graph.node_count(), nodes_before);
}

/** @brief Tests that a head of no width fails the trace and names both widths. */
TEST(Small_BlockLanguageHead, AHeadOfNoWidthIsRefused)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    LanguageHead empty{};

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = empty.forward(trace, x);

    EXPECT_EQ(y, x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "positive widths",
                      "dim=0 vocabulary=0");
}

/**
 * @brief Tests that a tied head scores the normalised rows with the table, transposed.
 *
 * Each row has mean 0 and variance 1, so with a gain of 1, a bias of 0 and an epsilon of 3 it
 * normalises to half of itself, and every logit is exact.
 */
TEST(Small_BlockLanguageHead, ATiedHeadScoresWithTheEmbeddingsTable)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 2, K_DIM}, "x");
    NN::Parameter table;
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID table_id,
                            table.declare(builder, Shape{K_VOCABULARY, K_DIM},
                                          NN::ParameterRole::Embedding, K_DIM, K_DIM, "table"));
    LanguageHead head(K_DIM, K_VOCABULARY, K_HALVING_EPSILON);
    head.tie_to(table);
    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = head.forward(trace, x);
    const std::vector<double> rows{1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1};
    const std::vector<double> scores = SaTest::eighths(K_VOCABULARY * K_DIM, 0);
    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, rows));
    SA_EXPECT_OK(evaluator.write(table_id, scores));
    SA_EXPECT_OK(evaluator.write(head.norm.weight.id_in(graph), std::vector<double>(K_DIM, 1.0)));
    SA_EXPECT_OK(evaluator.write(head.norm.bias.id_in(graph), std::vector<double>(K_DIM, 0.0)));

    SA_EXPECT_OK(evaluator.run());

    SA_EXPECT_OK(trace.status());
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> logits, evaluator.read(y));
    EXPECT_EQ(logits, halved_scores_on_host(rows, scores));
}

/** @brief Tests that a tied head keeps the [N, T, vocabulary] shape and declares no weight. */
TEST(Small_BlockLanguageHead, ATiedHeadDeclaresNoWeightOfItsOwn)
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

    const Graph::ValueID y = head.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_VOCABULARY}));
    EXPECT_FALSE(head.linear.weight.declared());
    EXPECT_EQ(NN::parameter_count(head), 2u);
    EXPECT_EQ(NN::parameter_count(embedding), 1u);
}

/** @brief Tests that a table no layer has declared, or of another shape, fails the trace. */
TEST_P(Small_BlockLanguageHeadTable, ATiedTableThatDoesNotFitFails)
{
    const TableRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    NN::Parameter table;
    SA_EXPECT_OK(row.arrange(builder, table));
    LanguageHead head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    head.tie_to(table);

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = head.forward(trace, x);

    EXPECT_EQ(y, x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockLanguageHeadTable,
    ::testing::Values(TableRow{"Undeclared", &leave_undeclared,
                               "tied to a table no layer has declared", ""},
                      TableRow{"Narrow", &declare_narrow, "table of [11, 6]",
                               "tied to a float32 table of [11, 5]"}),
    row_name<TableRow>);

/** @brief Tests that output_shape replaces the width of a [T, C] shape by the vocabulary. */
TEST(Small_BlockLanguageHead, OutputShapeReplacesTheWidthByTheVocabulary)
{
    const LanguageHead head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);

    EXPECT_EQ(head.output_shape(Shape{5, K_DIM}).value(), (Shape{5, K_VOCABULARY}));
}

/** @brief Tests that the factory reads "vocabulary" and "ln_epsilon" into a one-port node. */
TEST(Small_BlockLanguageHead, TheFactoryReadsTheVocabulary)
{
    const Shape input{5, K_DIM};
    const Config::Value good =
        require_value(Config::parse_json(R"({"vocabulary": 11, "ln_epsilon": 0.001})", "<test>"));

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
}

/** @brief Tests that the factory refuses an absent vocabulary and a rank-one input. */
TEST_P(Small_BlockLanguageHeadFactory, TheFactoryRefusesARowThatDoesNotFit)
{
    const FactoryRow& row = GetParam();
    const Config::Value attributes = require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_language_head_node(NN::ShapeSpan(&row.input, 1), attributes,
                                                  "layer 3"),
        StatusCode::InvalidDescription, "layer 3", row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockLanguageHeadFactory,
    ::testing::Values(FactoryRow{"NoVocabulary", Shape{5, K_DIM}, "{}", "'vocabulary'",
                                 "absent"},
                      FactoryRow{"RankOne", Shape{K_DIM}, R"({"vocabulary": 11})",
                                 "rank-2 [T, C] input", ""}),
    row_name<FactoryRow>);

/** @brief Tests that "tie" binds the head to the table of the embedding row it names. */
TEST_F(Small_BlockLanguageHeadTie, TheFactoryTiesTheHeadToAnEmbeddingRow)
{
    const NN::BuiltRows above(model_);
    const Shape input{5, K_DIM};
    const Config::Value tied = require_value(Config::parse_json(R"({"tie": 0})", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_language_head_node(NN::ShapeSpan(&input, 1), tied, "layer 2",
                                                  above));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<LanguageHead>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().vocabulary, K_VOCABULARY);
    EXPECT_EQ(wrapped->layer().tied_weight, &NN::find_embedding(model_.node(0))->weight);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, K_VOCABULARY}));
}

/** @brief Tests that a "tie" the rows above cannot serve is refused with its reason. */
TEST_P(Small_BlockLanguageHeadTie, TheFactoryRefusesATieTheRowsCannotServe)
{
    const TieRow& row = GetParam();
    const NN::BuiltRows above(model_);
    const Shape input{5, K_DIM};
    const Config::Value attributes = require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_language_head_node(NN::ShapeSpan(&input, 1), attributes,
                                                  "layer 2", above),
        StatusCode::InvalidDescription, row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockLanguageHeadTie,
    ::testing::Values(TieRow{"NotAnEmbedding", R"({"tie": -1})", "'tie'", "row 1",
                             "no 'embedding' row"},
                      TieRow{"OtherVocabulary", R"({"tie": 0, "vocabulary": 12})", "layer 2",
                             "12 tokens", "[11, 6] table"},
                      TieRow{"NotAbove", R"({"tie": 2})", "'tie'", "row 2", "not above"}),
    row_name<TieRow>);

/** @brief Tests that a tie with no rows above the head is refused. */
TEST(Small_BlockLanguageHead, ATieWithNoRowsAboveIsRefused)
{
    const Shape input{5, K_DIM};
    const Config::Value tied = require_value(Config::parse_json(R"({"tie": 0})", "<test>"));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_language_head_node(NN::ShapeSpan(&input, 1), tied, "layer 0"),
        StatusCode::InvalidDescription, "layer 0", "not above");
}

/** @brief Tests that output_shape over a per-sample shape of rank one stops the process. */
TEST(Medium_BlockLanguageHeadDeathTest, OutputShapeOverARankOneSampleIsFatal)
{
    SA_EXPECT_FATAL(shape_of_a_rank_one_sample(), "LanguageHead::output_shape", "[6]");
}

/**
 * @brief Tests that the incremental logits of one position are the ordinary logits at it.
 *
 * Both graphs run on the host evaluator over the same state and the same rows, so the row of
 * the position is computed by the same operations in the same order and compares exactly.
 */
TEST(Small_BlockLanguageHead, TheIncrementalFormIsTheOrdinaryTraceOfOnePosition)
{
    const std::vector<double> rows = SaTest::eighths(K_POSITIONS * K_DIM, 0);
    LanguageHead whole(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    Graph::Graph whole_graph;
    const Graph::GraphBuilder whole_builder(whole_graph);
    const Graph::ValueID sequence = whole_builder.input(Shape{1, K_POSITIONS, K_DIM}, "x");
    const Graph::ValueID whole_y = whole.forward(Trace::evaluation(whole_builder), sequence);
    Graph::HostEvaluator whole_run(whole_graph);
    SA_EXPECT_OK(whole_run.write(sequence, rows));
    SA_EXPECT_OK(SaTest::write_state(whole_run, whole_graph, whole));
    SA_EXPECT_OK(whole_run.run());
    LanguageHead sliced(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    Graph::Graph sliced_graph;
    const Graph::GraphBuilder sliced_builder(sliced_graph);
    const Graph::ValueID position = sliced_builder.input(Shape{1, 1, K_DIM}, "x");
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(sliced_builder, fed, carried);
    const Graph::ValueID sliced_y =
        sliced.forward_incremental(trace, position, Shape{K_POSITIONS, K_DIM});
    Graph::HostEvaluator sliced_run(sliced_graph);
    SA_EXPECT_OK(sliced_run.write(position, SaTest::row_of(rows, K_LAST_POSITION, K_DIM)));
    SA_EXPECT_OK(SaTest::write_state(sliced_run, sliced_graph, sliced));

    SA_EXPECT_OK(sliced_run.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> ordinary, whole_run.read(whole_y));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> incremental, sliced_run.read(sliced_y));
    EXPECT_EQ(incremental, SaTest::row_of(ordinary, K_LAST_POSITION, K_VOCABULARY));
}

/** @brief Tests that the incremental form declares no fed leaf and carries no value. */
TEST(Small_BlockLanguageHead, TheIncrementalFormFeedsAndCarriesNothing)
{
    LanguageHead head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON);
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(builder, fed, carried);

    const Graph::ValueID y = head.forward_incremental(
        trace, builder.input(Shape{1, 1, K_DIM}, "x"), Shape{K_POSITIONS, K_DIM});

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_VOCABULARY}));
    EXPECT_TRUE(fed.leaves().empty());
    EXPECT_TRUE(carried.values().empty());
}
