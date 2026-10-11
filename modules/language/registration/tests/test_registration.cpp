/**************************************************************************/
/* test_registration.cpp                                                  */
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
 * @file test_registration.cpp
 * @brief Tests that the register functions of language add their rows, once.
 * @author Mustafa Garip
 */

#include <memory>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/expert_trace.hpp>
#include <SushiAI/language/registration.hpp>
#include <SushiAI/language/training/expert_balance_objective.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include <common/model_text.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST body, where Test names the fixture base.
namespace SaTest = SushiAI::Test;

/** @brief Tests that the five block rows and the linked head row are added in order. */
TEST(Unit_LanguageRegistration, RegisterNodesAddsTheSixBlockRows)
{
    NN::NodeRegistry registry;
    SA_EXPECT_OK(Language::register_nodes(registry));

    EXPECT_EQ(SaTest::types_of(registry.nodes()),
              (std::vector<std::string_view>{"position_embedding", "causal_self_attention",
                                             "feed_forward", "transformer_block",
                                             "moe_feed_forward"}));
    EXPECT_EQ(SaTest::types_of(registry.linked()),
              (std::vector<std::string_view>{"language_head"}));

    SA_EXPECT_FAILURE(Language::register_nodes(registry), StatusCode::InvalidDescription,
                      "'position_embedding'", "already");
}

/** @brief Tests that the tokens row and the prompt row are added. */
TEST(Unit_LanguageRegistration, RegisterDatasetsAddsTokensAndPrompt)
{
    Data::DatasetRegistry registry;
    SA_EXPECT_OK(Language::register_datasets(registry));

    EXPECT_NE(registry.find("tokens"), nullptr);
    EXPECT_NE(registry.find("prompt"), nullptr);
    EXPECT_EQ(registry.rows().size(), 2u);
}

/** @brief Tests that the next_token row and the expert_balance row are added. */
TEST(Unit_LanguageRegistration, RegisterObjectivesAddsNextTokenAndExpertBalance)
{
    Train::ObjectiveRegistry registry;
    SA_EXPECT_OK(Language::register_objectives(registry));

    EXPECT_NE(registry.find("next_token"), nullptr);
    EXPECT_NE(registry.find("expert_balance"), nullptr);
    EXPECT_EQ(registry.rows().size(), 2u);
}

/** @brief Tests that the row joins the rows of train once and builds through the registry. */
TEST(Unit_LanguageRegistration, TheNextTokenRowBuildsBesideTheRowsOfTrain)
{
    Train::ObjectiveRegistry registry;
    SA_EXPECT_OK(Train::register_objectives(registry));
    SA_EXPECT_OK(Language::register_objectives(registry));

    // A bigram model over 5 token ids: 2 rows of 3 positions, scored as [2, 3, 5].
    Train::ObjectiveContext given;
    given.batch = 2;
    given.classes = 5;
    given.input_sample = Shape{3};
    given.outputs.push_back(Train::OutputShape{"out", Shape{2, 3, 5}});

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        Train::make_objective(registry, "next_token", given,
                              Config::parse_json("{}", "<test>"), "objective"));
    EXPECT_EQ(objective->name(), "next_token");

    SA_EXPECT_FAILURE(Language::register_objectives(registry),
                      StatusCode::InvalidDescription, "'next_token' is already registered");
}

/** @brief Tests that a decoder written as a model file builds from the rows of nn and language. */
TEST(Unit_LanguageRegistration, ADecoderModelFileBuilds)
{
    NN::NodeRegistry registry = SaTest::make_layer_registry();
    SA_EXPECT_OK(Language::register_nodes(registry));
    const Config::ModelSpec spec = SaTest::model_from(R"({
        "format_version": 2,
        "input": {"shape": [8], "dtype": "int32"},
        "layers": [
            {"type": "embedding", "num_embeddings": 32, "embedding_dim": 16},
            {"type": "position_embedding"},
            {"type": "transformer_block", "heads": 2},
            {"type": "language_head", "tie": 0}
        ]
    })");
    const Config::ModelSpec untied_row = SaTest::model_from(R"({
        "format_version": 2,
        "input": {"shape": [8], "dtype": "int32"},
        "layers": [
            {"type": "embedding", "num_embeddings": 32, "embedding_dim": 16},
            {"type": "position_embedding"},
            {"type": "language_head", "tie": 1}
        ]
    })");

    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelGraph model, NN::build_model(registry, spec));
    EXPECT_EQ(model.size(), 4u);
    EXPECT_EQ(model.output_shape(Shape{8}), (Shape{8, 32}));

    SA_EXPECT_FAILURE(NN::build_model(registry, untied_row), StatusCode::InvalidDescription,
                      "layer 2 'language_head'", "'tie'", "row 1", "no 'embedding' row");
}

/** @brief Tests that a block with experts publishes the outputs the balance row reads. */
TEST(Unit_LanguageRegistration, ADecoderWithExpertsPublishesWhatTheBalanceRowReads)
{
    NN::NodeRegistry nodes = SaTest::make_layer_registry();
    SA_EXPECT_OK(Language::register_nodes(nodes));
    const Config::ModelSpec spec = SaTest::model_from(R"({
        "format_version": 2,
        "input": {"shape": [8], "dtype": "int32"},
        "layers": [
            {"type": "embedding", "num_embeddings": 32, "embedding_dim": 16},
            {"type": "position_embedding"},
            {"type": "transformer_block", "heads": 2, "experts": 4, "hidden": 8},
            {"type": "moe_feed_forward", "experts": 2, "hidden": 8, "top_k": 1},
            {"type": "language_head", "tie": 0}
        ],
        "outputs": [
            {"node": 4},
            {"node": 2, "port": "balance", "name": "balance2"},
            {"node": 2, "port": "usage", "name": "usage2"},
            {"node": 3, "port": "balance", "name": "balance3"},
            {"node": 3, "port": "usage", "name": "usage3"}
        ]
    })");

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, NN::build_model(nodes, spec));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const NN::Trace trace = NN::Trace::evaluation(builder);
    SA_ASSERT_OK_AND_ASSIGN(
        const NN::ModelOutputs outputs,
        model.trace(trace, builder.input(Shape{2, 8}, "tokens", DType::Int32)));

    ASSERT_EQ(outputs.all().size(), 5u);
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID logits, outputs.at("output"));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID balance, outputs.at("balance2"));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID usage, outputs.at("usage3"));
    EXPECT_EQ(graph.value(logits).shape, (Shape{2, 8, 32}));
    EXPECT_EQ(graph.value(balance).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(usage).shape, (Shape{1, 2}));

    // The ports of a block and the names the objective looks for are one spelling.
    EXPECT_EQ(Language::Blocks::K_BALANCE_PORT, Language::Training::K_BALANCE_OUTPUT_PREFIX);
    EXPECT_EQ(Language::Blocks::K_USAGE_PORT, Language::Training::K_USAGE_OUTPUT_PREFIX);

    Train::ObjectiveRegistry objectives;
    SA_EXPECT_OK(Train::register_objectives(objectives));
    SA_EXPECT_OK(Language::register_objectives(objectives));
    Train::ObjectiveContext given;
    given.batch = 2;
    given.classes = 32;
    given.input_sample = Shape{8};
    SA_ASSERT_OK_AND_ASSIGN(given.outputs, Train::list_output_shapes(graph, outputs));

    const Config::Value sum = Config::parse_json(R"({"terms": [
        {"type": "next_token", "output": "output"},
        {"type": "expert_balance", "weight": 0.01}
    ]})", "<test>");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Objective> objective,
                            Train::make_objective(objectives, "sum", given, sum, "objective"));
    EXPECT_EQ(objective->name(), "sum");
}
