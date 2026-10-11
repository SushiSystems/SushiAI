/**************************************************************************/
/* test_training_next_token_objective.cpp                                 */
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
 * @file test_training_next_token_objective.cpp
 * @brief Tests the graph, the leaves, the factory and the registry row of the next-token
 *        objective.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/training/next_token_objective.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using Language::Training::make_next_token_objective;
using Language::Training::NextTokenObjective;

namespace
{
    /** @brief Holds the rows of a batch, N. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the positions of a row, T. */
    constexpr std::int64_t K_LENGTH = 3;

    /** @brief Holds the vocabulary size, V. */
    constexpr std::int64_t K_VOCABULARY = 5;

    /** @brief Returns a model whose [N, T] ids give [N, T, V] logits: one table row each. */
    NN::Embedding make_bigram_model()
    {
        NN::Embedding model{};
        model.num_embeddings = K_VOCABULARY;
        model.embedding_dim = K_VOCABULARY;
        return model;
    }

    /** @brief Counts the nodes of @p graph whose operation is @p op. */
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

    /** @brief Returns what a factory is given for the bigram model over K_VOCABULARY ids. */
    Train::ObjectiveContext make_context()
    {
        Train::ObjectiveContext given;
        given.batch = K_BATCH;
        given.classes = K_VOCABULARY;
        given.input_sample = Shape{K_LENGTH};
        given.outputs.push_back(
            Train::OutputShape{"out", Shape{K_BATCH, K_LENGTH, K_VOCABULARY}});
        return given;
    }

    /** @brief Returns the attributes of an objective that states no key. */
    Config::Value no_keys()
    {
        return Config::parse_json("{}", "<test>");
    }
} // namespace

/** @brief Tests that make() keeps the three extents and refuses one below 1. */
TEST(Unit_LanguageTrainingNextTokenObjective, MakeKeepsItsExtentsAndRefusesNone)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenObjective> objective,
                            NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));
    EXPECT_EQ(objective->batch(), K_BATCH);
    EXPECT_EQ(objective->length(), K_LENGTH);
    EXPECT_EQ(objective->vocabulary(), K_VOCABULARY);
    EXPECT_EQ(objective->name(), "next_token");

    SA_EXPECT_FAILURE(NextTokenObjective::make(0, K_LENGTH, K_VOCABULARY),
                      StatusCode::InvalidDescription, "at least 1");
    SA_EXPECT_FAILURE(NextTokenObjective::make(K_BATCH, K_LENGTH, 0),
                      StatusCode::InvalidDescription, "at least 1");
}

/** @brief Tests that a product INT32 cannot index is refused, and the largest one is not. */
TEST(Unit_LanguageTrainingNextTokenObjective, MakeRefusesWhatInt32CannotIndex)
{
    // 64 * 1024 * 50257 is 3 293 642 752, above 2 147 483 647.
    SA_EXPECT_FAILURE(NextTokenObjective::make(64, 1024, 50257),
                      StatusCode::InvalidDescription, "int32 index", "64 * 1024 * 50257");

    // 32 * 256 * 8192 is 67 108 864.
    SA_EXPECT_OK(NextTokenObjective::make(32, 256, 8192));
    SA_EXPECT_OK(NextTokenObjective::make(1, 1, 2147483647));
    SA_EXPECT_FAILURE(NextTokenObjective::make(1, 2, 2147483647),
                      StatusCode::InvalidDescription, "int32 index");
}

/** @brief Tests that the one leaf is INT32 [N * T] and that labels are read as [T] ids. */
TEST(Unit_LanguageTrainingNextTokenObjective, ItDeclaresAnIndexLeafAndReadsLabels)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenObjective> objective,
                            NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective->declare(builder, leaves);

    ASSERT_EQ(leaves.all().size(), 1u);
    const Graph::Value& targets = graph.value(leaves.at("targets"));
    EXPECT_EQ(targets.shape, (Shape{K_BATCH * K_LENGTH}));
    EXPECT_EQ(targets.dtype, DType::Int32);

    ASSERT_EQ(leaves.required().size(), 1u);
    const Data::TargetDescription& labels = leaves.required().front();
    EXPECT_EQ(labels.role, Data::K_ROLE_LABELS);
    EXPECT_EQ(labels.sample_shape, (Shape{K_LENGTH}));
    EXPECT_EQ(labels.dtype, DType::Int32);
    EXPECT_EQ(labels.classes, 0);
}

/** @brief Tests that the loss is one element built from a row gather and no one-hot product. */
TEST(Unit_LanguageTrainingNextTokenObjective, TheLossGathersOneLogProbabilityPerPosition)
{
    NN::Embedding model = make_bigram_model();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenObjective> objective,
                            NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));
    Train::GraphOptions options;
    options.dtype = DType::Int32;
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, *objective, Shape{K_BATCH, K_LENGTH}, options));

    EXPECT_EQ(built.graph.value(built.spec.loss).shape.num_elements(), 1);
    EXPECT_EQ(built.graph.value(built.spec.targets).dtype, DType::Int32);
    EXPECT_EQ(count_nodes(built.graph, Ops::K_NLL_LOSS.id), 0u);
    EXPECT_EQ(count_nodes(built.graph, Ops::K_LOG_SOFTMAX.id), 1u);

    // One gather reads the table by token, the other reads the log-probabilities by label.
    EXPECT_EQ(count_nodes(built.graph, Ops::K_GATHER_ROWS.id), 2u);
    EXPECT_EQ(count_nodes(built.graph, Ops::K_SCATTER_ADD_ROWS.id), 2u);
    EXPECT_EQ(count_nodes(built.graph, Ops::K_LOG_SOFTMAX_BACKWARD.id), 1u);

    ASSERT_EQ(built.parameters.size(), 1u);
    EXPECT_NE(built.parameters.front().gradient, Graph::K_NO_VALUE);
    EXPECT_EQ(built.parameters.front().role, NN::ParameterRole::Embedding);
}

/** @brief Tests that the factory reads N, T and V from the model's one output. */
TEST(Unit_LanguageTrainingNextTokenObjective, TheFactoryReadsTheModelsOutput)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        make_next_token_objective(make_context(), no_keys(), "objective"));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    objective->declare(builder, leaves);
    EXPECT_EQ(graph.value(leaves.at("targets")).shape, (Shape{K_BATCH * K_LENGTH}));
    EXPECT_EQ(objective->name(), "next_token");
}

/** @brief Tests each description the factory refuses. */
TEST(Unit_LanguageTrainingNextTokenObjective, TheFactoryRefusesWhatDoesNotFit)
{
    SA_EXPECT_FAILURE(
        make_next_token_objective(make_context(),
                                  Config::parse_json(R"({"smoothing": 0.1})", "<test>"),
                                  "objective"),
        StatusCode::InvalidDescription, "objective states the key 'smoothing'");

    Train::ObjectiveContext two_outputs = make_context();
    two_outputs.outputs.push_back(Train::OutputShape{"aux", Shape{K_BATCH, 1}});
    SA_EXPECT_FAILURE(make_next_token_objective(two_outputs, no_keys(), "objective"),
                      StatusCode::InvalidDescription, "one [N, T, V] output", "publishes 2");

    Train::ObjectiveContext flat = make_context();
    flat.outputs.front().shape = Shape{K_BATCH, K_VOCABULARY};
    SA_EXPECT_FAILURE(make_next_token_objective(flat, no_keys(), "objective"),
                      StatusCode::InvalidDescription, "[2, 3, V] logits", "'out'");

    Train::ObjectiveContext images = make_context();
    images.input_sample = Shape{4, 4, 3};
    SA_EXPECT_FAILURE(make_next_token_objective(images, no_keys(), "objective"),
                      StatusCode::InvalidDescription, "[T] token ids per sample");

    Train::ObjectiveContext wide_data = make_context();
    wide_data.classes = K_VOCABULARY + 1;
    SA_EXPECT_FAILURE(make_next_token_objective(wide_data, no_keys(), "objective"),
                      StatusCode::InvalidDescription, "6 token ids", "scores 5");
}

/** @brief Tests that the objective states a batch alone feeds it, so a sum may hold it. */
TEST(Unit_LanguageTrainingNextTokenObjective, ABatchAloneFeedsTheObjective)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenObjective> objective,
                            NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));

    EXPECT_TRUE(objective->is_fed_by_batch());
}

/** @brief Tests that the objective measures the perplexity of the loss a step computed. */
TEST(Unit_LanguageTrainingNextTokenObjective, MeasureReportsThePerplexityOfTheLoss)
{
    // A uniform guess over 8 tokens costs ln(8) nats and has a perplexity of 8.
    constexpr double K_RELATIVE_TOLERANCE = 1e-12;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenObjective> objective,
                            NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));
    const Data::Batch batch{};
    const Train::StepView step{batch, std::log(8.0), {}};

    Train::Metrics metrics;
    SA_EXPECT_OK(objective->measure(step, metrics));

    ASSERT_EQ(metrics.all().size(), 1u);
    EXPECT_EQ(metrics.all()[0].name, "perplexity");
    EXPECT_FALSE(metrics.all()[0].is_fraction);
    EXPECT_NEAR(metrics.all()[0].value, 8.0, 8.0 * K_RELATIVE_TOLERANCE);
}

/** @brief Tests that "output" names the logits of a model that publishes several values. */
TEST(Unit_LanguageTrainingNextTokenObjective, TheOutputKeyNamesTheLogitsAmongSeveral)
{
    Train::ObjectiveContext several = make_context();
    several.outputs.push_back(Train::OutputShape{"balance", Shape{1, 1}});

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        make_next_token_objective(several, Config::parse_json(R"({"output": "out"})", "<test>"),
                                  "objective"));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("balance", builder.input(Shape{1, 1}, "balance")));
    SA_EXPECT_OK(outputs.add(
        "out", builder.input(Shape{K_BATCH, K_LENGTH, K_VOCABULARY}, "logits")));
    Train::ObjectiveLeaves leaves;
    objective->declare(builder, leaves);

    const Graph::ValueID loss = objective->loss(builder, graph, outputs, leaves);

    EXPECT_EQ(graph.value(loss).shape.num_elements(), 1);
    EXPECT_EQ(count_nodes(graph, Ops::K_LOG_SOFTMAX.id), 1u);
}

/** @brief Tests each form of "output" the factory refuses. */
TEST(Unit_LanguageTrainingNextTokenObjective, TheFactoryRefusesAnOutputItCannotFind)
{
    Train::ObjectiveContext several = make_context();
    several.outputs.push_back(Train::OutputShape{"balance", Shape{1, 1}});
    const auto make = [&several](const char* text)
    {
        return make_next_token_objective(several, Config::parse_json(text, "<test>"),
                                         "objective");
    };

    SA_EXPECT_FAILURE(make("{}"), StatusCode::InvalidDescription, "publishes 2",
                      "name the logits with 'output'");
    SA_EXPECT_FAILURE(make(R"({"output": "logits"})"), StatusCode::InvalidDescription,
                      "the output 'logits'", "none of that name among its 2");
    SA_EXPECT_FAILURE(make(R"({"output": 3})"), StatusCode::InvalidDescription,
                      "'output' names a published output");
    SA_EXPECT_FAILURE(make(R"({"output": ""})"), StatusCode::InvalidDescription,
                      "'output'", "is empty");
    SA_EXPECT_FAILURE(make(R"({"output": "balance"})"), StatusCode::InvalidDescription,
                      "[2, 3, V] logits", "'balance'", "[1, 1]");
}
