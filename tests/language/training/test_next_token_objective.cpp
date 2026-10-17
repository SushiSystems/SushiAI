/**************************************************************************/
/* test_next_token_objective.cpp                                          */
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
 * @file test_next_token_objective.cpp
 * @brief Tests the next-token objective: its graph, its loss and gradient on an engine, and
 *        decoders that overfit through it.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/language_head.hpp>
#include <SushiAI/language/blocks/position_embedding.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/language/datasets/next_token_collator.hpp>
#include <SushiAI/language/datasets/token_dataset.hpp>
#include <SushiAI/language/datasets/token_file.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/training/next_token_objective.hpp>
#include <SushiAI/language/training/next_token_targets.hpp>
#include <SushiAI/language/training/perplexity.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/feed.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>

using namespace SushiAI;
using Language::Training::make_next_token_objective;
using Language::Training::NextTokenObjective;
// SaTest stays reachable inside a TEST_F body; see tests/language/training/README.md.
namespace SaTest = SushiAI::Test;

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
TEST(Medium_LanguageTrainingNextTokenObjective, MakeKeepsItsExtentsAndRefusesNone)
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
TEST(Medium_LanguageTrainingNextTokenObjective, MakeRefusesWhatInt32CannotIndex)
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
TEST(Medium_LanguageTrainingNextTokenObjective, ItDeclaresAnIndexLeafAndReadsLabels)
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
TEST(Medium_LanguageTrainingNextTokenObjective, TheLossGathersOneLogProbabilityPerPosition)
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
TEST(Medium_LanguageTrainingNextTokenObjective, TheFactoryReadsTheModelsOutput)
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
TEST(Medium_LanguageTrainingNextTokenObjective, TheFactoryRefusesWhatDoesNotFit)
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
TEST(Medium_LanguageTrainingNextTokenObjective, ABatchAloneFeedsTheObjective)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NextTokenObjective> objective,
                            NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));

    EXPECT_TRUE(objective->is_fed_by_batch());
}

/** @brief Tests that the objective measures the perplexity of the loss a step computed. */
TEST(Medium_LanguageTrainingNextTokenObjective, MeasureReportsThePerplexityOfTheLoss)
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
TEST(Medium_LanguageTrainingNextTokenObjective, TheOutputKeyNamesTheLogitsAmongSeveral)
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
TEST(Medium_LanguageTrainingNextTokenObjective, TheFactoryRefusesAnOutputItCannotFind)
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

namespace
{

    /** @brief Bounds the absolute error of a float32 loss or gradient of order one. */
    constexpr double K_TOLERANCE = 1e-5;

    /** @brief Returns the token of each of the N * T positions, row after row. */
    std::vector<std::int32_t> token_ids()
    {
        return {0, 3, 3, 1, 4, 2};
    }

    /** @brief Returns the label of each position; token 3 occurs twice with two labels. */
    std::vector<std::int32_t> label_ids()
    {
        return {3, 3, 1, 4, 2, 0};
    }

    /** @brief Returns the [V, V] table whose row r holds 0.3 r - 0.2 c + 0.1 r c. */
    std::vector<double> make_table()
    {
        std::vector<double> table;
        table.reserve(static_cast<std::size_t>(K_VOCABULARY * K_VOCABULARY));
        for (std::int64_t row = 0; row < K_VOCABULARY; ++row)
        {
            for (std::int64_t column = 0; column < K_VOCABULARY; ++column)
            {
                table.push_back(0.3 * static_cast<double>(row)
                                - 0.2 * static_cast<double>(column)
                                + 0.1 * static_cast<double>(row * column));
            }
        }
        return table;
    }

    /** @brief Holds the loss and its gradient with respect to the table, worked on the host. */
    struct Expected
    {
        double loss = 0.0;            /**< Mean of -log p(label) over the positions. */
        std::vector<double> gradient; /**< [V, V]: (softmax - one_hot) / (N T), summed by row. */
    };

    /** @brief Returns the loss and the gradient of a model whose logits are a table row. */
    Expected work_on_host(const std::vector<double>& table)
    {
        const std::size_t width = static_cast<std::size_t>(K_VOCABULARY);
        const std::vector<std::int32_t> tokens = token_ids();
        const std::vector<std::int32_t> labels = label_ids();
        const double positions = static_cast<double>(tokens.size());

        Expected expected;
        expected.gradient.assign(width * width, 0.0);
        for (std::size_t position = 0; position < tokens.size(); ++position)
        {
            const std::size_t row = static_cast<std::size_t>(tokens[position]);
            const std::size_t label = static_cast<std::size_t>(labels[position]);

            double total = 0.0;
            for (std::size_t column = 0; column < width; ++column)
                total += std::exp(table[row * width + column]);

            expected.loss -= (table[row * width + label] - std::log(total)) / positions;
            for (std::size_t column = 0; column < width; ++column)
            {
                const double probability = std::exp(table[row * width + column]) / total;
                const double hit = column == label ? 1.0 : 0.0;
                expected.gradient[row * width + column] += (probability - hit) / positions;
            }
        }
        return expected;
    }

    /** @brief Provides the engine the state and the step are built on. */
    class Medium_NextTokenObjective : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests the loss and the table's gradient of one step against the host's numbers. */
TEST_F(Medium_NextTokenObjective, TheLossAndItsGradientEqualTheHostComputation)
{
    NN::Embedding model{};
    model.num_embeddings = K_VOCABULARY;
    model.embedding_dim = K_VOCABULARY;

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Language::Training::NextTokenObjective> objective,
        Language::Training::NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));
    Train::GraphOptions options;
    options.dtype = DType::Int32;
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_training_graph(model, *objective,
                                                        Shape{K_BATCH, K_LENGTH}, options));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    ASSERT_EQ(slots.size(), 1u);
    const SushiBLAS::Tensor gradient = slots.front().gradient;

    const std::vector<double> table = make_table();
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* const weight, state.tensor("weight"));
    SaTest::write_tensor(*weight, SushiRuntime::span<const double>(table.data(), table.size()));

    // A rate of zero leaves the table as written, so the step can be read after it ran.
    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, sgd));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::TrainingStep> step,
                            Train::TrainingStep::make(*engine, built.graph, pool, *optimizer,
                                                      {}));

    const std::vector<std::int32_t> tokens = token_ids();
    const std::vector<std::int32_t> labels = label_ids();
    std::int32_t* const inputs = pool.tensor(built.spec.inputs).data_as<std::int32_t>();
    for (std::size_t i = 0; i < tokens.size(); ++i)
        inputs[i] = tokens[i];

    SA_ASSERT_OK_AND_ASSIGN(
        SushiBLAS::Tensor label_tensor,
        allocate_tensor(*engine, Shape{K_BATCH, K_LENGTH}, DType::Int32));
    for (std::size_t i = 0; i < labels.size(); ++i)
        label_tensor.data_as<std::int32_t>()[i] = labels[i];
    SA_EXPECT_OK(Language::Training::fill_next_token_targets(
        label_tensor, K_VOCABULARY, pool.tensor(built.spec.targets)));

    SA_EXPECT_OK(step->run());

    const Expected expected = work_on_host(table);
    EXPECT_NEAR(static_cast<double>(pool.tensor(built.spec.loss).data_as<float>()[0]),
                expected.loss, K_TOLERANCE);

    const std::vector<double> found = SaTest::read_tensor(gradient);
    ASSERT_EQ(found.size(), expected.gradient.size());
    for (std::size_t i = 0; i < found.size(); ++i)
        EXPECT_NEAR(found[i], expected.gradient[i], K_TOLERANCE) << "element " << i;
}

namespace
{

    /** @brief Holds the passes fit() makes over the one batch. */
    constexpr std::int64_t K_EPOCHS = 30;

    /** @brief Bounds the difference between exp of a float32 loss and the reported perplexity. */
    constexpr double K_TRAINER_TOLERANCE = 1e-9;

    /** @brief Returns the description of [T] int32 ids per sample. */
    Data::InputDescription token_input()
    {
        return Data::InputDescription{Shape{K_LENGTH}, DType::Int32};
    }

    /** @brief Returns the labels role as [T] int32 ids per sample over @p vocabulary tokens. */
    std::vector<Data::TargetDescription> label_roles(std::int64_t vocabulary)
    {
        return {Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{K_LENGTH},
                                        DType::Int32, vocabulary}};
    }

    /** @brief Returns a dataset of one batch in which each token is followed by one label. */
    Result<Data::MaterialisedDataset> make_dataset(SushiBLAS::Engine& engine,
                                                   const std::vector<std::int32_t>& labels)
    {
        SA_TRY_ASSIGN(Data::Batch batch,
                      Data::allocate_batch(engine, K_BATCH, token_input(),
                                           label_roles(K_VOCABULARY)));

        const std::vector<std::int32_t> tokens = {0, 3, 1, 1, 4, 2};
        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            batch.inputs.data_as<std::int32_t>()[i] = tokens[i];
            batch.targets.front().values.data_as<std::int32_t>()[i] = labels[i];
        }

        std::vector<Data::Batch> batches;
        batches.push_back(std::move(batch));
        return Data::MaterialisedDataset(std::move(batches), K_BATCH, token_input(),
                                         label_roles(K_VOCABULARY), "tokens");
    }

    /** @brief Provides the engine the state, the dataset and the step are built on. */
    class Medium_NextTokenTrainer : public SaTest::EngineTest
    {
        protected:
            /**
             * @brief Fits a bigram table to @p labels for K_EPOCHS passes.
             * @return The report of the run, or the first failure.
             */
            [[nodiscard]] Result<Train::RunReport> fit_bigram(
                const std::vector<std::int32_t>& labels)
            {
                NN::Embedding model{};
                model.num_embeddings = K_VOCABULARY;
                model.embedding_dim = K_VOCABULARY;

                SA_TRY_ASSIGN(
                    const std::unique_ptr<Language::Training::NextTokenObjective> objective,
                    Language::Training::NextTokenObjective::make(K_BATCH, K_LENGTH,
                                                                 K_VOCABULARY));
                Train::GraphOptions graph_options;
                graph_options.dtype = DType::Int32;
                SA_TRY_ASSIGN(Train::TrainingGraph built,
                              Train::build_training_graph(model, *objective,
                                                          Shape{K_BATCH, K_LENGTH},
                                                          graph_options));

                SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
                Graph::TensorPool pool(*engine);
                SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                              Train::bind_parameters(built, state, pool));
                SA_TRY(NN::initialize(*engine, state));

                Optim::SGDOptions sgd;
                sgd.learning_rate = 0.5;
                SA_TRY_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                              Optim::SGD::make(*engine, slots, sgd));

                Train::TrainerOptions options;
                options.epochs = K_EPOCHS;
                SA_TRY_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                              Train::Trainer::make(*engine, built, pool, *objective, *optimizer,
                                                   options));

                SA_TRY_ASSIGN(const Data::MaterialisedDataset dataset,
                              make_dataset(*engine, labels));
                return trainer->fit(dataset);
            }
    };
} // namespace

/** @brief Tests that fit lowers the next-token loss and reports its perplexity. */
TEST_F(Medium_NextTokenTrainer, FitLowersTheLossAndReportsPerplexity)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, fit_bigram({3, 1, 1, 4, 2, 0}));

    EXPECT_EQ(report.steps, static_cast<std::size_t>(K_EPOCHS));
    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_LT(report.last_loss, report.first_loss) << report.to_string();

    ASSERT_EQ(report.last_metrics.all().size(), 1u);
    const Train::Metric* const perplexity = report.last_metrics.find("perplexity");
    ASSERT_NE(perplexity, nullptr);
    EXPECT_FALSE(perplexity->is_fraction);
    EXPECT_NEAR(perplexity->value, std::exp(report.last_loss), K_TRAINER_TOLERANCE);
    EXPECT_NE(report.to_string().find("perplexity"), std::string::npos);
}

/** @brief Tests that a label outside the vocabulary stops the first step and names its row. */
TEST_F(Medium_NextTokenTrainer, ALabelOutsideTheVocabularyStopsTheFirstStep)
{
    SA_EXPECT_FAILURE(fit_bigram({3, 1, 1, 9, 2, 0}), StatusCode::Incompatible, "position 3",
                      "is 9", "holds 5 tokens");
}

namespace
{
    /** @brief Holds the tokens the model scores, V. */
    constexpr std::int64_t K_OVERFIT_VOCABULARY = 64;

    /** @brief Holds the rows of the one batch, N. */
    constexpr std::int64_t K_OVERFIT_BATCH = 4;

    /** @brief Holds the positions of a row, T. */
    constexpr std::int64_t K_OVERFIT_LENGTH = 16;

    /** @brief Holds the width of the model. */
    constexpr std::int64_t K_DIM = 32;

    /** @brief Holds the heads of each attention. */
    constexpr std::int64_t K_HEADS = 2;

    /** @brief Holds the width between the two linear layers of a feed-forward. */
    constexpr std::int64_t K_HIDDEN = 64;

    /** @brief Holds the updates the one batch is trained for. */
    constexpr std::int64_t K_STEPS = 400;

    /** @brief Holds the fraction of the first loss the last one must be below. */
    constexpr double K_REMAINING_FRACTION = 0.1;

    /** @brief Bounds how far the first loss may lie from ln(V), the loss of a uniform guess. */
    constexpr double K_FIRST_LOSS_TOLERANCE = 0.5;

    /** @brief Holds the norm every update's gradients are clipped to. */
    constexpr double K_CLIP_NORM = 1.0;

    /** @brief Holds a decoder of two blocks whose head scores with the token table. */
    struct Decoder
    {
        /** @brief Holds the [V, dim] token table, which the head is tied to. */
        NN::Embedding tokens{};

        /** @brief Holds the learned position table. */
        Language::Blocks::PositionEmbedding positions;

        /** @brief Holds the first block. */
        Language::Blocks::TransformerBlock first;

        /** @brief Holds the second block. */
        Language::Blocks::TransformerBlock second;

        /** @brief Holds the final normalisation and the tied scores. */
        Language::Blocks::LanguageHead head;

        /** @brief Builds the five children and ties the head to the token table. */
        Decoder()
            : positions(K_OVERFIT_LENGTH, K_DIM),
              first(K_DIM, K_HEADS, K_HIDDEN, Language::Blocks::AttentionMask::Causal,
                    NN::K_LAYER_NORM_EPSILON),
              second(K_DIM, K_HEADS, K_HIDDEN, Language::Blocks::AttentionMask::Causal,
                     NN::K_LAYER_NORM_EPSILON),
              head(K_DIM, K_OVERFIT_VOCABULARY, NN::K_LAYER_NORM_EPSILON)
        {
            tokens.num_embeddings = K_OVERFIT_VOCABULARY;
            tokens.embedding_dim = K_DIM;
            head.tie_to(tokens.weight);
        }

        /** @brief Forbids a copy, whose head would stay tied to this object's table. */
        Decoder(const Decoder&) = delete;

        /** @brief Forbids copy assignment, for the same reason. */
        Decoder& operator=(const Decoder&) = delete;

        /** @brief Lists the five children in the order they are traced. */
        static constexpr auto fields()
        {
            return std::tuple{
                NN::child("tokens", &Decoder::tokens),
                NN::child("positions", &Decoder::positions),
                NN::child("first", &Decoder::first),
                NN::child("second", &Decoder::second),
                NN::child("head", &Decoder::head),
            };
        }

        /** @brief Traces [N, T] ids to [N, T, V] logits. */
        Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID ids)
        {
            Graph::ValueID x = NN::trace_layer(tokens, trace, ids);
            x = NN::trace_layer(positions, trace, x);
            x = NN::trace_layer(first, trace, x);
            x = NN::trace_layer(second, trace, x);
            return NN::trace_layer(head, trace, x);
        }
    };

    /** @brief Owns one file under the system's temporary directory and removes it. */
    class ScratchFile final
    {
        public:
            /** @brief Names the file; nothing is written yet. */
            explicit ScratchFile(const char* name)
                : path_(std::filesystem::temp_directory_path() / name)
            {

            }

            /** @brief Removes the file when it exists. */
            ~ScratchFile()
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }

            ScratchFile(const ScratchFile&) = delete;
            ScratchFile& operator=(const ScratchFile&) = delete;
            ScratchFile(ScratchFile&&) = delete;
            ScratchFile& operator=(ScratchFile&&) = delete;

            /** @brief Returns the path of the file. */
            [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

        private:
            /// @brief Holds the path this object removes.
            std::filesystem::path path_;
    };

    /** @brief Returns K_OVERFIT_BATCH windows of K_OVERFIT_LENGTH + 1 ids, id i being 7 i^2 + 3 i + 1 mod V. */
    std::vector<Language::Text::TokenID> make_ids()
    {
        const std::int64_t count = K_OVERFIT_BATCH * (K_OVERFIT_LENGTH + 1);
        std::vector<Language::Text::TokenID> ids;
        ids.reserve(static_cast<std::size_t>(count));
        for (std::int64_t i = 0; i < count; ++i)
            ids.push_back(static_cast<Language::Text::TokenID>((7 * i * i + 3 * i + 1)
                                                               % K_OVERFIT_VOCABULARY));
        return ids;
    }

    /** @brief Returns whether the slot named @p name decays; fails the test when none has it. */
    bool decays(const std::vector<Optim::ParameterSlot>& slots, const std::string& name)
    {
        for (const Optim::ParameterSlot& slot : slots)
        {
            if (slot.name == name)
                return slot.decays;
        }

        ADD_FAILURE() << "no slot is named " << name;
        return false;
    }

    /** @brief Provides the engine the dataset, the state and the step are built on. */
    class Medium_DecoderOverfit : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that the loss of one fixed batch falls below a tenth of its start. */
TEST_F(Medium_DecoderOverfit, ATwoBlockDecoderOverfitsOneBatch)
{
    const ScratchFile file("sushiai_decoder_overfit.satk");
    const std::vector<Language::Text::TokenID> ids = make_ids();
    SA_EXPECT_OK(Language::Datasets::write_token_file(
        file.path(),
        SushiRuntime::span<const Language::Text::TokenID>(ids.data(), ids.size()),
        static_cast<std::int32_t>(K_OVERFIT_VOCABULARY)));

    SA_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Language::Datasets::NextTokenCollator> collator,
        Language::Datasets::NextTokenCollator::make(K_OVERFIT_LENGTH,
                                                    static_cast<std::int32_t>(K_OVERFIT_VOCABULARY)));
    Language::Datasets::TokenDatasetOptions dataset_options;
    dataset_options.batch_size = K_OVERFIT_BATCH;
    dataset_options.stride = K_OVERFIT_LENGTH + 1;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Language::Datasets::TokenDataset> dataset,
        Language::Datasets::TokenDataset::make(*engine, file.path(), std::move(collator),
                                               dataset_options));
    ASSERT_EQ(dataset->batch_count(), 1u);

    Decoder model;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Language::Training::NextTokenObjective> objective,
        Language::Training::NextTokenObjective::make(K_OVERFIT_BATCH, K_OVERFIT_LENGTH, K_OVERFIT_VOCABULARY));
    Train::GraphOptions graph_options;
    graph_options.dtype = DType::Int32;
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_training_graph(model, *objective,
                                                        Shape{K_OVERFIT_BATCH, K_OVERFIT_LENGTH},
                                                        graph_options));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));

    // The head has no weight of its own, and only the linear weights decay.
    SA_EXPECT_FAILURE(state.tensor("head.linear.weight"), StatusCode::InvalidDescription,
                      "head.linear.weight");
    EXPECT_FALSE(decays(slots, "tokens.weight"));
    EXPECT_FALSE(decays(slots, "positions.weight"));
    EXPECT_FALSE(decays(slots, "head.norm.weight"));
    EXPECT_FALSE(decays(slots, "first.attn.qkv.bias"));
    EXPECT_TRUE(decays(slots, "first.attn.qkv.weight"));

    NN::InitPolicy policy;
    policy.family = NN::InitFamily::Xavier;
    policy.nonlinearity = NN::Nonlinearity::Linear;
    policy.zero_final_layer = false;
    SA_EXPECT_OK(NN::initialize(*engine, state, policy));

    Optim::AdamWOptions optimizer_options;
    optimizer_options.weight_decay = 0.01;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, optimizer_options));
    SA_EXPECT_OK(optimizer->enable_gradient_clip(K_CLIP_NORM));

    Optim::CosineSchedule schedule;
    schedule.base_rate = 3e-3;
    schedule.warmup_iterations = 20;
    schedule.total_iterations = K_STEPS;
    schedule.final_rate = 3e-4;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::TrainingStep> step,
                            Train::TrainingStep::make(*engine, built.graph, pool, *optimizer,
                                                      {}));

    // The one batch is fed once: no task is recorded yet, so the host may write the leaves.
    Data::Batch batch;
    SA_EXPECT_OK(dataset->read(0, batch));
    SushiBLAS::Tensor& inputs = pool.tensor(built.spec.inputs);
    SA_EXPECT_OK(Train::check_feed(batch.inputs, inputs, "inputs"));
    const std::int32_t* const fed = batch.inputs.data_as<std::int32_t>();
    std::int32_t* const leaf = inputs.data_as<std::int32_t>();
    for (std::int64_t i = 0; i < inputs.num_elements; ++i)
        leaf[i] = fed[i];

    const Data::BatchTarget* const labels =
        Data::find_target(batch.targets, Data::K_ROLE_LABELS);
    ASSERT_NE(labels, nullptr);
    SA_EXPECT_OK(Language::Training::fill_next_token_targets(
        labels->values, K_OVERFIT_VOCABULARY, pool.tensor(built.spec.targets)));

    double first_loss = 0.0;
    double last_loss = 0.0;
    for (std::int64_t update = 0; update < K_STEPS; ++update)
    {
        Optim::apply_step_schedule(schedule, *optimizer);
        SA_EXPECT_OK(step->run());

        last_loss = static_cast<double>(pool.tensor(built.spec.loss).data_as<float>()[0]);
        ASSERT_TRUE(std::isfinite(last_loss)) << "update " << update;
        if (update == 0)
            first_loss = last_loss;
    }

    EXPECT_NEAR(first_loss, std::log(static_cast<double>(K_OVERFIT_VOCABULARY)),
                K_FIRST_LOSS_TOLERANCE);
    EXPECT_LT(last_loss, K_REMAINING_FRACTION * first_loss);
    EXPECT_LT(Language::Training::perplexity(last_loss),
              Language::Training::perplexity(first_loss));
    EXPECT_EQ(step->step().compile_count(), 1u);
    EXPECT_EQ(optimizer->step_count(), static_cast<std::uint64_t>(K_STEPS));
}

namespace
{
    /** @brief Holds the host tables of the graph: the cosines and the sines of each block. */
    constexpr std::size_t K_HOST_TABLES = 4;

    /** @brief Holds a decoder of two rotary blocks whose head scores with the token table. */
    struct RotaryDecoder
    {
        /** @brief Holds the [V, dim] token table, which the head is tied to. */
        NN::Embedding tokens{};

        /** @brief Holds the first block. */
        Language::Blocks::TransformerBlock first;

        /** @brief Holds the second block. */
        Language::Blocks::TransformerBlock second;

        /** @brief Holds the final normalisation and the tied scores. */
        Language::Blocks::LanguageHead head;

        /** @brief Builds the four children with rotary attentions and ties the head. */
        RotaryDecoder()
            : first(K_DIM, K_HEADS, K_HIDDEN, Language::Blocks::AttentionMask::Causal,
                    NN::K_LAYER_NORM_EPSILON, Language::Blocks::AttentionPositions::Rotary),
              second(K_DIM, K_HEADS, K_HIDDEN, Language::Blocks::AttentionMask::Causal,
                     NN::K_LAYER_NORM_EPSILON, Language::Blocks::AttentionPositions::Rotary),
              head(K_DIM, K_OVERFIT_VOCABULARY, NN::K_LAYER_NORM_EPSILON)
        {
            tokens.num_embeddings = K_OVERFIT_VOCABULARY;
            tokens.embedding_dim = K_DIM;
            head.tie_to(tokens.weight);
        }

        /** @brief Forbids a copy, whose head would stay tied to this object's table. */
        RotaryDecoder(const RotaryDecoder&) = delete;

        /** @brief Forbids copy assignment, for the same reason. */
        RotaryDecoder& operator=(const RotaryDecoder&) = delete;

        /** @brief Lists the four children in the order they are traced. */
        static constexpr auto fields()
        {
            return std::tuple{
                NN::child("tokens", &RotaryDecoder::tokens),
                NN::child("first", &RotaryDecoder::first),
                NN::child("second", &RotaryDecoder::second),
                NN::child("head", &RotaryDecoder::head),
            };
        }

        /** @brief Traces [N, T] ids to [N, T, V] logits. */
        Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID ids)
        {
            Graph::ValueID x = NN::trace_layer(tokens, trace, ids);
            x = NN::trace_layer(first, trace, x);
            x = NN::trace_layer(second, trace, x);
            return NN::trace_layer(head, trace, x);
        }
    };

    /** @brief Provides the engine the dataset, the state and the step are built on. */
    class Medium_RotaryDecoderOverfit : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that the loss of one fixed batch falls below a tenth of its start. */
TEST_F(Medium_RotaryDecoderOverfit, ATwoBlockRotaryDecoderOverfitsOneBatch)
{
    const ScratchFile file("sushiai_rotary_decoder_overfit.satk");
    const std::vector<Language::Text::TokenID> ids = make_ids();
    SA_EXPECT_OK(Language::Datasets::write_token_file(
        file.path(),
        SushiRuntime::span<const Language::Text::TokenID>(ids.data(), ids.size()),
        static_cast<std::int32_t>(K_OVERFIT_VOCABULARY)));

    SA_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Language::Datasets::NextTokenCollator> collator,
        Language::Datasets::NextTokenCollator::make(K_OVERFIT_LENGTH,
                                                    static_cast<std::int32_t>(K_OVERFIT_VOCABULARY)));
    Language::Datasets::TokenDatasetOptions dataset_options;
    dataset_options.batch_size = K_OVERFIT_BATCH;
    dataset_options.stride = K_OVERFIT_LENGTH + 1;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Language::Datasets::TokenDataset> dataset,
        Language::Datasets::TokenDataset::make(*engine, file.path(), std::move(collator),
                                               dataset_options));
    ASSERT_EQ(dataset->batch_count(), 1u);

    RotaryDecoder model;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Language::Training::NextTokenObjective> objective,
        Language::Training::NextTokenObjective::make(K_OVERFIT_BATCH, K_OVERFIT_LENGTH, K_OVERFIT_VOCABULARY));
    Train::GraphOptions graph_options;
    graph_options.dtype = DType::Int32;
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_training_graph(model, *objective,
                                                        Shape{K_OVERFIT_BATCH, K_OVERFIT_LENGTH},
                                                        graph_options));
    EXPECT_EQ(built.tables.tables().size(), K_HOST_TABLES);

    // The angles are leaves the step fills, so the state holds the parameters and no table.
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    for (const NN::StateEntry& entry : state.entries())
        EXPECT_EQ(entry.kind, NN::StateKind::Parameter) << entry.name;

    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    EXPECT_EQ(slots.size(), state.entries().size());

    NN::InitPolicy policy;
    policy.family = NN::InitFamily::Xavier;
    policy.nonlinearity = NN::Nonlinearity::Linear;
    policy.zero_final_layer = false;
    SA_EXPECT_OK(NN::initialize(*engine, state, policy));

    Optim::AdamWOptions optimizer_options;
    optimizer_options.weight_decay = 0.01;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, optimizer_options));
    SA_EXPECT_OK(optimizer->enable_gradient_clip(K_CLIP_NORM));

    Optim::CosineSchedule schedule;
    schedule.base_rate = 3e-3;
    schedule.warmup_iterations = 20;
    schedule.total_iterations = K_STEPS;
    schedule.final_rate = 3e-4;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::TrainingStep> step,
                            Train::TrainingStep::make(*engine, built.graph, pool, *optimizer,
                                                      {}));

    // The one batch is fed once: no task is recorded yet, so the host may write the leaves.
    Data::Batch batch;
    SA_EXPECT_OK(dataset->read(0, batch));
    SushiBLAS::Tensor& inputs = pool.tensor(built.spec.inputs);
    SA_EXPECT_OK(Train::check_feed(batch.inputs, inputs, "inputs"));
    const std::int32_t* const fed = batch.inputs.data_as<std::int32_t>();
    std::int32_t* const leaf = inputs.data_as<std::int32_t>();
    for (std::int64_t i = 0; i < inputs.num_elements; ++i)
        leaf[i] = fed[i];

    const Data::BatchTarget* const labels =
        Data::find_target(batch.targets, Data::K_ROLE_LABELS);
    ASSERT_NE(labels, nullptr);
    SA_EXPECT_OK(Language::Training::fill_next_token_targets(
        labels->values, K_OVERFIT_VOCABULARY, pool.tensor(built.spec.targets)));

    double first_loss = 0.0;
    double last_loss = 0.0;
    for (std::int64_t update = 0; update < K_STEPS; ++update)
    {
        Optim::apply_step_schedule(schedule, *optimizer);
        SA_EXPECT_OK(step->run());

        last_loss = static_cast<double>(pool.tensor(built.spec.loss).data_as<float>()[0]);
        ASSERT_TRUE(std::isfinite(last_loss)) << "update " << update;
        if (update == 0)
            first_loss = last_loss;
    }

    EXPECT_NEAR(first_loss, std::log(static_cast<double>(K_OVERFIT_VOCABULARY)),
                K_FIRST_LOSS_TOLERANCE);
    EXPECT_LT(last_loss, K_REMAINING_FRACTION * first_loss);
    EXPECT_EQ(step->step().compile_count(), 1u);
}
