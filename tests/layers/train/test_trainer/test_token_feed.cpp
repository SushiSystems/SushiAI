/**************************************************************************/
/* test_token_feed.cpp                                                    */
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
 * @file test_token_feed.cpp
 * @brief Tests that a dataset of int32 ids feeds an int32 input leaf and no float32 one.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/input_match.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine the batches, the states and the steps are built on. */
    class Integration_TokenFeed : public SaTest::EngineTest
    {

    };

    /** @brief The model the refusal case compiles a float32 input leaf for. */
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    /** @brief Rows of ids per batch. */
    constexpr int64_t K_BATCH = 2;

    /** @brief Ids per row, T. */
    constexpr int64_t K_LENGTH = 5;

    /** @brief Rows of the embedding table, V; every id is below it. */
    constexpr int64_t K_VOCABULARY = 7;

    /** @brief Values per row of the embedding table, D. */
    constexpr int64_t K_WIDTH = 4;

    /** @brief Hidden units of the MLP. */
    constexpr int64_t K_HIDDEN = 8;

    /** @brief Classes the MLP ends in. */
    constexpr int64_t K_CLASSES = 3;

    /** @brief Bounds the difference between a table row and its copy, which is exact. */
    constexpr double K_EXACT_TOLERANCE = 0.0;

    /** @brief Returns the ids of the one batch, row by row. */
    std::vector<std::int32_t> token_ids()
    {
        return {6, 0, 3, 3, 1, 2, 6, 5, 0, 4};
    }

    /** @brief Returns the input of a token batch: K_LENGTH int32 ids per row. */
    Data::InputDescription token_input()
    {
        return Data::InputDescription{Shape{K_LENGTH}, DType::Int32};
    }

    /** @brief Returns the labels role as one int32 per row over K_CLASSES classes. */
    std::vector<Data::TargetDescription> label_roles()
    {
        return {Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{},
                                        DType::Int32, K_CLASSES}};
    }

    /** @brief Returns a [K_VOCABULARY, K_WIDTH] table whose row r holds r + column / 8. */
    std::vector<double> dyadic_table()
    {
        std::vector<double> table;
        table.reserve(static_cast<std::size_t>(K_VOCABULARY * K_WIDTH));
        for (int64_t row = 0; row < K_VOCABULARY; ++row)
        {
            for (int64_t column = 0; column < K_WIDTH; ++column)
                table.push_back(static_cast<double>(row) + static_cast<double>(column) / 8.0);
        }

        return table;
    }

    /** @brief Returns a dataset of one batch: token_ids() with every label at class 0. */
    Result<Data::MaterialisedDataset> make_token_dataset(SushiBLAS::Engine& engine)
    {
        SA_TRY_ASSIGN(Data::Batch batch,
                      Data::allocate_batch(engine, K_BATCH, token_input(), label_roles()));

        const std::vector<std::int32_t> ids = token_ids();
        std::int32_t* const inputs = batch.inputs.data_as<std::int32_t>();
        for (std::size_t i = 0; i < ids.size(); ++i)
            inputs[i] = ids[i];

        std::int32_t* const labels = batch.targets.front().values.data_as<std::int32_t>();
        for (int64_t row = 0; row < K_BATCH; ++row)
            labels[row] = 0;

        std::vector<Data::Batch> batches;
        batches.push_back(std::move(batch));

        return Data::MaterialisedDataset(std::move(batches), K_BATCH, token_input(),
                                         label_roles(), "tokens");
    }
} // namespace

/** @brief Tests that an int32 [N, T] batch read from a dataset feeds an Embedding's leaf. */
TEST_F(Integration_TokenFeed, AnInt32BatchOfADatasetFeedsAnEmbedding)
{
    NN::Embedding model{};
    model.num_embeddings = K_VOCABULARY;
    model.embedding_dim = K_WIDTH;

    Train::GraphOptions options;
    options.dtype = DType::Int32;
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TracedStep traced,
        Train::build_evaluation_graph(model, Shape{K_BATCH, K_LENGTH}, options));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    ASSERT_EQ(state.entries().size(), 1u);

    const std::vector<double> table = dyadic_table();
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* const weight,
                            state.tensor(state.entries()[0].state));
    SaTest::write_tensor(*weight,
                         SushiRuntime::span<const double>(table.data(), table.size()));

    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(*engine, traced, state, pool));

    SA_ASSERT_OK_AND_ASSIGN(const Data::MaterialisedDataset dataset,
                            make_token_dataset(*engine));
    SA_ASSERT_OK_AND_ASSIGN(const Data::InputDescription required,
                            Train::describe_input_leaf(traced.graph.value(traced.input)));
    EXPECT_EQ(required.dtype, DType::Int32);
    SA_EXPECT_OK(Train::match_input(required, dataset));

    Data::Batch batch;
    SA_EXPECT_OK(dataset.read(0, batch));
    ASSERT_EQ(batch.inputs.dtype, DType::Int32);
    SA_EXPECT_OK(step->run(batch.inputs));

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const output,
                            step->output(NN::K_SINGLE_PORT_NAME));
    const std::vector<double> embedded = SaTest::read_tensor(*output);
    const std::vector<std::int32_t> ids = token_ids();
    const std::size_t width = static_cast<std::size_t>(K_WIDTH);

    ASSERT_EQ(embedded.size(), ids.size() * width);
    for (std::size_t position = 0; position < ids.size(); ++position)
    {
        const std::size_t row = static_cast<std::size_t>(ids[position]);
        for (std::size_t column = 0; column < width; ++column)
        {
            EXPECT_NEAR(embedded[position * width + column], table[row * width + column],
                        K_EXACT_TOLERANCE)
                << "position " << position << ", column " << column;
        }
    }
}

/** @brief Tests that fit refuses an int32 dataset for a float32 leaf before any step runs. */
TEST_F(Integration_TokenFeed, FitRefusesAnInt32DatasetForAFloat32Leaf)
{
    MLP model(NN::Linear{K_LENGTH, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_LENGTH, K_CLASSES));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, Optim::SGDOptions{}));
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built, pool, objective,
                                                 *optimizer));

    SA_ASSERT_OK_AND_ASSIGN(const Data::MaterialisedDataset dataset,
                            make_token_dataset(*engine));

    SA_EXPECT_FAILURE(trainer->fit(dataset), StatusCode::Incompatible, "'tokens'",
                      "as int32 [5] per sample", "compiled for float32 [5] per sample");
    EXPECT_EQ(trainer->step_count(), 0u);
}
