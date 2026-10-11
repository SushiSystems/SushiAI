/**************************************************************************/
/* test_next_token_trainer.cpp                                            */
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
 * @file test_next_token_trainer.cpp
 * @brief Tests that Train::Trainer trains the next-token objective through its own feed.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/training/next_token_objective.hpp>
#include <SushiAI/language/training/perplexity.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the rows of the batch, N. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the positions of a row, T. */
    constexpr std::int64_t K_LENGTH = 3;

    /** @brief Holds the vocabulary size, V, which is also the width of the table. */
    constexpr std::int64_t K_VOCABULARY = 5;

    /** @brief Holds the passes fit() makes over the one batch. */
    constexpr std::int64_t K_EPOCHS = 30;

    /** @brief Bounds the difference between exp of a float32 loss and the reported perplexity. */
    constexpr double K_TOLERANCE = 1e-9;

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
    class Integration_NextTokenTrainer : public SaTest::EngineTest
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
TEST_F(Integration_NextTokenTrainer, FitLowersTheLossAndReportsPerplexity)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, fit_bigram({3, 1, 1, 4, 2, 0}));

    EXPECT_EQ(report.steps, static_cast<std::size_t>(K_EPOCHS));
    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_LT(report.last_loss, report.first_loss) << report.to_string();

    ASSERT_EQ(report.last_metrics.all().size(), 1u);
    const Train::Metric* const perplexity = report.last_metrics.find("perplexity");
    ASSERT_NE(perplexity, nullptr);
    EXPECT_FALSE(perplexity->is_fraction);
    EXPECT_NEAR(perplexity->value, std::exp(report.last_loss), K_TOLERANCE);
    EXPECT_NE(report.to_string().find("perplexity"), std::string::npos);
}

/** @brief Tests that a label outside the vocabulary stops the first step and names its row. */
TEST_F(Integration_NextTokenTrainer, ALabelOutsideTheVocabularyStopsTheFirstStep)
{
    SA_EXPECT_FAILURE(fit_bigram({3, 1, 1, 9, 2, 0}), StatusCode::Incompatible, "position 3",
                      "is 9", "holds 5 tokens");
}
