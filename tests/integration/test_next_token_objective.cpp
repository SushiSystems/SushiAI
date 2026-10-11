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
 * @brief Tests the value and the gradient of the next-token loss against a host computation.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/training/next_token_objective.hpp>
#include <SushiAI/language/training/next_token_targets.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>
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
    class Integration_NextTokenObjective : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests the loss and the table's gradient of one step against the host's numbers. */
TEST_F(Integration_NextTokenObjective, TheLossAndItsGradientEqualTheHostComputation)
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
