/**************************************************************************/
/* test_evaluation_step.cpp                                               */
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
 * @file test_evaluation_step.cpp
 * @brief Tests the compiled forward-only step over an evaluation graph and a probe graph.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_EvaluationStep : public SaTest::EngineTest {};

    /** @brief The model the cases without a buffer run. */
    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    /** @brief Samples per step in every case of this file. */
    constexpr int64_t K_BATCH = 4;

    /** @brief Input features of the MLP. */
    constexpr int64_t K_FEATURES = 6;

    /** @brief Hidden units of the MLP. */
    constexpr int64_t K_HIDDEN = 8;

    /** @brief Classes the MLP ends in. */
    constexpr int64_t K_CLASSES = 3;

    /** @brief Channels of the normalised image. */
    constexpr int64_t K_CHANNELS = 3;

    /** @brief Height and width of the normalised image. */
    constexpr int64_t K_SIDE = 4;

    /** @brief The seed of every input and every dataset this file draws. */
    constexpr std::uint64_t K_INPUT_SEED = 0x5EED5EEDull;

    /** @brief Builds the MLP over K_FEATURES, K_HIDDEN and K_CLASSES. */
    Mlp make_mlp()
    {
        return Mlp(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    }

    /** @brief Allocates a float32 tensor of @p shape and fills it from the seeded stream. */
    Result<SushiBLAS::Tensor> noise_of(SushiBLAS::Engine& engine, const Shape& shape)
    {
        SA_TRY_ASSIGN(SushiBLAS::Tensor tensor, allocate_tensor(engine, shape));

        std::vector<double> values(static_cast<std::size_t>(tensor.num_elements));
        SaTest::HostNoise noise(K_INPUT_SEED);
        noise.fill(SushiRuntime::span<double>(values.data(), values.size()));
        SaTest::write_tensor(tensor,
                             SushiRuntime::span<const double>(values.data(), values.size()));

        return std::move(tensor);
    }

    /** @brief Runs @p step over @p inputs and returns the bytes of its single output. */
    Result<std::vector<unsigned char>> run_and_read(Train::EvaluationStep& step,
                                                    const SushiBLAS::Tensor& inputs)
    {
        SA_TRY(step.run(inputs));
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const output,
                      step.output(NN::K_SINGLE_PORT_NAME));

        return SaTest::read_tensor_bytes(*output);
    }

    /** @brief Returns the bytes of every buffer of @p state, in entry order. */
    Result<std::vector<std::vector<unsigned char>>> buffer_bytes(NN::ModelState& state)
    {
        std::vector<std::vector<unsigned char>> bytes;
        for (const NN::StateEntry& entry : state.entries())
        {
            if (entry.kind != NN::StateKind::Buffer)
                continue;

            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor, state.tensor(entry.state));
            bytes.push_back(SaTest::read_tensor_bytes(*tensor));
        }

        return bytes;
    }

} // namespace

/** @brief Tests that a step compiles in make() and not again over three runs. */
TEST_F(Integration_EvaluationStep, TheStepCompilesOnce)
{
    Mlp model = make_mlp();
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TracedStep traced,
        Train::build_evaluation_graph(model, Shape{K_BATCH, K_FEATURES}));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(*engine, traced, state, pool));
    EXPECT_EQ(step->step().compile_count(), 1u);

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs,
                            noise_of(*engine, Shape{K_BATCH, K_FEATURES}));
    for (int run = 0; run < 3; ++run)
        SA_EXPECT_OK(step->run(inputs));

    EXPECT_EQ(step->step().compile_count(), 1u);
}

/**
 * @brief Tests that an evaluation step over its own pool reads the weights a training step
 *        over another pool wrote into the shared state.
 */
TEST_F(Integration_EvaluationStep, AnEvaluationStepSeesAWeightTheTrainingStepJustMoved)
{
    Mlp model = make_mlp();
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TracedStep traced,
        Train::build_evaluation_graph(model, Shape{K_BATCH, K_FEATURES}));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    Graph::TensorPool training_pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, training_pool));

    // The default policy zeroes the classifier, so the first evaluation gives zero logits.
    SA_EXPECT_OK(NN::initialize(*engine, state));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(*engine, slots, Optim::SGDOptions{}));
    Train::TrainerOptions options;
    options.track_accuracy = false;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built.graph, training_pool,
                                                 built.spec, *optimizer, options));

    Graph::TensorPool evaluation_pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::EvaluationStep> step,
        Train::EvaluationStep::make(*engine, traced, state, evaluation_pool));

    Data::SyntheticOptions data_options;
    data_options.batch_size = K_BATCH;
    data_options.batches = 1;
    data_options.features = K_FEATURES;
    data_options.classes = K_CLASSES;
    data_options.seed = K_INPUT_SEED;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs,
                            noise_of(*engine, Shape{K_BATCH, K_FEATURES}));

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<unsigned char> before,
                            run_and_read(*step, inputs));
    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                            trainer->run_step(dataset->batch(0)));
    static_cast<void>(metrics);
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<unsigned char> after,
                            run_and_read(*step, inputs));

    ASSERT_EQ(before.size(), after.size());
    EXPECT_LT(SaTest::first_difference(before, after), before.size())
        << "the evaluation step computed the same logits before and after the training step";
}

/** @brief Tests that a run of a probe step writes neither running vector of a batch norm. */
TEST_F(Integration_EvaluationStep, AProbeRunLeavesBothRunningVectorsUnchanged)
{
    NN::BatchNorm2d model(K_CHANNELS, NN::NormOptions{});
    const Shape image{K_BATCH, K_SIDE, K_SIDE, K_CHANNELS};

    // The evaluation trace gives the buffers their leaves; the probe trace reads neither.
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep evaluation,
                            Train::build_evaluation_graph(model, image));
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep probe,
                            Train::build_probe_graph(model, image));
    static_cast<void>(evaluation);

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(*engine, probe, state, pool));

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::vector<unsigned char>> before,
                            buffer_bytes(state));
    ASSERT_EQ(before.size(), 2u);

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, noise_of(*engine, image));
    SA_EXPECT_OK(step->run(inputs));

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::vector<unsigned char>> after,
                            buffer_bytes(state));
    EXPECT_EQ(before, after);
}

/** @brief Tests that run() refuses an input of another batch and prints both shapes. */
TEST_F(Integration_EvaluationStep, AnInputOfAnotherShapeIsRefusedWithBothShapes)
{
    Mlp model = make_mlp();
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TracedStep traced,
        Train::build_evaluation_graph(model, Shape{K_BATCH, K_FEATURES}));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(*engine, traced, state, pool));

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor half_batch,
                            noise_of(*engine, Shape{K_BATCH / 2, K_FEATURES}));

    SA_EXPECT_FAILURE(step->run(half_batch), StatusCode::Incompatible, "inputs", "[2, 6]",
                      "[4, 6]");
}

/** @brief Tests that output() refuses a name the model did not publish and lists what it did. */
TEST_F(Integration_EvaluationStep, AnUnknownOutputNameListsTheNames)
{
    Mlp model = make_mlp();
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TracedStep traced,
        Train::build_evaluation_graph(model, Shape{K_BATCH, K_FEATURES}));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(*engine, traced, state, pool));

    SA_EXPECT_FAILURE(step->output("boxes"), StatusCode::InvalidDescription, "boxes",
                      NN::K_SINGLE_PORT_NAME);
}
