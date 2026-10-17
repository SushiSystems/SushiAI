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
 * @brief Tests an EvaluationStep: its feeds, the values it carries between runs and the step a
 *        trainer compiles.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the elements of the one row every case feeds. */
    constexpr int64_t K_WIDTH = 3;

    /** @brief Holds what the running sum is before the first run and after a reset. */
    constexpr double K_INITIAL = 1.0;

    /** @brief Holds the absolute tolerance of a sum of small whole numbers in float32. */
    constexpr float K_TOLERANCE = 1e-6f;

    /** @brief Names the leaf the accumulator is fed beside the model input. */
    constexpr std::string_view K_GAIN_LEAF = "gain";

    /** @brief Declares the two leaves of the running sum and traces sum + x * gain. */
    [[nodiscard]] Result<Graph::ValueID> accumulate(const NN::Trace& trace, Graph::ValueID x)
    {
        const Graph::GraphBuilder& builder = trace.builder();
        const Shape shape = builder.graph().value(x).shape;
        SA_TRY_ASSIGN(const Graph::ValueID gain, trace.feed(K_GAIN_LEAF, shape, DType::Float32));
        SA_TRY_ASSIGN(const Graph::ValueID sum,
                      trace.carried_leaf(shape, "sum", DType::Float32));
        SA_TRY_ASSIGN(const Graph::ValueID next, NN::Detail::bridge_throwing([&]()
        {
            return builder.add(sum, builder.mul(x, gain));
        }));
        SA_TRY(trace.carry(sum, next, K_INITIAL));

        return next;
    }

    /** @brief Returns the running sum of x * gain: a fed leaf and a carried value. */
    struct Accumulator
    {
        /** @brief Lists no state; the sum belongs to the step and not to the model. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Traces the sum, carries it and returns it, or fails the trace. */
        Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x)
        {
            return trace.value_or_fail(accumulate(trace, x), x);
        }

        /** @brief Returns @p input. */
        Shape output_shape(const Shape& input) const
        {
            return input;
        }
    };

    static_assert(NN::TraceModule<Accumulator>, "Accumulator must be a TraceModule");

    /** @brief Builds the carrying step of the accumulator and the two tensors a run feeds. */
    class Medium_CarryingStep : public SaTest::EngineTest
    {
        protected:
            /** @brief Traces the model, compiles the step and fills x and the gain. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();

                const Shape row{1, K_WIDTH};
                SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep traced,
                                        Train::build_carrying_graph(model, row));
                ASSERT_EQ(traced.fed.leaves().size(), 1u);
                ASSERT_EQ(traced.carried.values().size(), 1u);

                SA_ASSERT_OK_AND_ASSIGN(NN::ModelState created,
                                        NN::ModelState::create(*engine, model));
                state = std::make_unique<NN::ModelState>(std::move(created));
                pool = std::make_unique<Graph::TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::EvaluationStep> compiled,
                                        Train::EvaluationStep::make(*engine, traced, *state,
                                                                    *pool));
                step = std::move(compiled);

                SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor row_of_x,
                                        allocate_tensor(*engine, row));
                SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor row_of_gain,
                                        allocate_tensor(*engine, row));
                x = std::move(row_of_x);
                gain = std::move(row_of_gain);
                for (int64_t at = 0; at < K_WIDTH; ++at)
                {
                    x.data_as<float>()[at] = static_cast<float>(at + 1);
                    gain.data_as<float>()[at] = 2.0f;
                }
            }

            /** @brief Releases the parts before the engine they were built on. */
            void TearDown() override
            {
                step.reset();
                pool.reset();
                state.reset();
                x = SushiBLAS::Tensor();
                gain = SushiBLAS::Tensor();
                SaTest::EngineTest::TearDown();
            }

            /** @brief Runs the step once with x and the gain, each under its name. */
            [[nodiscard]] Status run_once()
            {
                const Train::NamedFeed feeds[] = {{K_GAIN_LEAF, &gain},
                                                  {Train::K_MODEL_INPUT_NAME, &x}};
                return step->run_by_name(SushiRuntime::span<const Train::NamedFeed>(feeds, 2));
            }

            /** @brief Expects element i of the output to be K_INITIAL + (i + 1) * 2 * runs. */
            void expect_sum_after(int runs)
            {
                SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const sum,
                                        step->output(NN::K_SINGLE_PORT_NAME));
                for (int64_t at = 0; at < K_WIDTH; ++at)
                {
                    const float expected = static_cast<float>(K_INITIAL)
                                         + static_cast<float>((at + 1) * 2 * runs);
                    EXPECT_NEAR(sum->data_as<float>()[at], expected, K_TOLERANCE)
                        << "element " << at << " after " << runs << " runs";
                }
            }

            /// @brief Holds the model, which has no state of its own.
            Accumulator model;

            /// @brief Holds the empty model state the step is bound to.
            std::unique_ptr<NN::ModelState> state;

            /// @brief Holds every value of the graph.
            std::unique_ptr<Graph::TensorPool> pool;

            /// @brief Holds the step under test.
            std::unique_ptr<Train::EvaluationStep> step;

            /// @brief Holds the row 1, 2, 3 fed as the model input.
            SushiBLAS::Tensor x;

            /// @brief Holds the row 2, 2, 2 fed as the gain.
            SushiBLAS::Tensor gain;
    };

    /** @brief Provides the engine an ordinary evaluation step is built on. */
    class Medium_PlainEvaluationStep : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that the step lists the model input first and the fed leaf behind it. */
TEST_F(Medium_CarryingStep, TheStepNamesTheModelInputThenTheFedLeaves)
{
    ASSERT_EQ(step->input_names().size(), 2u);
    EXPECT_EQ(step->input_names()[0], Train::K_MODEL_INPUT_NAME);
    EXPECT_EQ(step->input_names()[1], K_GAIN_LEAF);
    EXPECT_TRUE(step->carries_state());
    EXPECT_TRUE(step->step().has_trailing_work());

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const leaf, step->input(K_GAIN_LEAF));
    EXPECT_EQ(leaf->shape[1], K_WIDTH);
    SA_EXPECT_FAILURE(step->input("absent"), StatusCode::InvalidDescription, "'absent'",
                      "'inputs'", "'gain'");
}

/** @brief Tests that each run adds x * gain to what the run before left. */
TEST_F(Medium_CarryingStep, ARunReadsWhatTheRunBeforeCarried)
{
    SA_EXPECT_OK(run_once());
    expect_sum_after(1);
    SA_EXPECT_OK(run_once());
    expect_sum_after(2);
    SA_EXPECT_OK(run_once());
    expect_sum_after(3);

    EXPECT_EQ(step->step().compile_count(), 1u);
    EXPECT_EQ(step->step().run_count(), 3u);
}

/** @brief Tests that a reset puts the carried value back to its initial value. */
TEST_F(Medium_CarryingStep, AResetRestoresTheInitialValue)
{
    SA_EXPECT_OK(run_once());
    SA_EXPECT_OK(run_once());
    SA_EXPECT_OK(step->reset());
    SA_EXPECT_OK(run_once());

    expect_sum_after(1);
    EXPECT_EQ(step->step().compile_count(), 1u);
}

/** @brief Tests that the feeds in the order of the step give what the named feeds give. */
TEST_F(Medium_CarryingStep, ARunInOrderFeedsTheLeavesAsTheStepListsThem)
{
    const SushiBLAS::Tensor* const feeds[] = {&x, &gain};
    SA_EXPECT_OK(step->run_in_order(Train::FeedSpan(feeds, 2)));

    expect_sum_after(1);
}

/** @brief Tests that a run with one tensor is refused by a step that is fed two leaves. */
TEST_F(Medium_CarryingStep, ARunOfOneTensorIsRefusedByAStepFedTwoLeaves)
{
    SA_EXPECT_FAILURE(step->run(x), StatusCode::Incompatible, "1 tensors", "fed 2", "'gain'");
    EXPECT_EQ(step->step().run_count(), 0u);
}

/** @brief Tests that an unknown name, a repeated name and a missing leaf are each refused. */
TEST_F(Medium_CarryingStep, NamedFeedsThatDoNotCoverTheLeavesAreRefused)
{
    const Train::NamedFeed unknown[] = {{"scale", &gain}, {Train::K_MODEL_INPUT_NAME, &x}};
    const Train::NamedFeed twice[] = {{K_GAIN_LEAF, &gain}, {K_GAIN_LEAF, &gain}};
    const Train::NamedFeed one[] = {{K_GAIN_LEAF, &gain}};

    SA_EXPECT_FAILURE(step->run_by_name(SushiRuntime::span<const Train::NamedFeed>(unknown, 2)),
                      StatusCode::InvalidDescription, "'scale'", "no input leaf");
    SA_EXPECT_FAILURE(step->run_by_name(SushiRuntime::span<const Train::NamedFeed>(twice, 2)),
                      StatusCode::InvalidDescription, "'gain' twice");
    SA_EXPECT_FAILURE(step->run_by_name(SushiRuntime::span<const Train::NamedFeed>(one, 1)),
                      StatusCode::InvalidDescription, "1 named tensors", "fed 2");
    EXPECT_EQ(step->step().run_count(), 0u);
}

/** @brief Tests that a fed tensor of another shape than its leaf is refused by name. */
TEST_F(Medium_CarryingStep, AFeedOfAnotherShapeIsRefusedUnderItsName)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor wide,
                            allocate_tensor(*engine, Shape{1, K_WIDTH + 1}));
    const Train::NamedFeed feeds[] = {{K_GAIN_LEAF, &wide}, {Train::K_MODEL_INPUT_NAME, &x}};

    SA_EXPECT_FAILURE(step->run_by_name(SushiRuntime::span<const Train::NamedFeed>(feeds, 2)),
                      StatusCode::Incompatible, "gain", "[1, 4]", "[1, 3]");
}

/** @brief Tests that an ordinary evaluation step has one input, no state and no trailing work. */
TEST_F(Medium_PlainEvaluationStep, AnOrdinaryStepFeedsOneLeafAndCarriesNothing)
{
    NN::Linear model{K_WIDTH, K_WIDTH};
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep traced,
                            Train::build_evaluation_graph(model, Shape{1, K_WIDTH}));
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelState state, NN::ModelState::create(*engine, model));
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(*engine, traced, state, pool));

    ASSERT_EQ(step->input_names().size(), 1u);
    EXPECT_EQ(step->input_names()[0], Train::K_MODEL_INPUT_NAME);
    EXPECT_FALSE(step->carries_state());
    EXPECT_FALSE(step->step().has_trailing_work());
    SA_EXPECT_OK(step->reset());
    EXPECT_EQ(step->step().run_count(), 0u);
}

namespace
{
    class Medium_EvaluationStep : public SaTest::EngineTest {};

    /** @brief The model the cases without a buffer run. */
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

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
    MLP make_mlp()
    {
        return MLP(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});
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
TEST_F(Medium_EvaluationStep, TheStepCompilesOnce)
{
    MLP model = make_mlp();
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
TEST_F(Medium_EvaluationStep, AnEvaluationStepSeesAWeightTheTrainingStepJustMoved)
{
    MLP model = make_mlp();
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
    options.track_metrics = false;
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built, training_pool,
                                                 objective, *optimizer, options));

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
    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                            trainer->run_step(dataset_batch));
    static_cast<void>(metrics);
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<unsigned char> after,
                            run_and_read(*step, inputs));

    ASSERT_EQ(before.size(), after.size());
    EXPECT_LT(SaTest::first_difference(before, after), before.size())
        << "the evaluation step computed the same logits before and after the training step";
}

/** @brief Tests that a run of a probe step writes neither running vector of a batch norm. */
TEST_F(Medium_EvaluationStep, AProbeRunLeavesBothRunningVectorsUnchanged)
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
TEST_F(Medium_EvaluationStep, AnInputOfAnotherShapeIsRefusedWithBothShapes)
{
    MLP model = make_mlp();
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
TEST_F(Medium_EvaluationStep, AnUnknownOutputNameListsTheNames)
{
    MLP model = make_mlp();
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
