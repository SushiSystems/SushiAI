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
 * @file test_evaluation_step.cpp
 * @brief Tests an EvaluationStep: its feeds, the values it carries between runs and the step a
 *        trainer compiles.
 * @author Mustafa Garip
 */

#include <cmath>
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
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
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
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the elements of the one row every case feeds. */
    constexpr int64_t K_WIDTH = 3;

    /** @brief Holds what the running sum is before the first run and after a reset. */
    constexpr double K_INITIAL = 1.0;

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
        SA_TRY_ASSIGN(const Graph::ValueID next, builder.add(sum, builder.mul(x, gain)));
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
        Result<Shape> output_shape(const Shape& input) const
        {
            return input;
        }
    };

    static_assert(NN::TraceModule<Accumulator>, "Accumulator must be a TraceModule");

    /** @brief Allocates a float32 tensor of @p shape holding @p values. */
    Result<SushiBLAS::Tensor> tensor_of(SushiBLAS::Engine& engine, const Shape& shape,
                                        const std::vector<double>& values)
    {
        SA_TRY_ASSIGN(SushiBLAS::Tensor tensor, allocate_tensor(engine, shape));
        SaTest::write_tensor(tensor,
                             SushiRuntime::span<const double>(values.data(), values.size()));

        return std::move(tensor);
    }

    /** @brief Provides the host backend the steps are compiled for and the tensors live on. */
    class HostBackendTest : public ::testing::Test
    {
        protected:
            /** @brief The host backend; declared first so that it outlives every part below. */
            HostExecutionBackend backend;
    };

    /** @brief Builds the carrying step of the accumulator and the two tensors a run feeds. */
    class CarryingStep : public HostBackendTest
    {
        protected:
            /** @brief Traces the model, compiles the step and fills x and the gain. */
            void SetUp() override
            {
                const Shape row{1, K_WIDTH};
                SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep traced,
                                        Train::build_carrying_graph(model, row));
                ASSERT_EQ(traced.fed.leaves().size(), 1u);
                ASSERT_EQ(traced.carried.values().size(), 1u);

                SA_ASSERT_OK_AND_ASSIGN(NN::ModelState created,
                                        NN::ModelState::create(backend.engine(), model));
                state = std::make_unique<NN::ModelState>(std::move(created));
                pool = std::make_unique<Graph::TensorPool>(backend.engine());
                SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::EvaluationStep> compiled,
                                        Train::EvaluationStep::make(backend, traced, *state,
                                                                    *pool));
                step = std::move(compiled);

                SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor row_of_x,
                                        tensor_of(backend.engine(), row, {1.0, 2.0, 3.0}));
                SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor row_of_gain,
                                        tensor_of(backend.engine(), row, {2.0, 2.0, 2.0}));
                x = std::move(row_of_x);
                gain = std::move(row_of_gain);
            }

            /** @brief Runs the step once with x and the gain, each under its name. */
            [[nodiscard]] Status run_once()
            {
                const Train::NamedFeed feeds[] = {{K_GAIN_LEAF, &gain},
                                                  {Train::K_MODEL_INPUT_NAME, &x}};
                return step->run_by_name(SushiRuntime::span<const Train::NamedFeed>(feeds, 2));
            }

            /**
             * @brief Expects element i of the output to be K_INITIAL + (i + 1) * 2 * runs.
             *
             * Every term is a whole number far below 2^24, so each float32 operation is exact
             * and the comparison needs no rounding bound.
             */
            void expect_sum_after(int runs)
            {
                SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const sum,
                                        step->output(NN::K_SINGLE_PORT_NAME));
                const double r = static_cast<double>(runs);

                EXPECT_EQ(SaTest::read_tensor(*sum),
                          (std::vector<double>{K_INITIAL + 2.0 * r, K_INITIAL + 4.0 * r,
                                               K_INITIAL + 6.0 * r}))
                    << "after " << runs << " runs";
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

    /** @brief Runs the carrying step on the host backend. */
    class Small_CarryingStep : public CarryingStep
    {

    };

    /** @brief Holds a list of feed names and the words of the refusal the step gives it. */
    struct FeedRow
    {
        std::string name;                        /**< The row's name. */
        std::vector<std::string_view> names;     /**< The leaf each fed tensor is named for. */
        std::string first;                       /**< A text the message holds. */
        std::string second;                      /**< Another text the message holds. */
    };

    /** @brief Runs the refusal of a feed list over each row. */
    class Small_CarryingStepNamedFeeds : public CarryingStep,
                                         public ::testing::WithParamInterface<FeedRow>
    {

    };

    /** @brief Builds the evaluation step of a Linear over K_WIDTH on the host backend. */
    class Small_PlainEvaluationStep : public HostBackendTest
    {
        protected:
            /** @brief Traces the model, binds a state and compiles the step. */
            void SetUp() override
            {
                SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep traced,
                                        Train::build_evaluation_graph(model, Shape{1, K_WIDTH}));
                SA_ASSERT_OK_AND_ASSIGN(NN::ModelState created,
                                        NN::ModelState::create(backend.engine(), model));
                state = std::make_unique<NN::ModelState>(std::move(created));
                pool = std::make_unique<Graph::TensorPool>(backend.engine());
                SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::EvaluationStep> compiled,
                                        Train::EvaluationStep::make(backend, traced, *state,
                                                                    *pool));
                step = std::move(compiled);
            }

            /// @brief Holds the model the step evaluates.
            NN::Linear model{K_WIDTH, K_WIDTH};

            /// @brief Holds the state the step is bound to.
            std::unique_ptr<NN::ModelState> state;

            /// @brief Holds every value of the graph.
            std::unique_ptr<Graph::TensorPool> pool;

            /// @brief Holds the step under test.
            std::unique_ptr<Train::EvaluationStep> step;
    };
} // namespace

/** @brief Tests that the step lists the model input first and the fed leaf behind it. */
TEST_F(Small_CarryingStep, TheStepNamesTheModelInputThenTheFedLeaves)
{
    ASSERT_EQ(step->input_names().size(), 2u);
    EXPECT_EQ(step->input_names()[0], Train::K_MODEL_INPUT_NAME);
    EXPECT_EQ(step->input_names()[1], K_GAIN_LEAF);
}

/** @brief Tests that a carrying step reports its state and its trailing work. */
TEST_F(Small_CarryingStep, ACarryingStepReportsItsStateAndItsTrailingWork)
{
    EXPECT_TRUE(step->carries_state());
    EXPECT_TRUE(step->step().has_trailing_work());
}

/** @brief Tests that input() returns the fed leaf of the given name. */
TEST_F(Small_CarryingStep, InputReturnsTheFedLeafOfThatName)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const leaf, step->input(K_GAIN_LEAF));

    EXPECT_EQ(leaf->shape[1], K_WIDTH);
}

/** @brief Tests that input() refuses an unknown name and lists the names the step has. */
TEST_F(Small_CarryingStep, InputRefusesAnUnknownNameAndListsTheNames)
{
    SA_EXPECT_FAILURE(step->input("absent"), StatusCode::InvalidDescription, "'absent'",
                      "'inputs'", "'gain'");
}

/** @brief Tests that each run adds x * gain to what the run before left. */
TEST_F(Small_CarryingStep, ARunReadsWhatTheRunBeforeCarried)
{
    SA_EXPECT_OK(run_once());
    expect_sum_after(1);
    SA_EXPECT_OK(run_once());
    expect_sum_after(2);
    SA_EXPECT_OK(run_once());
    expect_sum_after(3);
}

/** @brief Tests that three runs compile the step once and count three runs. */
TEST_F(Small_CarryingStep, ThreeRunsCompileOnceAndCountThreeRuns)
{
    SA_EXPECT_OK(run_once());
    SA_EXPECT_OK(run_once());
    SA_EXPECT_OK(run_once());

    EXPECT_EQ(step->step().compile_count(), 1u);
    EXPECT_EQ(step->step().run_count(), 3u);
}

/** @brief Tests that a reset puts the carried value back to its initial value. */
TEST_F(Small_CarryingStep, AResetRestoresTheInitialValue)
{
    SA_EXPECT_OK(run_once());
    SA_EXPECT_OK(run_once());
    SA_EXPECT_OK(step->reset());
    SA_EXPECT_OK(run_once());

    expect_sum_after(1);
}

/** @brief Tests that a reset does not compile the step again. */
TEST_F(Small_CarryingStep, AResetDoesNotCompileAgain)
{
    SA_EXPECT_OK(run_once());
    SA_EXPECT_OK(step->reset());

    EXPECT_EQ(step->step().compile_count(), 1u);
}

/** @brief Tests that the feeds in the order of the step give what the named feeds give. */
TEST_F(Small_CarryingStep, ARunInOrderFeedsTheLeavesAsTheStepListsThem)
{
    const SushiBLAS::Tensor* const feeds[] = {&x, &gain};
    SA_EXPECT_OK(step->run_in_order(Train::FeedSpan(feeds, 2)));

    expect_sum_after(1);
}

/** @brief Tests that a run with one tensor is refused by a step that is fed two leaves. */
TEST_F(Small_CarryingStep, ARunOfOneTensorIsRefusedByAStepFedTwoLeaves)
{
    SA_EXPECT_FAILURE(step->run(x), StatusCode::Incompatible, "1 tensors", "fed 2", "'gain'");
    EXPECT_EQ(step->step().run_count(), 0u);
}

/** @brief Tests that an unknown name, a repeated name and a missing leaf are each refused. */
TEST_P(Small_CarryingStepNamedFeeds, NamedFeedsThatDoNotCoverTheLeavesAreRefused)
{
    const FeedRow& row = GetParam();
    std::vector<Train::NamedFeed> feeds;
    for (const std::string_view name : row.names)
        feeds.push_back(Train::NamedFeed{name, name == Train::K_MODEL_INPUT_NAME ? &x : &gain});

    SA_EXPECT_FAILURE(
        step->run_by_name(SushiRuntime::span<const Train::NamedFeed>(feeds.data(), feeds.size())),
        StatusCode::InvalidDescription, row.first, row.second);
    EXPECT_EQ(step->step().run_count(), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_CarryingStepNamedFeeds,
    ::testing::Values(FeedRow{"AnUnknownName", {"scale", Train::K_MODEL_INPUT_NAME}, "'scale'",
                              "no input leaf"},
                      FeedRow{"ARepeatedName", {K_GAIN_LEAF, K_GAIN_LEAF}, "'gain' twice",
                              "'gain'"},
                      FeedRow{"AShortList", {K_GAIN_LEAF}, "1 named tensors", "fed 2"}),
    [](const ::testing::TestParamInfo<FeedRow>& info) { return info.param.name; });

/** @brief Tests that a fed tensor of another shape than its leaf is refused by name. */
TEST_F(Small_CarryingStep, AFeedOfAnotherShapeIsRefusedUnderItsName)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor wide,
                            allocate_tensor(backend.engine(), Shape{1, K_WIDTH + 1}));
    const Train::NamedFeed feeds[] = {{K_GAIN_LEAF, &wide}, {Train::K_MODEL_INPUT_NAME, &x}};

    SA_EXPECT_FAILURE(step->run_by_name(SushiRuntime::span<const Train::NamedFeed>(feeds, 2)),
                      StatusCode::Incompatible, "gain", "[1, 4]", "[1, 3]");
}

/** @brief Tests that an ordinary evaluation step has one input, the model input. */
TEST_F(Small_PlainEvaluationStep, AnOrdinaryStepFeedsOneLeaf)
{
    ASSERT_EQ(step->input_names().size(), 1u);
    EXPECT_EQ(step->input_names()[0], Train::K_MODEL_INPUT_NAME);
}

/** @brief Tests that an ordinary evaluation step carries no state and has no trailing work. */
TEST_F(Small_PlainEvaluationStep, AnOrdinaryStepCarriesNothing)
{
    EXPECT_FALSE(step->carries_state());
    EXPECT_FALSE(step->step().has_trailing_work());
}

/** @brief Tests that a reset of a step that never ran succeeds and leaves zero runs. */
TEST_F(Small_PlainEvaluationStep, AResetWithZeroRunsDoesNothing)
{
    SA_EXPECT_OK(step->reset());

    EXPECT_EQ(step->step().run_count(), 0u);
}

namespace
{
    /** @brief Provides the device engine the evaluation cases that need a kernel run on. */
    class Medium_EvaluationStep : public SaTest::EngineTest {};

    /** @brief Provides the host backend the evaluation refusals are decided on. */
    class Small_EvaluationStep : public HostBackendTest {};

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

    /** @brief Runs @p step over @p inputs and returns its single output as host reals. */
    Result<std::vector<double>> run_and_read(Train::EvaluationStep& step,
                                             const SushiBLAS::Tensor& inputs)
    {
        SA_TRY(step.run(inputs));
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const output,
                      step.output(NN::K_SINGLE_PORT_NAME));

        return SaTest::read_tensor(*output);
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

    /** @brief Holds the parameters of the MLP, in entry order: weight, bias, weight, bias. */
    using MlpParameters = std::vector<std::vector<double>>;

    /** @brief Reads every entry of @p state as host reals, in entry order. */
    Result<MlpParameters> read_parameters(NN::ModelState& state)
    {
        MlpParameters parameters;
        for (const NN::StateEntry& entry : state.entries())
        {
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor, state.tensor(entry.state));
            parameters.push_back(SaTest::read_tensor(*tensor));
        }

        return parameters;
    }

    /** @brief Holds the logits of the host reference and the rounding bound of each. */
    struct ReferenceLogits
    {
        std::vector<double> logits;  /**< Row-major [K_BATCH, K_CLASSES]. */
        std::vector<double> bounds;  /**< The absolute float32 rounding bound of each logit. */
    };

    /** @brief Returns the tanh form of GELU, the form the graph's GELU node has. */
    double gelu_of(double v)
    {
        return 0.5 * v * (1.0 + std::tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
    }

    /**
     * @brief Returns the logits of the MLP in double with the float32 bound of each: u times
     *        9 M + 17 sum |w2| A, u the unit roundoff.
     *
     * The second layer rounds 9 times against M = sum |w2 h| + |b2|. A hidden unit carries 7
     * roundings of the first layer against A = sum |w1 x| + |b1|, times a GELU slope of at most
     * 1.2, and the 8 operations of the tanh form against |h| <= A: 17 A.
     */
    ReferenceLogits reference_logits(const MlpParameters& p, const std::vector<double>& inputs)
    {
        ReferenceLogits reference;
        const double u = TestSupport::unit_roundoff<float>();
        for (int64_t row = 0; row < K_BATCH; ++row)
        {
            std::vector<double> hidden(static_cast<std::size_t>(K_HIDDEN));
            std::vector<double> reach(static_cast<std::size_t>(K_HIDDEN));
            for (int64_t j = 0; j < K_HIDDEN; ++j)
            {
                double pre = p[1][static_cast<std::size_t>(j)];
                double magnitude = std::abs(pre);
                for (int64_t i = 0; i < K_FEATURES; ++i)
                {
                    const double term = p[0][static_cast<std::size_t>(j * K_FEATURES + i)]
                                      * inputs[static_cast<std::size_t>(row * K_FEATURES + i)];
                    pre += term;
                    magnitude += std::abs(term);
                }
                hidden[static_cast<std::size_t>(j)] = gelu_of(pre);
                reach[static_cast<std::size_t>(j)] = magnitude;
            }
            for (int64_t c = 0; c < K_CLASSES; ++c)
            {
                double logit = p[3][static_cast<std::size_t>(c)];
                double second = std::abs(logit);
                double carried = 0.0;
                for (int64_t j = 0; j < K_HIDDEN; ++j)
                {
                    const double weight = p[2][static_cast<std::size_t>(c * K_HIDDEN + j)];
                    const double product = weight * hidden[static_cast<std::size_t>(j)];
                    logit += product;
                    second += std::abs(product);
                    carried += std::abs(weight) * reach[static_cast<std::size_t>(j)];
                }
                reference.logits.push_back(logit);
                reference.bounds.push_back(u * (9.0 * second + 17.0 * carried));
            }
        }

        return reference;
    }

    /** @brief Expects each of @p result to sit within the bound of its reference logit. */
    void expect_logits_match(const std::vector<double>& result, const ReferenceLogits& reference)
    {
        ASSERT_EQ(result.size(), reference.logits.size());
        for (std::size_t at = 0; at < result.size(); ++at)
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                result[at], reference.logits[at], reference.bounds[at])))
                << "logit " << at;
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
    SA_EXPECT_OK(NN::initialize(backend(), state));

    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(backend(), traced, state, pool));
    EXPECT_EQ(step->step().compile_count(), 1u);

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs,
                            noise_of(*engine, Shape{K_BATCH, K_FEATURES}));
    SA_EXPECT_OK(step->run(inputs));
    SA_EXPECT_OK(step->run(inputs));
    SA_EXPECT_OK(step->run(inputs));

    EXPECT_EQ(step->step().compile_count(), 1u);
}

/**
 * @brief Tests that an evaluation step over its own pool reads the weights a training step
 *        over another pool wrote into the shared state: its logits match a host reference of
 *        the weights before the step and of the weights after it.
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
                            Train::bind_parameters(backend(), built, state, training_pool));

    // The default policy zeroes the classifier, so the first evaluation gives zero logits.
    SA_EXPECT_OK(NN::initialize(backend(), state));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend(), slots, Optim::SGDOptions{}));
    Train::TrainerOptions options;
    options.track_metrics = false;
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(backend(), built, training_pool,
                                                 objective, *optimizer, options));

    Graph::TensorPool evaluation_pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::EvaluationStep> step,
        Train::EvaluationStep::make(backend(), traced, state, evaluation_pool));

    Data::SyntheticOptions data_options;
    data_options.batch_size = K_BATCH;
    data_options.batches = 1;
    data_options.features = K_FEATURES;
    data_options.classes = K_CLASSES;
    data_options.seed = K_INPUT_SEED;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(backend(), data_options));

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs,
                            noise_of(*engine, Shape{K_BATCH, K_FEATURES}));
    const std::vector<double> input_values = SaTest::read_tensor(inputs);

    SA_ASSERT_OK_AND_ASSIGN(const MlpParameters before_parameters, read_parameters(state));
    ASSERT_EQ(before_parameters.size(), 4u);
    const ReferenceLogits before_reference = reference_logits(before_parameters, input_values);
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> before, run_and_read(*step, inputs));
    expect_logits_match(before, before_reference);

    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                            trainer->run_step(dataset_batch));
    static_cast<void>(metrics);

    SA_ASSERT_OK_AND_ASSIGN(const MlpParameters after_parameters, read_parameters(state));
    const ReferenceLogits after_reference = reference_logits(after_parameters, input_values);
    ASSERT_NE(before_reference.logits, after_reference.logits)
        << "the training step left the logits of the reference where they were";
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> after, run_and_read(*step, inputs));
    expect_logits_match(after, after_reference);
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
    SA_EXPECT_OK(NN::initialize(backend(), state));

    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(backend(), probe, state, pool));

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
TEST_F(Small_EvaluationStep, AnInputOfAnotherShapeIsRefusedWithBothShapes)
{
    MLP model = make_mlp();
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TracedStep traced,
        Train::build_evaluation_graph(model, Shape{K_BATCH, K_FEATURES}));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(backend.engine(), model));

    Graph::TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(backend, traced, state, pool));

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor half_batch,
                            noise_of(backend.engine(), Shape{K_BATCH / 2, K_FEATURES}));

    SA_EXPECT_FAILURE(step->run(half_batch), StatusCode::Incompatible, "inputs", "[2, 6]",
                      "[4, 6]");
}

/** @brief Tests that output() refuses a name the model did not publish and lists what it did. */
TEST_F(Small_EvaluationStep, AnUnknownOutputNameListsTheNames)
{
    MLP model = make_mlp();
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TracedStep traced,
        Train::build_evaluation_graph(model, Shape{K_BATCH, K_FEATURES}));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(backend.engine(), model));

    Graph::TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(backend, traced, state, pool));

    SA_EXPECT_FAILURE(step->output("boxes"), StatusCode::InvalidDescription, "boxes",
                      NN::K_SINGLE_PORT_NAME);
}
