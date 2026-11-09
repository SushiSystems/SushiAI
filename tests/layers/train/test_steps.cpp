/**************************************************************************/
/* test_steps.cpp                                                         */
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
 * @file test_steps.cpp
 * @brief Tests the builders that trace a model into a training, probe or evaluation graph.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/flatten.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/sample_models/two_output_model.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using NN::BatchNorm2d;
using NN::NormOptions;
using Train::CrossEntropyObjective;

namespace
{
    /** @brief The model whose training graph the cases below read. */
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    /** @brief A model that requests buffer updates and still ends in [N, classes]. */
    using NormClassifier = NN::Sequential<BatchNorm2d, NN::Flatten, NN::Linear>;

    /** @brief Samples per step in every case of this file. */
    constexpr int64_t K_BATCH = 4;

    /** @brief Input features of the MLP. */
    constexpr int64_t K_FEATURES = 6;

    /** @brief Hidden units of the MLP. */
    constexpr int64_t K_HIDDEN = 8;

    /** @brief Classes every classifier of this file ends in. */
    constexpr int64_t K_CLASSES = 3;

    /** @brief Channels of the normalised image. */
    constexpr int64_t K_CHANNELS = 3;

    /** @brief Height and width of the normalised image. */
    constexpr int64_t K_SIDE = 2;

    /** @brief Channels of the layer whose values the host evaluator computes. */
    constexpr int64_t K_VALUE_CHANNELS = 2;

    /** @brief The constant added to a variance in the value cases: 5 + 4 is 9, and 9 has a root. */
    constexpr double K_VALUE_EPSILON = 4.0;

    /**
     * @brief Counts the roundings on the longest path to an element the probe writes: the mean of
     *        four values rounds 3 additions and a division, the centred value, its square, 3
     *        additions and a division give the variance, then the epsilon, the root and the
     *        inverse, then the centred value, the product with the inverse, the gain and the
     *        shift: 4 + 1 + 1 + 3 + 1 + 1 + 1 + 1 + 4 = 17.
     */
    constexpr std::size_t K_PROBE_OPERATIONS = 17;

    /**
     * @brief Counts the roundings on the longest path to an element the folded layer writes: the
     *        scale rounds the epsilon sum, the root and the quotient, the shift rounds the
     *        product of the mean with the scale and the difference, and the element rounds the
     *        product with the scale and the sum with the shift: 3 + 1 + 1 + 1 = 6.
     */
    constexpr std::size_t K_FOLD_OPERATIONS = 6;

    /** @brief Builds the MLP over K_FEATURES, K_HIDDEN and K_CLASSES. */
    MLP make_mlp()
    {
        return MLP(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});
    }

    /** @brief Builds the batch-norm classifier over a K_SIDE by K_SIDE image. */
    NormClassifier make_norm_classifier()
    {
        return NormClassifier(BatchNorm2d(K_CHANNELS, NormOptions{}), NN::Flatten{},
                              NN::Linear{K_SIDE * K_SIDE * K_CHANNELS, K_CLASSES});
    }

    /** @brief Returns the [N, H, W, C] shape of one batch of the normalised image. */
    Shape image_batch()
    {
        return Shape{K_BATCH, K_SIDE, K_SIDE, K_CHANNELS};
    }

    /** @brief Returns the [1, H, W, C] shape of the image whose values the host computes. */
    Shape value_image()
    {
        return Shape{1, K_SIDE, K_SIDE, K_VALUE_CHANNELS};
    }

    /** @brief Returns the state identity of every declared parameter of @p model, in walk order. */
    template<typename M>
    std::vector<Graph::StateID> parameter_states(M& model)
    {
        std::vector<Graph::StateID> states;
        NN::for_each_parameter(model, [&states](const std::string&, NN::Parameter& parameter)
        {
            states.push_back(parameter.state());
        });

        return states;
    }

    /** @brief Returns the name of every declared parameter of @p model, in walk order. */
    template<typename M>
    std::vector<std::string> parameter_names(M& model)
    {
        std::vector<std::string> names;
        NN::for_each_parameter(model, [&names](const std::string& name, NN::Parameter&)
        {
            names.push_back(name);
        });

        return names;
    }

    /** @brief Returns the leaf of every declared parameter of @p model in @p graph, in walk order. */
    template<typename M>
    std::vector<Graph::ValueID> parameter_leaves(M& model, const Graph::Graph& graph)
    {
        std::vector<Graph::ValueID> leaves;
        NN::for_each_parameter(model, [&](const std::string&, NN::Parameter& parameter)
        {
            leaves.push_back(parameter.id_in(graph));
        });

        return leaves;
    }

    /** @brief Tells whether @p id is one of the pinned values of @p built. */
    bool is_pinned(const Train::TrainingGraph& built, Graph::ValueID id)
    {
        return std::find(built.pinned.begin(), built.pinned.end(), id) != built.pinned.end();
    }

    /** @brief Holds the elements the value cases write into the state of a BatchNorm2d. */
    struct NormState
    {
        std::vector<double> weight;    /**< The gain of each channel. */
        std::vector<double> bias;      /**< The shift of each channel. */
        std::vector<double> mean;      /**< The stored mean of each channel. */
        std::vector<double> variance;  /**< The stored variance of each channel. */
    };

    /**
     * @brief Writes @p state into every leaf of @p model that @p graph holds.
     * @return A success, or the first failure HostEvaluator::write() returns.
     */
    Status write_norm_state(Graph::HostEvaluator& evaluator, const Graph::Graph& graph,
                            BatchNorm2d& model, const NormState& state)
    {
        Status outcome;
        const auto keep = [&outcome](const Status& written)
        {
            if (outcome.is_ok())
                outcome = written;
        };
        NN::for_each_parameter(model, [&](const std::string& name, NN::Parameter& parameter)
        {
            if (parameter.id_in(graph) != Graph::K_NO_VALUE)
                keep(evaluator.write(parameter.id_in(graph),
                                     name == "weight" ? state.weight : state.bias));
        });
        NN::for_each_buffer(model, [&](const std::string& name, NN::Buffer& buffer)
        {
            if (buffer.id_in(graph) != Graph::K_NO_VALUE)
                keep(evaluator.write(buffer.id_in(graph),
                                     name == "running_mean" ? state.mean : state.variance));
        });

        return outcome;
    }

    /** @brief Expects each of @p result to sit within @p operations roundings of @p expected. */
    void expect_each_within(const std::vector<double>& result, const std::vector<double>& expected,
                            std::size_t operations)
    {
        ASSERT_EQ(result.size(), expected.size());
        for (std::size_t at = 0; at < expected.size(); ++at)
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
                result[at], expected[at], operations)))
                << "element " << at;
    }

    /**
     * @brief Runs the graph of @p step over the image whose channel 0 is {1, 3, 5, 7} and whose
     *        channel 1 is {2, 2, 2, 2}, under the gain {2, 1}, the shift {0, 1}, the stored mean
     *        {-1, 0} and the stored variance {5, 0}.
     * @return The elements of the layer's one output, or the failure of the run.
     */
    Result<std::vector<double>> run_layer(const Train::TracedStep& step, BatchNorm2d& model)
    {
        Graph::HostEvaluator evaluator(step.graph);
        SA_TRY(evaluator.write(step.input, {1, 2, 3, 2, 5, 2, 7, 2}));
        SA_TRY(write_norm_state(evaluator, step.graph, model,
                                NormState{{2, 1}, {0, 1}, {-1, 0}, {5, 0}}));
        SA_TRY(evaluator.run());

        return evaluator.read(step.outputs.all().front().second);
    }

    /** @brief Traces @p model as a probe graph. */
    Result<Train::TracedStep> trace_probe(BatchNorm2d& model, const Shape& shape)
    {
        return Train::build_probe_graph(model, shape);
    }

    /** @brief Traces @p model as an evaluation graph. */
    Result<Train::TracedStep> trace_evaluation(BatchNorm2d& model, const Shape& shape)
    {
        return Train::build_evaluation_graph(model, shape);
    }

    /** @brief Traces @p model as a carrying graph. */
    Result<Train::TracedStep> trace_carrying(BatchNorm2d& model, const Shape& shape)
    {
        return Train::build_carrying_graph(model, shape);
    }

    /** @brief Holds a builder of a forward graph and the name of its row. */
    struct BuilderRow
    {
        std::string name;                                                /**< The row's name. */
        Result<Train::TracedStep> (*trace)(BatchNorm2d&, const Shape&);  /**< The builder. */
    };

    /** @brief Names a row by its name field. */
    std::string builder_row_name(const ::testing::TestParamInfo<BuilderRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a parameter row by its index. */
    std::string index_row_name(const ::testing::TestParamInfo<std::size_t>& info)
    {
        return "Index" + std::to_string(info.param);
    }

    /** @brief Runs one parameter index of the MLP's training graph. */
    class Small_StepsParameter : public ::testing::TestWithParam<std::size_t>
    {

    };

    /** @brief Runs one parameter index of the batch-norm classifier's state identities. */
    class Small_StepsState : public ::testing::TestWithParam<std::size_t>
    {

    };

    /** @brief Runs one source index of the batch-norm classifier's buffer updates. */
    class Small_StepsUpdate : public ::testing::TestWithParam<std::size_t>
    {

    };

    /** @brief Runs one builder of a forward graph over a batch-norm layer. */
    class Small_StepsForward : public ::testing::TestWithParam<BuilderRow>
    {

    };

    /** @brief Runs one builder of a forward graph that folds the stored statistics. */
    class Small_StepsFold : public ::testing::TestWithParam<BuilderRow>
    {

    };

    /** @brief Computes cross-entropy over the one output it is told to read, by name. */
    class NamedOutputObjective final : public Train::Objective
    {
        public:
            /** @brief Reads the output @p output of a model that ends in [batch, classes]. */
            NamedOutputObjective(std::string output, int64_t batch, int64_t classes)
                : output_(std::move(output)), batch_(batch), classes_(classes)
            {

            }

            /** @brief Declares the [batch, classes] leaf "targets". */
            void declare(const Graph::GraphBuilder& builder,
                         Train::ObjectiveLeaves& leaves) const override
            {
                leaves.add("targets", builder.input(Shape{batch_, classes_}, "targets"));
            }

            /**
             * @brief Emits the cross-entropy of the named output against "targets".
             * @pre @p outputs holds the name; a missing one stops the process.
             */
            [[nodiscard]] Result<Graph::ValueID> loss(
                const Graph::GraphBuilder& builder, const Graph::Graph&,
                const NN::ModelOutputs& outputs,
                const Train::ObjectiveLeaves& leaves) const override
            {
                const Graph::ValueID logits = outputs.at(output_).value();
                SA_TRY_ASSIGN(const Graph::ValueID loss,
                              NN::CrossEntropyLoss{}.forward(builder, logits,
                                                             leaves.at("targets")));
                return loss;
            }

            /** @brief Returns "named_output". */
            std::string_view name() const noexcept override
            {
                return "named_output";
            }

        private:
            /** @brief The name of the output the loss reads. */
            std::string output_;

            /** @brief Samples per step. */
            int64_t batch_;

            /** @brief Classes of the target leaf. */
            int64_t classes_;
    };

} // namespace

/** @brief Tests that the training graph of the MLP ends in a scalar loss over [batch, classes]. */
TEST(Small_Steps, ATrainingGraphOfAnMLPHasAScalarLossAndTheLogits)
{
    MLP model = make_mlp();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));

    ASSERT_NE(built.spec.loss, Graph::K_NO_VALUE);
    ASSERT_NE(built.spec.logits, Graph::K_NO_VALUE);
    EXPECT_EQ(built.graph.value(built.spec.loss).shape, Shape{1});
    EXPECT_EQ(built.graph.value(built.spec.logits).shape, (Shape{K_BATCH, K_CLASSES}));
}

/** @brief Tests that every parameter of the MLP has a gradient of its own extents. */
TEST_P(Small_StepsParameter, ATrainingGraphGivesEveryParameterAGradientOfItsShape)
{
    const std::size_t at = GetParam();
    MLP model = make_mlp();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));

    // Two layers carry a weight and a bias each.
    ASSERT_EQ(built.parameters.size(), 4u);
    ASSERT_NE(built.parameters[at].gradient, Graph::K_NO_VALUE);
    EXPECT_EQ(built.graph.value(built.parameters[at].gradient).shape,
              built.graph.value(built.parameters[at].value).shape);
}

/** @brief Tests that every binding names the leaf its parameter has in the built graph. */
TEST_P(Small_StepsParameter, ATrainingGraphListsEveryParameterWithItsLeafInThatGraph)
{
    const std::size_t at = GetParam();
    MLP model = make_mlp();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));
    const std::vector<std::string> names = parameter_names(model);
    const std::vector<Graph::ValueID> leaves = parameter_leaves(model, built.graph);

    ASSERT_EQ(built.parameters.size(), 4u);
    EXPECT_EQ(built.parameters[at].name, names[at]);
    EXPECT_NE(built.parameters[at].value, Graph::K_NO_VALUE);
    EXPECT_EQ(built.parameters[at].value, leaves[at]);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_StepsParameter, ::testing::Values(0u, 1u, 2u, 3u),
                         index_row_name);

/** @brief Tests that every binding carries the role its parameter declared. */
TEST(Small_Steps, ATrainingGraphListsEveryParameterWithItsRole)
{
    MLP model = make_mlp();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));

    // The walk yields each layer's weight, then its bias.
    ASSERT_EQ(built.parameters.size(), 4u);
    EXPECT_EQ(built.parameters[0].role, NN::ParameterRole::Weight);
    EXPECT_EQ(built.parameters[1].role, NN::ParameterRole::Bias);
    EXPECT_EQ(built.parameters[2].role, NN::ParameterRole::Weight);
    EXPECT_EQ(built.parameters[3].role, NN::ParameterRole::Bias);
}

/** @brief Tests that a batch-norm model requests two buffer updates. */
TEST(Small_Steps, ABatchNormModelYieldsTwoUpdates)
{
    NormClassifier model = make_norm_classifier();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, image_batch()));

    EXPECT_EQ(built.updates.updates().size(), 2u);
}

/** @brief Tests that the source of each buffer update is pinned, since the host reads it. */
TEST_P(Small_StepsUpdate, TheSourceOfAnUpdateIsPinned)
{
    const std::size_t at = GetParam();
    NormClassifier model = make_norm_classifier();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, image_batch()));

    const SushiRuntime::span<const Graph::ValueID> sources = built.updates.sources();
    ASSERT_EQ(sources.size(), 2u);
    EXPECT_TRUE(is_pinned(built, sources[at]));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_StepsUpdate, ::testing::Values(0u, 1u), index_row_name);

/** @brief Tests that the loss is pinned, since the host reads it. */
TEST(Small_Steps, ATrainingGraphPinsTheLoss)
{
    NormClassifier model = make_norm_classifier();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, image_batch()));

    EXPECT_TRUE(is_pinned(built, built.spec.loss));
}

/** @brief Tests that the one output the model published is pinned, since the host reads it. */
TEST(Small_Steps, ATrainingGraphPinsTheOutput)
{
    NormClassifier model = make_norm_classifier();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, image_batch()));

    ASSERT_EQ(built.outputs.all().size(), 1u);
    EXPECT_TRUE(is_pinned(built, built.outputs.all().front().second));
}

/**
 * @brief Tests that a probe normalises with the statistics of its own batch and reads no buffer.
 *
 * Channel 0 = {1, 3, 5, 7} has mean 4 and variance 5, channel 1 = {2, 2, 2, 2} has mean 2 and
 * variance 0. With epsilon 4 the inverse deviations are 1/3 and 1/2, and with gain {2, 1} and
 * shift {0, 1} the result is {-2, 1, -2/3, 1, 2/3, 1, 2, 1}. The stored statistics written to
 * the state differ from these, and the run needs no buffer leaf: a probe has none.
 */
TEST(Small_Steps, AProbeGraphNormalisesWithTheStatisticsOfItsBatch)
{
    BatchNorm2d model(K_VALUE_CHANNELS, NormOptions{K_VALUE_EPSILON, 0.1});
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep probe,
                            Train::build_probe_graph(model, value_image()));

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, run_layer(probe, model));

    expect_each_within(result, {-2.0, 1.0, -2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0, 2.0, 1.0},
                       K_PROBE_OPERATIONS);
}

/**
 * @brief Tests that an evaluation and a carrying graph normalise with the stored statistics.
 *
 * The stored mean {-1, 0} and variance {5, 0} with epsilon 4 give the scales 2/3 and 1/2 and the
 * shifts 2/3 and 1, so the batch of the probe test becomes {4/3, 2, 8/3, 2, 4, 2, 16/3, 2}.
 */
TEST_P(Small_StepsFold, AGraphThatFoldsReadsTheStoredStatistics)
{
    BatchNorm2d model(K_VALUE_CHANNELS, NormOptions{K_VALUE_EPSILON, 0.1});
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep step,
                            GetParam().trace(model, value_image()));

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> result, run_layer(step, model));

    expect_each_within(result,
                       {4.0 / 3.0, 2.0, 8.0 / 3.0, 2.0, 4.0, 2.0, 16.0 / 3.0, 2.0},
                       K_FOLD_OPERATIONS);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_StepsFold,
                         ::testing::Values(BuilderRow{"Evaluation", &trace_evaluation},
                                           BuilderRow{"Carrying", &trace_carrying}),
                         builder_row_name);

/** @brief Tests that a probe, an evaluation and a carrying graph publish one output. */
TEST_P(Small_StepsForward, AGraphPublishesTheOneOutputOfTheLayer)
{
    BatchNorm2d model(K_CHANNELS, NormOptions{});
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep step,
                            GetParam().trace(model, image_batch()));

    EXPECT_EQ(step.outputs.all().size(), 1u);
}

/** @brief Tests that a graph feeds the model input alone and carries nothing. */
TEST_P(Small_StepsForward, AGraphFeedsTheModelInputAloneAndCarriesNothing)
{
    BatchNorm2d model(K_CHANNELS, NormOptions{});
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep step,
                            GetParam().trace(model, image_batch()));

    EXPECT_TRUE(step.fed.leaves().empty());
    EXPECT_TRUE(step.carried.values().empty());
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_StepsForward,
                         ::testing::Values(BuilderRow{"Probe", &trace_probe},
                                           BuilderRow{"Evaluation", &trace_evaluation},
                                           BuilderRow{"Carrying", &trace_carrying}),
                         builder_row_name);

/** @brief Tests that three traces of one model bind each parameter under one state identity. */
TEST_P(Small_StepsState, OneModelTracedThreeTimesKeepsOneStateIDPerParameter)
{
    const std::size_t at = GetParam();
    NormClassifier model = make_norm_classifier();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph training,
                            Train::build_training_graph(model, objective, image_batch()));
    const std::vector<Graph::StateID> states = parameter_states(model);

    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep probe,
                            Train::build_probe_graph(model, image_batch()));
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep evaluation,
                            Train::build_evaluation_graph(model, image_batch()));

    // Gamma and beta of the norm, then the weight and the bias of the classifier.
    ASSERT_EQ(states.size(), 4u);
    EXPECT_TRUE(parameter_states(model) == states);
    EXPECT_TRUE(states[at] != Graph::K_NO_STATE);
    EXPECT_NE(training.graph.state_leaf(states[at]), Graph::K_NO_VALUE);
    EXPECT_NE(probe.graph.state_leaf(states[at]), Graph::K_NO_VALUE);
    EXPECT_NE(evaluation.graph.state_leaf(states[at]), Graph::K_NO_VALUE);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_StepsState, ::testing::Values(0u, 1u, 2u, 3u),
                         index_row_name);

/** @brief Tests that a model publishing two values leaves the logits of the step unset. */
TEST(Small_Steps, AModelGraphWithTwoOutputsLeavesLogitsUnset)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, SushiAI::Test::build_two_output_model());
    const NamedOutputObjective objective("logits", K_BATCH, K_CLASSES);

    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));

    EXPECT_EQ(built.spec.logits, Graph::K_NO_VALUE);
    EXPECT_EQ(built.outputs.all().size(), 2u);
    EXPECT_NE(built.spec.loss, Graph::K_NO_VALUE);
}

/** @brief Tests that a builder returns the status of the layer that refused its input. */
TEST(Small_Steps, ARefusingLayerFailsTheBuilderWithItsStatus)
{
    // One value per channel leaves a training batch norm no variance to compute.
    BatchNorm2d model(K_CHANNELS, NormOptions{});
    const Shape one_value_per_channel{1, 1, 1, K_CHANNELS};

    SA_EXPECT_FAILURE(Train::build_probe_graph(model, one_value_per_channel),
                      StatusCode::InvalidDescription, "BatchNorm2d(3)",
                      "more than one value per channel");
}

/** @brief Tests that batch_shape puts the batch before the sample. */
TEST(Small_Steps, BatchShapePrependsTheBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const Shape shape, Train::batch_shape(16, Shape{28, 28, 1}));

    EXPECT_EQ(shape, (Shape{16, 28, 28, 1}));
}

/** @brief Tests that batch_shape refuses a batch of zero. */
TEST(Small_Steps, BatchShapeRefusesABatchThatIsNotPositive)
{
    SA_EXPECT_FAILURE(Train::batch_shape(0, Shape{28, 28, 1}), StatusCode::InvalidDescription,
                      "batch", "positive");
}

/** @brief Tests that the incremental builder refuses a node without an incremental form. */
TEST(Small_Steps, AnIncrementalGraphNamesTheNodeThatHasNoIncrementalForm)
{
    NN::ModelGraph model(Shape{K_FEATURES});
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRef dense,
                            model.add("dense", NN::Linear{K_FEATURES, K_CLASSES}, model.input()));
    SA_EXPECT_OK(model.output("logits", dense));

    SA_EXPECT_FAILURE(Train::build_incremental_graph(model, Shape{1, K_FEATURES}),
                      StatusCode::Unsupported, "node 'dense'", "no incremental form");
}
