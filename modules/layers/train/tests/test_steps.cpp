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
 * @file test_steps.cpp
 * @brief Tests the builders that trace a model into a training, probe or evaluation graph.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/graph/builder.hpp>
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
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;
using NN::BatchNorm2d;
using NN::NormOptions;
using Train::CrossEntropyObjective;

namespace
{
    /** @brief The model whose training graph the golden below was derived for. */
    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

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

    /** @brief The operations build_graph gave for the MLP under cross-entropy, in node order. */
    constexpr std::string_view K_MLP_TRAINING_OPS[] = {
        Ops::K_LEAF.name,          Ops::K_LEAF.name,     Ops::K_LEAF.name,
        Ops::K_LEAF.name,          Ops::K_GEMM_BIAS_GELU.name,
        Ops::K_LEAF.name,          Ops::K_LEAF.name,     Ops::K_GEMM_BIAS.name,
        Ops::K_LOG_SOFTMAX.name,   Ops::K_NLL_LOSS.name, Ops::K_EXP_SUB_SCALE.name,
        Ops::K_SUM_ROWS.name,      Ops::K_MATMUL.name,   Ops::K_MATMUL.name,
        Ops::K_GELU_BACKWARD.name, Ops::K_SUM_ROWS.name, Ops::K_MATMUL.name,
    };

    /** @brief Builds the MLP over K_FEATURES, K_HIDDEN and K_CLASSES. */
    Mlp make_mlp()
    {
        return Mlp(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});
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

    /** @brief Returns the name of the operation of every node of @p graph, in order. */
    std::vector<std::string_view> op_names(const Graph::Graph& graph)
    {
        std::vector<std::string_view> names;
        names.reserve(graph.node_count());
        for (const Graph::Node& node : graph.nodes())
            names.push_back(Ops::name_of(node.op));

        return names;
    }

    /** @brief Counts the nodes of @p graph that carry the operation @p op. */
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

    /** @brief Counts the values of @p graph whose lifetime is Buffer. */
    std::size_t count_buffer_leaves(const Graph::Graph& graph)
    {
        std::size_t count = 0;
        for (std::size_t id = 0; id < graph.value_count(); ++id)
        {
            if (graph.value(static_cast<Graph::ValueId>(id)).lifetime == Graph::Lifetime::Buffer)
                ++count;
        }

        return count;
    }

    /** @brief Returns the state identity of every declared parameter of @p model, in walk order. */
    template<typename M>
    std::vector<Graph::StateId> parameter_states(M& model)
    {
        std::vector<Graph::StateId> states;
        NN::for_each_parameter(model, [&states](const std::string&, NN::Parameter& parameter)
        {
            states.push_back(parameter.state());
        });

        return states;
    }

    using Test::build_two_output_model;

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
            Graph::ValueId loss(const Graph::GraphBuilder& builder, const Graph::Graph&,
                                const NN::ModelOutputs& outputs,
                                const Train::ObjectiveLeaves& leaves) const override
            {
                const Graph::ValueId logits = outputs.at(output_).value();
                return NN::CrossEntropyLoss{}.forward(builder, logits, leaves.at("targets"));
            }

            /** @brief Returns no metric. */
            std::optional<Train::Metric> score(const Data::Batch&,
                                               const SushiBLAS::Tensor&) const override
            {
                return std::nullopt;
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

/**
 * @brief Tests that the training graph of the MLP has the node count and the operation
 *        sequence build_graph gave for it.
 */
TEST(Unit_Steps, ATrainingGraphOfAnMlpHasTheNodesBuildGraphGave)
{
    Mlp model = make_mlp();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));

    const std::vector<std::string_view> golden(std::begin(K_MLP_TRAINING_OPS),
                                               std::end(K_MLP_TRAINING_OPS));

    EXPECT_EQ(built.graph.node_count(), golden.size());
    EXPECT_EQ(op_names(built.graph), golden);
}

/** @brief Tests that every binding names the leaf its parameter has in the built graph. */
TEST(Unit_Steps, ATrainingGraphListsEveryParameterWithItsLeafInThatGraph)
{
    Mlp model = make_mlp();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));

    std::vector<std::string> names;
    std::vector<Graph::ValueId> leaves;
    NN::for_each_parameter(model, [&](const std::string& name, NN::Parameter& parameter)
    {
        names.push_back(name);
        leaves.push_back(parameter.id_in(built.graph));
    });

    // Two layers carry a weight and a bias each.
    ASSERT_EQ(built.parameters.size(), 4u);
    ASSERT_EQ(leaves.size(), built.parameters.size());
    for (std::size_t i = 0; i < built.parameters.size(); ++i)
    {
        EXPECT_EQ(built.parameters[i].name, names[i]);
        EXPECT_NE(built.parameters[i].value, Graph::K_NO_VALUE) << names[i];
        EXPECT_EQ(built.parameters[i].value, leaves[i]) << names[i];
        EXPECT_NE(built.parameters[i].gradient, Graph::K_NO_VALUE) << names[i];
    }
}

/** @brief Tests that a batch-norm model requests two updates and that both sources are pinned. */
TEST(Unit_Steps, ABatchNormModelYieldsTwoUpdatesAndPinsTheirSources)
{
    NormClassifier model = make_norm_classifier();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, image_batch()));

    ASSERT_EQ(built.updates.updates().size(), 2u);

    const SushiRuntime::span<const Graph::ValueId> sources = built.updates.sources();
    for (const Graph::ValueId source : sources)
        EXPECT_NE(std::find(built.pinned.begin(), built.pinned.end(), source),
                  built.pinned.end()) << "source " << source;
}

/** @brief Tests that the loss and every output are pinned, since the host reads them. */
TEST(Unit_Steps, ATrainingGraphPinsTheLossAndTheOutputs)
{
    NormClassifier model = make_norm_classifier();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, image_batch()));

    const auto is_pinned = [&built](Graph::ValueId id)
    {
        return std::find(built.pinned.begin(), built.pinned.end(), id) != built.pinned.end();
    };

    EXPECT_TRUE(is_pinned(built.spec.loss));
    for (const NN::ModelOutputs::Entry& entry : built.outputs.all())
        EXPECT_TRUE(is_pinned(entry.second)) << entry.first;
}

/** @brief Tests that a probe graph normalises with batch statistics and reads no buffer. */
TEST(Unit_Steps, AProbeGraphHasBatchStatisticsAndNoUpdate)
{
    BatchNorm2d model(K_CHANNELS, NormOptions{});
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep probe,
                            Train::build_probe_graph(model, image_batch()));

    EXPECT_EQ(count_nodes(probe.graph, Ops::K_BATCHNORM2D.id), 1u);
    EXPECT_EQ(count_nodes(probe.graph, Ops::K_BATCHNORM_FOLD.id), 0u);
    EXPECT_EQ(count_buffer_leaves(probe.graph), 0u);
    EXPECT_EQ(probe.outputs.all().size(), 1u);
}

/** @brief Tests that an evaluation graph folds the two stored statistics and has no batch form. */
TEST(Unit_Steps, AnEvaluationGraphFoldsTheStoredStatistics)
{
    BatchNorm2d model(K_CHANNELS, NormOptions{});
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep evaluation,
                            Train::build_evaluation_graph(model, image_batch()));

    EXPECT_EQ(count_nodes(evaluation.graph, Ops::K_BATCHNORM_FOLD.id), 1u);
    EXPECT_EQ(count_buffer_leaves(evaluation.graph), 2u);
    EXPECT_EQ(count_nodes(evaluation.graph, Ops::K_BATCHNORM2D.id), 0u);
}

/** @brief Tests that three traces of one model bind each parameter under one state identity. */
TEST(Unit_Steps, OneModelTracedThreeTimesKeepsOneStateIdPerParameter)
{
    NormClassifier model = make_norm_classifier();
    const CrossEntropyObjective objective(K_BATCH, K_CLASSES);

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph training,
                            Train::build_training_graph(model, objective, image_batch()));
    const std::vector<Graph::StateId> states = parameter_states(model);

    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep probe,
                            Train::build_probe_graph(model, image_batch()));
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep evaluation,
                            Train::build_evaluation_graph(model, image_batch()));

    // Gamma and beta of the norm, then the weight and the bias of the classifier.
    ASSERT_EQ(states.size(), 4u);
    EXPECT_TRUE(parameter_states(model) == states);

    for (std::size_t i = 0; i < states.size(); ++i)
    {
        EXPECT_TRUE(states[i] != Graph::K_NO_STATE) << "parameter " << i;
        EXPECT_NE(training.graph.state_leaf(states[i]), Graph::K_NO_VALUE) << "parameter " << i;
        EXPECT_NE(probe.graph.state_leaf(states[i]), Graph::K_NO_VALUE) << "parameter " << i;
        EXPECT_NE(evaluation.graph.state_leaf(states[i]), Graph::K_NO_VALUE)
            << "parameter " << i;
    }
}

/** @brief Tests that a model publishing two values leaves the logits of the step unset. */
TEST(Unit_Steps, AModelGraphWithTwoOutputsLeavesLogitsUnset)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, build_two_output_model());
    const NamedOutputObjective objective("logits", K_BATCH, K_CLASSES);

    SA_ASSERT_OK_AND_ASSIGN(
        const Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));

    EXPECT_EQ(built.spec.logits, Graph::K_NO_VALUE);
    EXPECT_EQ(built.outputs.all().size(), 2u);
    EXPECT_NE(built.spec.loss, Graph::K_NO_VALUE);
}

/** @brief Tests that a builder returns the status of the layer that refused its input. */
TEST(Unit_Steps, ARefusingLayerFailsTheBuilderWithItsStatus)
{
    // One value per channel leaves a training batch norm no variance to compute.
    BatchNorm2d model(K_CHANNELS, NormOptions{});
    const Shape one_value_per_channel{1, 1, 1, K_CHANNELS};

    SA_EXPECT_FAILURE(Train::build_probe_graph(model, one_value_per_channel),
                      StatusCode::InvalidDescription, "BatchNorm2d(3)",
                      "more than one value per channel");
}

/** @brief Tests that batch_shape puts the batch before the sample and refuses a batch of zero. */
TEST(Unit_Steps, BatchShapePrependsTheBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const Shape shape, Train::batch_shape(16, Shape{28, 28, 1}));
    EXPECT_EQ(shape, (Shape{16, 28, 28, 1}));

    SA_EXPECT_FAILURE(Train::batch_shape(0, Shape{28, 28, 1}), StatusCode::InvalidDescription,
                      "batch", "positive");
}
