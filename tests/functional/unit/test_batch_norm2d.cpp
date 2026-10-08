/**************************************************************************/
/* test_batch_norm2d.cpp                                                  */
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
 * @file test_batch_norm2d.cpp
 * @brief Tests BatchNorm2d's two trace forms and the running-statistic update.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include "../../common/reference_fixture.hpp"
#include "../../common/test_status.hpp"

using namespace SushiAI;
using NN::BatchNorm2d;
using NN::BufferUpdateList;
using NN::NormOptions;
using NN::Trace;

namespace
{
    /** @brief Holds the epsilon the reference layers were run with. */
    constexpr double K_REFERENCE_EPSILON = 1e-3;

    /** @brief Holds the momentum the reference layers were run with. */
    constexpr double K_REFERENCE_MOMENTUM = 0.03;

    /** @brief Holds 4 float32 epsilons of the unit case's largest running variance, 1.1086. */
    constexpr float K_UNIT_CASE_TOLERANCE = 5.3e-7f;

    /** @brief Holds 4 float32 epsilons of the model case's largest running variance, 1.4578. */
    constexpr float K_MODEL_CASE_TOLERANCE = 7.0e-7f;

    /** @brief Holds the absolute tolerance of a double computed the same way on both sides. */
    constexpr double K_EXACT_TOLERANCE = 1e-15;

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

    /** @brief Returns the first node of @p graph that carries @p op; the caller counted one. */
    const Graph::Node& only_node(const Graph::Graph& graph, OpID op)
    {
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                return node;
        }

        return graph.nodes().front();
    }

    /** @brief Counts the values of @p graph whose lifetime is BUFFER. */
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

    /** @brief Holds the per-channel mean and biased variance of one NHWC activation. */
    struct ChannelStatistics
    {
        std::vector<double> mean;     /**< The mean over batch, height and width. */
        std::vector<double> variance; /**< The biased variance over the same values. */
    };

    /** @brief Computes the statistics of @p values, stored NHWC with @p channels last. */
    ChannelStatistics batch_statistics(const std::vector<float>& values, std::size_t channels)
    {
        const std::size_t count = values.size() / channels;
        ChannelStatistics statistics{std::vector<double>(channels, 0.0),
                                     std::vector<double>(channels, 0.0)};

        for (std::size_t i = 0; i < values.size(); ++i)
            statistics.mean[i % channels] += static_cast<double>(values[i]);
        for (double& mean : statistics.mean)
            mean /= static_cast<double>(count);

        for (std::size_t i = 0; i < values.size(); ++i)
        {
            const double gap = static_cast<double>(values[i]) - statistics.mean[i % channels];
            statistics.variance[i % channels] += gap * gap;
        }
        for (double& variance : statistics.variance)
            variance /= static_cast<double>(count);

        return statistics;
    }

    /** @brief Applies updated_statistic per channel and rounds each result to float32. */
    std::vector<float> updated_vector(const std::vector<float>& running,
                                      const std::vector<double>& batch, double momentum,
                                      double source_scale)
    {
        std::vector<float> updated(running.size());
        for (std::size_t c = 0; c < running.size(); ++c)
        {
            updated[c] = static_cast<float>(NN::updated_statistic(
                static_cast<double>(running[c]), batch[c], momentum, source_scale));
        }

        return updated;
    }

    /** @brief Returns a view of @p values for near_reference. */
    SushiRuntime::span<const float> view(const std::vector<float>& values)
    {
        return SushiRuntime::span<const float>(values.data(), values.size());
    }

    /**
     * @brief Checks one case of batchnorm.sacp against the updates a training trace requests.
     * @param case_name The part of the tensor names between "batchnorm_factor." and the suffix.
     * @param tolerance Largest absolute difference from the running statistics torch left.
     */
    void expect_reference_case(const std::string& case_name, float tolerance)
    {
        SA_OPEN_REFERENCE_OR_SKIP(file, "batchnorm.sacp");
        const std::string prefix = "batchnorm_factor." + case_name + ".";
        SA_READ_REFERENCE_OR_FAIL(input, file, prefix + "input");
        SA_READ_REFERENCE_OR_FAIL(mean_before, file, prefix + "before.running_mean");
        SA_READ_REFERENCE_OR_FAIL(variance_before, file, prefix + "before.running_var");
        SA_READ_REFERENCE_OR_FAIL(mean_after, file, prefix + "after.running_mean");
        SA_READ_REFERENCE_OR_FAIL(variance_after, file, prefix + "after.running_var");
        ASSERT_EQ(input.shape.rank(), 4);

        const int64_t channels = input.shape[3];
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        BatchNorm2d layer(channels, NormOptions{K_REFERENCE_EPSILON, K_REFERENCE_MOMENTUM});
        BufferUpdateList list;
        static_cast<void>(layer.trace_training(Trace::training(builder, list),
                                               builder.input(input.shape, "x")));
        ASSERT_EQ(list.updates().size(), 2u);
        const NN::BufferUpdate mean_update = list.updates()[0];
        const NN::BufferUpdate variance_update = list.updates()[1];

        const ChannelStatistics batch =
            batch_statistics(input.floats(), static_cast<std::size_t>(channels));
        const std::vector<float> mean_reference = mean_after.floats();
        const std::vector<float> variance_reference = variance_after.floats();

        const std::vector<float> mean = updated_vector(
            mean_before.floats(), batch.mean, mean_update.momentum, mean_update.source_scale);
        const std::vector<float> variance = updated_vector(
            variance_before.floats(), batch.variance, variance_update.momentum,
            variance_update.source_scale);
        const std::vector<float> biased = updated_vector(
            variance_before.floats(), batch.variance, variance_update.momentum, 1.0);

        EXPECT_TRUE(Test::near_reference(view(mean), view(mean_reference), tolerance));
        EXPECT_TRUE(Test::near_reference(view(variance), view(variance_reference), tolerance));

        // The fixture separates the two forms only if the unscaled variance misses it.
        EXPECT_FALSE(Test::near_reference(view(biased), view(variance_reference), tolerance));
    }

    /** @brief Traces the training form, expects the input back, returns what was recorded. */
    Status refused_training(BatchNorm2d& layer, const Graph::GraphBuilder& builder,
                            BufferUpdateList& list, Graph::ValueId x)
    {
        const Trace trace = Trace::training(builder, list);
        EXPECT_EQ(layer.trace_training(trace, x), x);
        return trace.status();
    }

    /** @brief Traces the evaluation form, expects the input back, returns what was recorded. */
    Status refused_evaluation(BatchNorm2d& layer, const Graph::GraphBuilder& builder,
                              Graph::ValueId x)
    {
        const Trace trace = Trace::evaluation(builder);
        EXPECT_EQ(layer.trace_evaluation(trace, x), x);
        return trace.status();
    }

    /** @brief Calls the training form under an evaluation trace, for a death test. */
    void train_under_an_evaluation_trace()
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        BatchNorm2d layer(3, NormOptions{});
        static_cast<void>(layer.trace_training(Trace::evaluation(builder),
                                               builder.input(Shape{2, 2, 2, 3}, "x")));
    }

    /** @brief Traces the evaluation form over an id the graph never issued, for a death test. */
    void evaluate_over_a_value_the_graph_lacks()
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        BatchNorm2d layer(3, NormOptions{});
        static_cast<void>(builder.input(Shape{2, 2, 2, 3}, "x"));
        static_cast<void>(layer.trace_evaluation(Trace::evaluation(builder), Graph::ValueId{7}));
    }

} // namespace

static_assert(NN::ModalModule<BatchNorm2d> && !NN::Module<BatchNorm2d> &&
                  !NN::TraceModule<BatchNorm2d>,
              "BatchNorm2d has the modal form alone");
static_assert(!NN::Module<NN::Sequential<NN::Conv2d, BatchNorm2d>> &&
                  NN::TraceModule<NN::Sequential<NN::Conv2d, BatchNorm2d>>,
              "a Sequential holding BatchNorm2d is traced through a trace only");

/** @brief Tests that a training trace emits one BATCHNORM2D with the layer's epsilon. */
TEST(Unit_BatchNorm2d, ATrainingTraceEmitsOneBatchNorm)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 2, 2, 3}, "x");
    BatchNorm2d layer(3, NormOptions{1e-3, 0.03});
    BufferUpdateList list;
    const std::size_t nodes_before = graph.node_count();

    const Graph::ValueId y = layer.trace_training(Trace::training(builder, list), x);

    ASSERT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM_FOLD.id), 0u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CHANNEL_AFFINE.id), 0u);

    // The trace adds the two parameter leaves and the BATCHNORM2D, and no buffer leaf.
    EXPECT_EQ(graph.node_count(), nodes_before + 3u);

    const Graph::Node& node = only_node(graph, Ops::K_BATCHNORM2D.id);
    ASSERT_EQ(node.inputs.size(), 3u);
    ASSERT_EQ(node.outputs.size(), 4u);
    EXPECT_EQ(node.inputs[0], x);
    EXPECT_EQ(node.inputs[1], layer.weight.id_in(graph));
    EXPECT_EQ(node.inputs[2], layer.bias.id_in(graph));
    EXPECT_EQ(node.outputs[0], y);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 2, 2, 3}));

    // The epsilon is stored as given, so it compares exactly.
    EXPECT_EQ(node.attributes.values[0], 1e-3);
}

/** @brief Tests that a training trace adds no buffer leaf to its graph. */
TEST(Unit_BatchNorm2d, ATrainingTraceReadsNoBuffer)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 2, 2, 3}, "x");
    BatchNorm2d layer(3, NormOptions{});
    BufferUpdateList list;

    static_cast<void>(layer.trace_training(Trace::training(builder, list), x));

    EXPECT_EQ(count_buffer_leaves(graph), 0u);
    EXPECT_EQ(layer.running_mean.id_in(graph), Graph::K_NO_VALUE);
    EXPECT_EQ(layer.running_var.id_in(graph), Graph::K_NO_VALUE);
}

/**
 * @brief Tests that a training trace requests two updates, from outputs 1 and 3 of the node
 *        into the running mean and the running variance.
 */
TEST(Unit_BatchNorm2d, ATrainingTraceRequestsTheTwoRunningUpdates)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 2, 2, 3}, "x");
    BatchNorm2d layer(3, NormOptions{1e-3, 0.03});
    BufferUpdateList list;

    static_cast<void>(layer.trace_training(Trace::training(builder, list), x));

    ASSERT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 1u);
    const Graph::Node& node = only_node(graph, Ops::K_BATCHNORM2D.id);
    ASSERT_EQ(node.outputs.size(), 4u);
    ASSERT_EQ(list.updates().size(), 2u);

    EXPECT_EQ(list.updates()[0].source, node.outputs[1]);
    EXPECT_EQ(list.updates()[1].source, node.outputs[3]);
    EXPECT_NE(layer.running_mean.state(), Graph::K_NO_STATE);
    EXPECT_NE(layer.running_var.state(), Graph::K_NO_STATE);
    EXPECT_NE(layer.running_mean.state(), layer.running_var.state());
    EXPECT_EQ(list.updates()[0].target, layer.running_mean.state());
    EXPECT_EQ(list.updates()[1].target, layer.running_var.state());

    // The momentum and the mean's scale are stored as given, so they compare exactly.
    EXPECT_NEAR(list.updates()[0].momentum, 0.03, K_EXACT_TOLERANCE);
    EXPECT_NEAR(list.updates()[1].momentum, 0.03, K_EXACT_TOLERANCE);
    EXPECT_NEAR(list.updates()[0].source_scale, 1.0, K_EXACT_TOLERANCE);
}

/** @brief Tests that the variance update carries n / (n - 1) for n values per channel. */
TEST(Unit_BatchNorm2d, TheVarianceUpdateCarriesTheUnbiasedFactor)
{
    Graph::Graph small_graph;
    const Graph::GraphBuilder small_builder(small_graph);
    BatchNorm2d small(3, NormOptions{});
    BufferUpdateList small_list;
    static_cast<void>(small.trace_training(Trace::training(small_builder, small_list),
                                           small_builder.input(Shape{2, 2, 2, 3}, "x")));

    Graph::Graph large_graph;
    const Graph::GraphBuilder large_builder(large_graph);
    BatchNorm2d large(128, NormOptions{});
    BufferUpdateList large_list;
    static_cast<void>(large.trace_training(Trace::training(large_builder, large_list),
                                           large_builder.input(Shape{1, 20, 20, 128}, "x")));

    ASSERT_EQ(small_list.updates().size(), 2u);
    ASSERT_EQ(large_list.updates().size(), 2u);
    EXPECT_NEAR(small_list.updates()[1].source_scale, 8.0 / 7.0, K_EXACT_TOLERANCE);
    EXPECT_NEAR(large_list.updates()[1].source_scale, 400.0 / 399.0, K_EXACT_TOLERANCE);
}

/**
 * @brief Tests that an evaluation trace emits one BATCHNORM_FOLD over the two buffer leaves
 *        and one CHANNEL_AFFINE over its results, and no BATCHNORM2D.
 */
TEST(Unit_BatchNorm2d, AnEvaluationTraceFoldsTheRunningStatistics)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 3}, "x");
    BatchNorm2d layer(3, NormOptions{1e-3, 0.03});
    const std::size_t nodes_before = graph.node_count();

    const Graph::ValueId y = layer.trace_evaluation(Trace::evaluation(builder), x);

    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 0u);
    ASSERT_EQ(count_nodes(graph, Ops::K_BATCHNORM_FOLD.id), 1u);
    ASSERT_EQ(count_nodes(graph, Ops::K_CHANNEL_AFFINE.id), 1u);
    EXPECT_EQ(count_buffer_leaves(graph), 2u);

    // The trace adds two parameter leaves, two buffer leaves, the fold and the affine.
    EXPECT_EQ(graph.node_count(), nodes_before + 6u);

    const Graph::Node& fold = only_node(graph, Ops::K_BATCHNORM_FOLD.id);
    ASSERT_EQ(fold.inputs.size(), 4u);
    ASSERT_EQ(fold.outputs.size(), 2u);
    EXPECT_EQ(fold.inputs[0], layer.weight.id_in(graph));
    EXPECT_EQ(fold.inputs[1], layer.bias.id_in(graph));
    EXPECT_EQ(fold.inputs[2], layer.running_mean.id_in(graph));
    EXPECT_EQ(fold.inputs[3], layer.running_var.id_in(graph));
    EXPECT_EQ(graph.value(fold.inputs[2]).lifetime, Graph::Lifetime::Buffer);
    EXPECT_EQ(graph.value(fold.inputs[3]).lifetime, Graph::Lifetime::Buffer);

    // The epsilon is stored as given, so it compares exactly.
    EXPECT_EQ(fold.attributes.values[0], 1e-3);

    const Graph::Node& affine = only_node(graph, Ops::K_CHANNEL_AFFINE.id);
    ASSERT_EQ(affine.inputs.size(), 3u);
    ASSERT_EQ(affine.outputs.size(), 1u);
    EXPECT_EQ(affine.inputs[0], x);
    EXPECT_EQ(affine.inputs[1], fold.outputs[0]);
    EXPECT_EQ(affine.inputs[2], fold.outputs[1]);
    EXPECT_EQ(affine.outputs[0], y);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 3}));
}

/**
 * @brief Tests that the buffers a training trace updates are the ones an evaluation trace of
 *        the same layer reads, and that both traces share the parameters.
 */
TEST(Unit_BatchNorm2d, BothFormsNameTheSameState)
{
    BatchNorm2d layer(3, NormOptions{});

    Graph::Graph training_graph;
    const Graph::GraphBuilder training_builder(training_graph);
    BufferUpdateList list;
    static_cast<void>(layer.trace_training(Trace::training(training_builder, list),
                                           training_builder.input(Shape{2, 2, 2, 3}, "x")));
    ASSERT_EQ(list.updates().size(), 2u);
    const Graph::StateId gamma = layer.weight.state();
    const Graph::StateId beta = layer.bias.state();

    Graph::Graph evaluation_graph;
    const Graph::GraphBuilder evaluation_builder(evaluation_graph);
    static_cast<void>(layer.trace_evaluation(Trace::evaluation(evaluation_builder),
                                             evaluation_builder.input(Shape{1, 2, 2, 3}, "x")));

    EXPECT_EQ(evaluation_graph.state_leaf(list.updates()[0].target),
              layer.running_mean.id_in(evaluation_graph));
    EXPECT_EQ(evaluation_graph.state_leaf(list.updates()[1].target),
              layer.running_var.id_in(evaluation_graph));
    EXPECT_NE(layer.running_mean.id_in(evaluation_graph), Graph::K_NO_VALUE);
    EXPECT_NE(layer.running_var.id_in(evaluation_graph), Graph::K_NO_VALUE);
    EXPECT_EQ(layer.weight.state(), gamma);
    EXPECT_EQ(layer.bias.state(), beta);
    EXPECT_NE(layer.weight.id_in(training_graph), Graph::K_NO_VALUE);
    EXPECT_NE(layer.weight.id_in(evaluation_graph), Graph::K_NO_VALUE);
    EXPECT_NE(layer.bias.id_in(evaluation_graph), Graph::K_NO_VALUE);
}

/**
 * @brief Tests that the state walk yields weight, bias, running_mean and running_var in that
 *        order, each [channels], starting at one, zero, zero and one.
 */
TEST(Unit_BatchNorm2d, TheWalkYieldsTwoParametersThenTwoBuffers)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    BatchNorm2d layer(5, NormOptions{});
    BufferUpdateList list;
    static_cast<void>(layer.trace_training(Trace::training(builder, list),
                                           builder.input(Shape{2, 2, 2, 5}, "x")));

    std::vector<std::string> names;
    const auto on_parameter = [&names](const std::string& name, NN::Parameter& parameter)
    {
        names.push_back(name);
        EXPECT_EQ(parameter.shape(), (Shape{5}));
        EXPECT_EQ(parameter.init().kind, NN::ParameterInit::Kind::Constant);
    };
    const auto on_buffer = [&names](const std::string& name, NN::Buffer& buffer)
    {
        names.push_back(name);
        EXPECT_EQ(buffer.shape(), (Shape{5}));
    };
    const NN::StateVisitor visitor(on_parameter, on_buffer);
    NN::walk_state(layer, "bn", visitor);

    EXPECT_EQ(names, (std::vector<std::string>{"bn.weight", "bn.bias", "bn.running_mean",
                                               "bn.running_var"}));

    // The four starting values are stored as given, so they compare exactly.
    EXPECT_NEAR(layer.weight.init().constant, 1.0, K_EXACT_TOLERANCE);
    EXPECT_NEAR(layer.bias.init().constant, 0.0, K_EXACT_TOLERANCE);
    EXPECT_NEAR(layer.running_mean.initial_value(), 0.0, K_EXACT_TOLERANCE);
    EXPECT_NEAR(layer.running_var.initial_value(), 1.0, K_EXACT_TOLERANCE);
}

/** @brief Tests that training over one value per channel fails and declares nothing. */
TEST(Unit_BatchNorm2d, TrainingOverOneValuePerChannelFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{1, 1, 1, 3}, "x");
    BatchNorm2d layer(3, NormOptions{});
    BufferUpdateList list;
    const std::size_t before = graph.value_count();
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(refused_training(layer, builder, list, x), StatusCode::InvalidDescription,
                      "BatchNorm2d(3)", "more than one value per channel", "got 1");

    EXPECT_EQ(graph.value_count(), before);
    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_EQ(list.updates().size(), 0u);
    EXPECT_FALSE(layer.weight.declared());
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that both forms refuse an integer input and declare nothing. */
TEST(Unit_BatchNorm2d, AnIntegerInputFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 2, 2, 3}, "x", DType::Int32);
    BatchNorm2d layer(3, NormOptions{});
    BufferUpdateList list;
    const std::size_t before = graph.value_count();
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(refused_training(layer, builder, list, x), StatusCode::InvalidDescription,
                      "BatchNorm2d(3)", "half, float32 or float64 input", "int32");
    SA_EXPECT_FAILURE(refused_evaluation(layer, builder, x), StatusCode::InvalidDescription,
                      "BatchNorm2d(3)", "half, float32 or float64 input", "int32");

    EXPECT_EQ(graph.value_count(), before);
    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_EQ(list.updates().size(), 0u);
    EXPECT_FALSE(layer.weight.declared());
    EXPECT_FALSE(layer.running_mean.declared());
}

/** @brief Tests that an evaluation trace accepts one value per channel. */
TEST(Unit_BatchNorm2d, EvaluationOverOneValuePerChannelTraces)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{1, 1, 1, 3}, "x");
    BatchNorm2d layer(3, NormOptions{});

    const Graph::ValueId y = layer.trace_evaluation(Trace::evaluation(builder), x);

    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, 1, 3}));
}

/** @brief Tests that both forms refuse an input of another channel count or rank. */
TEST(Unit_BatchNorm2d, AnInputThatDoesNotFitFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId wide = builder.input(Shape{2, 2, 2, 4}, "wide");
    const Graph::ValueId flat = builder.input(Shape{2, 3}, "flat");
    BatchNorm2d layer(3, NormOptions{});
    BufferUpdateList list;
    const std::size_t before = graph.value_count();

    SA_EXPECT_FAILURE(refused_training(layer, builder, list, wide),
                      StatusCode::InvalidDescription, "BatchNorm2d(3)", "[N, H, W, 3]");
    SA_EXPECT_FAILURE(refused_evaluation(layer, builder, wide), StatusCode::InvalidDescription,
                      "BatchNorm2d(3)", "[N, H, W, 3]");
    SA_EXPECT_FAILURE(refused_training(layer, builder, list, flat),
                      StatusCode::InvalidDescription, "BatchNorm2d(3)", "rank-4");
    SA_EXPECT_FAILURE(refused_evaluation(layer, builder, flat), StatusCode::InvalidDescription,
                      "BatchNorm2d(3)", "rank-4");

    EXPECT_EQ(graph.value_count(), before);
    EXPECT_EQ(list.updates().size(), 0u);
}

/** @brief Tests that both forms refuse an NCHW input before they declare a value. */
TEST(Unit_BatchNorm2d, AnNchwInputFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{3, 3, 3, 3}, "x");
    const Graph::ValueId nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);
    BatchNorm2d layer(3, NormOptions{});
    BufferUpdateList list;
    const std::size_t before = graph.value_count();

    SA_EXPECT_FAILURE(refused_training(layer, builder, list, nchw),
                      StatusCode::InvalidDescription, "BatchNorm2d(3)", "NHWC");
    SA_EXPECT_FAILURE(refused_evaluation(layer, builder, nchw), StatusCode::InvalidDescription,
                      "BatchNorm2d(3)", "NHWC");

    EXPECT_EQ(graph.value_count(), before);
}

/** @brief Tests that a layer without channels, epsilon or a momentum in [0, 1] fails to trace. */
TEST(Unit_BatchNorm2d, ALayerDescribedWrongFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 2, 2, 3}, "x");
    const std::size_t before = graph.value_count();

    BatchNorm2d unset;
    BatchNorm2d flat_epsilon(3, NormOptions{0.0, 0.1});
    BatchNorm2d wide_momentum(3, NormOptions{1e-5, 1.5});

    SA_EXPECT_FAILURE(refused_evaluation(unset, builder, x), StatusCode::InvalidDescription,
                      "positive channel count", "got 0");
    SA_EXPECT_FAILURE(refused_evaluation(flat_epsilon, builder, x),
                      StatusCode::InvalidDescription, "BatchNorm2d(3)", "positive epsilon");
    SA_EXPECT_FAILURE(refused_evaluation(wide_momentum, builder, x),
                      StatusCode::InvalidDescription, "BatchNorm2d(3)", "momentum in [0, 1]",
                      "1.5");

    EXPECT_EQ(graph.value_count(), before);
}

/** @brief Tests that a second trace under another element type fails and names both types. */
TEST(Unit_BatchNorm2d, ASecondTraceUnderAnotherElementTypeFails)
{
    Graph::Graph first;
    const Graph::GraphBuilder first_builder(first);
    BatchNorm2d layer(3, NormOptions{});
    const Trace accepted = Trace::evaluation(first_builder);
    static_cast<void>(
        layer.trace_evaluation(accepted, first_builder.input(Shape{2, 2, 2, 3}, "x")));
    SA_EXPECT_OK(accepted.status());

    Graph::Graph second;
    const Graph::GraphBuilder second_builder(second);
    const Graph::ValueId wide =
        second_builder.input(Shape{2, 2, 2, 3}, "x", DType::Float64);

    SA_EXPECT_FAILURE(refused_evaluation(layer, second_builder, wide),
                      StatusCode::InvalidDescription, "'weight'", dtype_name(DType::Float32),
                      dtype_name(DType::Float64));
    EXPECT_EQ(layer.running_mean.dtype(), DType::Float32);
}

/** @brief Tests that gamma is a SCALE that starts at one and beta a BIAS that starts at zero. */
TEST(Unit_BatchNorm2d, TheScaleStartsAtOneAndTheShiftAtZero)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    BatchNorm2d layer(3, NormOptions{});
    static_cast<void>(layer.trace_evaluation(Trace::evaluation(builder),
                                             builder.input(Shape{2, 2, 2, 3}, "x")));

    EXPECT_EQ(layer.weight.role(), NN::ParameterRole::Scale);
    EXPECT_EQ(layer.weight.init().kind, NN::ParameterInit::Kind::Constant);
    EXPECT_NEAR(layer.weight.init().constant, 1.0, K_EXACT_TOLERANCE);
    EXPECT_EQ(layer.bias.role(), NN::ParameterRole::Bias);
    EXPECT_EQ(layer.bias.init().kind, NN::ParameterInit::Kind::Constant);
    EXPECT_NEAR(layer.bias.init().constant, 0.0, K_EXACT_TOLERANCE);
}

/** @brief Tests that the factory takes the channels from its input and reads both constants. */
TEST(Unit_BatchNorm2d, TheFactoryReadsTheChannelsAndTheTwoConstants)
{
    const Shape input{8, 8, 6};
    const Config::Value attributes =
        Config::parse_json(R"({"bn_epsilon": 0.001, "bn_momentum": 0.03})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        NN::make_batch_norm2d_node(NN::ShapeSpan(&input, 1), attributes, "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<BatchNorm2d>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().channels, 6);
    EXPECT_NEAR(wrapped->layer().options.epsilon, 0.001, K_EXACT_TOLERANCE);
    EXPECT_NEAR(wrapped->layer().options.momentum, 0.03, K_EXACT_TOLERANCE);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, input);
    EXPECT_EQ(node->input_count(), 1u);
}

/** @brief Tests that the factory refuses an unknown key, a flat input and a wrong constant. */
TEST(Unit_BatchNorm2d, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape image{8, 8, 6};
    const Shape flat{48};
    const Config::Value none = Config::parse_json("{}", "<test>");
    const Config::Value stray = Config::parse_json(R"({"channels": 6})", "<test>");
    const Config::Value negative = Config::parse_json(R"({"bn_epsilon": -1.0})", "<test>");

    SA_EXPECT_FAILURE(NN::make_batch_norm2d_node(NN::ShapeSpan(&image, 1), stray, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'channels'", "bn_epsilon");
    SA_EXPECT_FAILURE(NN::make_batch_norm2d_node(NN::ShapeSpan(&flat, 1), none, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "[48]", "[H, W, C]");
    SA_EXPECT_FAILURE(NN::make_batch_norm2d_node(NN::ShapeSpan(&image, 1), negative, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'bn_epsilon'", "positive");
}

/** @brief Tests that the training form under an evaluation trace stops the process. */
TEST(Unit_BatchNorm2dDeathTest, TheTrainingFormUnderAnEvaluationTraceIsFatal)
{
    SA_EXPECT_FATAL(train_under_an_evaluation_trace(), "trace_training",
                    "was given an evaluation trace");
}

/** @brief Tests that a trace over a value the graph does not hold stops the process. */
TEST(Unit_BatchNorm2dDeathTest, ATraceOverAValueTheGraphLacksIsFatal)
{
    SA_EXPECT_FATAL(evaluate_over_a_value_the_graph_lacks(), "BatchNorm2d(3)", "value 7",
                    "holds 1 values");
}

/** @brief Tests that trace_layer picks the form of the trace's mode for BatchNorm2d. */
TEST(Unit_BatchNorm2d, TraceLayerPicksTheFormOfTheMode)
{
    Graph::Graph training_graph;
    const Graph::GraphBuilder training_builder(training_graph);
    BatchNorm2d layer(3, NormOptions{});
    static_cast<void>(NN::trace_layer(
        layer, Trace::training(training_builder, NN::discard_updates()),
        training_builder.input(Shape{2, 2, 2, 3}, "x")));

    Graph::Graph evaluation_graph;
    const Graph::GraphBuilder evaluation_builder(evaluation_graph);
    static_cast<void>(NN::trace_layer(layer, Trace::evaluation(evaluation_builder),
                                      evaluation_builder.input(Shape{2, 2, 2, 3}, "x")));

    EXPECT_EQ(count_nodes(training_graph, Ops::K_BATCHNORM2D.id), 1u);
    EXPECT_EQ(count_nodes(training_graph, Ops::K_BATCHNORM_FOLD.id), 0u);
    EXPECT_EQ(count_nodes(evaluation_graph, Ops::K_BATCHNORM2D.id), 0u);
    EXPECT_EQ(count_nodes(evaluation_graph, Ops::K_BATCHNORM_FOLD.id), 1u);
}

/** @brief Tests that output_shape returns the shape it is given, whatever that shape is. */
TEST(Unit_BatchNorm2d, OutputShapeIsTheInputShape)
{
    const BatchNorm2d layer(3, NormOptions{});

    EXPECT_EQ(layer.output_shape(Shape{7, 5, 3}), (Shape{7, 5, 3}));
    EXPECT_EQ(layer.output_shape(Shape{7, 5, 4}), (Shape{7, 5, 4}));
}

/**
 * @brief Tests that updated_statistic returns (1 - momentum) * running + momentum * scale *
 *        batch, on numbers a double holds exactly.
 */
TEST(Unit_BatchNorm2d, UpdatedStatisticBlendsTheScaledBatchValue)
{
    EXPECT_NEAR(NN::updated_statistic(1.0, 3.0, 0.25, 2.0), 2.25, K_EXACT_TOLERANCE);
    EXPECT_NEAR(NN::updated_statistic(4.0, 8.0, 0.5, 1.0), 6.0, K_EXACT_TOLERANCE);
    EXPECT_NEAR(NN::updated_statistic(4.0, 8.0, 0.0, 1.0), 4.0, K_EXACT_TOLERANCE);
    EXPECT_NEAR(NN::updated_statistic(4.0, 8.0, 1.0, 1.0), 8.0, K_EXACT_TOLERANCE);
}

/**
 * @brief Tests that the requested updates reproduce the running statistics torch left on the
 *        [2, 2, 2, 3] reference case, where the unscaled variance misses by 8.3e-3.
 */
TEST(Unit_BatchNorm2d, TheUpdatesReproduceTheUnitReferenceCase)
{
    expect_reference_case("unit", K_UNIT_CASE_TOLERANCE);
}

/**
 * @brief Tests that the requested updates reproduce the running statistics torch left on
 *        model.9.cv1.bn, where the unscaled variance misses by 1.4e-5.
 */
TEST(Unit_BatchNorm2d, TheUpdatesReproduceTheModelReferenceCase)
{
    expect_reference_case("model.9.cv1.bn", K_MODEL_CASE_TOLERANCE);
}
