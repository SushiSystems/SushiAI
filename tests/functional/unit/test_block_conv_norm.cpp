/**************************************************************************/
/* test_block_conv_norm.cpp                                               */
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
 * @file test_block_conv_norm.cpp
 * @brief Tests ConvNorm's trace, its state names against YOLO26n row 0, and make_conv_node.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/blocks/conv_norm.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include "../../common/test_status.hpp"

using namespace SushiAI;
using NN::BufferUpdateList;
using NN::Conv;
using NN::ConvLinear;
using NN::NormOptions;
using NN::Trace;

namespace
{
    /** @brief Holds the absolute tolerance of a double stored and read back unchanged. */
    constexpr double K_EXACT_TOLERANCE = 1e-15;

    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 0 'conv'";

    /** @brief Names which of the two kinds of state a record is. */
    enum class StateKind : std::uint8_t
    {
        Parameter, /**< A learnable tensor. */
        Buffer,    /**< A stored tensor no gradient reaches. */
    };

    /** @brief Writes one piece of state as "name [extents] kind", for a readable mismatch. */
    std::string record(const std::string& name, const Shape& shape, StateKind kind)
    {
        return name + " " + shape.to_string() +
               (kind == StateKind::Buffer ? " buffer" : " parameter");
    }

    /** @brief Lists every piece of state under @p module, in the order the walk yields. */
    template<typename M>
    std::vector<std::string> walk(M& module, std::string_view prefix)
    {
        std::vector<std::string> records;
        const auto on_parameter = [&records](const std::string& name, NN::Parameter& state)
        {
            records.push_back(record(name, state.shape(), StateKind::Parameter));
        };
        const auto on_buffer = [&records](const std::string& name, NN::Buffer& state)
        {
            records.push_back(record(name, state.shape(), StateKind::Buffer));
        };
        const NN::StateVisitor visitor(on_parameter, on_buffer);
        NN::walk_state(module, prefix, visitor);

        return records;
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

    /** @brief Returns the index of the first node of @p graph that carries @p op. */
    std::size_t first_node(const Graph::Graph& graph, OpID op)
    {
        std::size_t index = 0;
        while (index < graph.node_count() && graph.nodes()[index].op != op)
            ++index;

        return index;
    }

    /** @brief Views one shape as the inputs of a one-input row. */
    NN::ShapeSpan one_input(const Shape& shape)
    {
        return NN::ShapeSpan(&shape, static_cast<std::size_t>(1));
    }

    /** @brief Parses @p text as the attributes of one row. */
    Config::Value attributes_of(std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    }

} // namespace

/** @brief Tests that a 3x3 Conv at stride 2 halves an even H and W and sets the width. */
TEST(Unit_BlockConvNorm, OutputShapeHalvesAnEvenExtentAtStrideTwo)
{
    const Conv block(3, 16, 3, 2, K_YOLO_NORM);

    EXPECT_EQ(block.output_shape(Shape{640, 640, 3}), (Shape{320, 320, 16}));
}

/** @brief Tests that a 1x1 Conv at stride 1 keeps H and W. */
TEST(Unit_BlockConvNorm, OutputShapeOfAPointwiseBlockKeepsTheExtents)
{
    const Conv block(32, 64, 1, 1, K_YOLO_NORM);

    EXPECT_EQ(block.output_shape(Shape{7, 9, 32}), (Shape{7, 9, 64}));
}

/**
 * @brief Tests that a training trace emits one convolution, one BATCHNORM2D and one SILU, in
 *        that order, and no bias.
 */
TEST(Unit_BlockConvNorm, ATrainingTraceEmitsConvolutionNormThenSilu)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 3}, "x");
    Conv block(3, 16, 3, 2, K_YOLO_NORM);
    BufferUpdateList list;

    const Graph::ValueId y = block.forward(Trace::training(builder, list), x);

    ASSERT_EQ(count_nodes(graph, Ops::MATMUL.id), 1u);
    ASSERT_EQ(count_nodes(graph, Ops::BATCHNORM2D.id), 1u);
    ASSERT_EQ(count_nodes(graph, Ops::SILU.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::BIAS_ADD.id), 0u);
    EXPECT_LT(first_node(graph, Ops::MATMUL.id), first_node(graph, Ops::BATCHNORM2D.id));
    EXPECT_LT(first_node(graph, Ops::BATCHNORM2D.id), first_node(graph, Ops::SILU.id));

    const Graph::Node& activation = graph.nodes()[first_node(graph, Ops::SILU.id)];
    EXPECT_EQ(activation.outputs[0], y);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 16}));
}

/** @brief Tests that a training trace hands the norm's options to the node and the updates. */
TEST(Unit_BlockConvNorm, ATrainingTraceCarriesTheNormOptions)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 3}, "x");
    Conv block(3, 16, 3, 2, K_YOLO_NORM);
    BufferUpdateList list;

    static_cast<void>(block.forward(Trace::training(builder, list), x));

    ASSERT_EQ(count_nodes(graph, Ops::BATCHNORM2D.id), 1u);
    const Graph::Node& norm = graph.nodes()[first_node(graph, Ops::BATCHNORM2D.id)];

    // Both constants are stored as given, so they compare exactly.
    EXPECT_EQ(norm.attributes.values[0], 1e-3);
    ASSERT_EQ(list.updates().size(), 2u);
    EXPECT_NEAR(list.updates()[0].momentum, 0.03, K_EXACT_TOLERANCE);
    EXPECT_NEAR(list.updates()[1].momentum, 0.03, K_EXACT_TOLERANCE);
}

/** @brief Tests that an evaluation trace folds the norm and emits no BATCHNORM2D. */
TEST(Unit_BlockConvNorm, AnEvaluationTraceFoldsTheNorm)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 3}, "x");
    Conv block(3, 16, 3, 2, K_YOLO_NORM);

    const Graph::ValueId y = block.forward(Trace::evaluation(builder), x);

    EXPECT_EQ(count_nodes(graph, Ops::BATCHNORM2D.id), 0u);
    EXPECT_EQ(count_nodes(graph, Ops::BATCHNORM_FOLD.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::CHANNEL_AFFINE.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::SILU.id), 1u);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 16}));
}

/** @brief Tests that ConvLinear ends at the normalisation and emits no SILU. */
TEST(Unit_BlockConvNorm, ConvLinearEmitsNoActivation)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 3}, "x");
    ConvLinear block(3, 16, 1, 1, K_YOLO_NORM);
    BufferUpdateList list;

    const Graph::ValueId y = block.forward(Trace::training(builder, list), x);

    ASSERT_EQ(count_nodes(graph, Ops::BATCHNORM2D.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::SILU.id), 0u);
    EXPECT_EQ(graph.nodes()[first_node(graph, Ops::BATCHNORM2D.id)].outputs[0], y);
}

/**
 * @brief Tests that a traced Conv(3, 16, 3, 2) yields the five records MANIFEST.json lists
 *        under model.0, names, NHWC shapes and order.
 */
TEST(Unit_BlockConvNorm, StateMatchesManifestRow0)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 8, 8, 3}, "x");
    Conv block(3, 16, 3, 2, K_YOLO_NORM);
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    // tests/fixtures/yolo26/MANIFEST.json, weights.sacp tensors, from line 169.
    const std::vector<std::string> expected = {
        record("model.0.conv.weight", Shape{16, 3, 3, 3}, StateKind::Parameter),
        record("model.0.bn.weight", Shape{16}, StateKind::Parameter),
        record("model.0.bn.bias", Shape{16}, StateKind::Parameter),
        record("model.0.bn.running_mean", Shape{16}, StateKind::Buffer),
        record("model.0.bn.running_var", Shape{16}, StateKind::Buffer),
    };

    EXPECT_EQ(walk(block, "model.0"), expected);
}

/** @brief Tests that an untraced block yields no state, since nothing is declared yet. */
TEST(Unit_BlockConvNorm, AnUntracedBlockYieldsNoState)
{
    Conv block(3, 16, 3, 2, K_YOLO_NORM);

    EXPECT_TRUE(walk(block, "model.0").empty());
}

/**
 * @brief Tests that make_conv_node builds row 0 from its file row: the port's shape, the
 *        traced shape and the state names.
 */
TEST(Unit_BlockConvNorm, TheFactoryBuildsRow0)
{
    const Shape input{640, 640, 3};
    const Config::Value attributes = attributes_of(
        R"({"filters": 16, "kernel": 3, "stride": 2, "bn_epsilon": 0.001, "bn_momentum": 0.03})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_conv_node(one_input(input), attributes, K_CONTEXT));
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].name, NN::K_SINGLE_PORT_NAME);
    EXPECT_EQ(node->ports()[0].shape, (Shape{320, 320, 16}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{1, 640, 640, 3}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{1, 320, 320, 16}));
    ASSERT_EQ(count_nodes(graph, Ops::BATCHNORM2D.id), 1u);
    EXPECT_EQ(graph.nodes()[first_node(graph, Ops::BATCHNORM2D.id)].attributes.values[0], 1e-3);
    ASSERT_EQ(list.updates().size(), 2u);
    EXPECT_NEAR(list.updates()[0].momentum, 0.03, K_EXACT_TOLERANCE);

    const std::vector<std::string> state = walk(*node, "model.0");
    ASSERT_EQ(state.size(), 5u);
    EXPECT_EQ(state[0], record("model.0.conv.weight", Shape{16, 3, 3, 3}, StateKind::Parameter));
    EXPECT_EQ(state[4], record("model.0.bn.running_var", Shape{16}, StateKind::Buffer));
}

/** @brief Tests that a row stating neither kernel nor stride gets a 1x1 window at stride 1. */
TEST(Unit_BlockConvNorm, TheFactoryDefaultsKernelAndStrideToOne)
{
    const Shape input{6, 10, 3};
    const Config::Value attributes = attributes_of(R"({"filters": 8})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_conv_node(one_input(input), attributes, K_CONTEXT));
    EXPECT_EQ(node->ports()[0].shape, (Shape{6, 10, 8}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 6, 10, 3}, "x");
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    const std::vector<std::string> state = walk(*node, "");
    ASSERT_FALSE(state.empty());
    EXPECT_EQ(state[0], record("conv.weight", Shape{8, 1, 1, 3}, StateKind::Parameter));
}

/** @brief Tests that the factory refuses a key it does not know and names it and the row. */
TEST(Unit_BlockConvNorm, TheFactoryRefusesAnUnknownKey)
{
    const Shape input{8, 8, 3};
    const Config::Value attributes = attributes_of(R"({"filters": 8, "padding": 1})");

    SA_EXPECT_FAILURE(NN::make_conv_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 0 'conv'", "'padding'", "filters");
}

/** @brief Tests that the factory refuses a row without filters. */
TEST(Unit_BlockConvNorm, TheFactoryRefusesARowWithoutFilters)
{
    const Shape input{8, 8, 3};
    const Config::Value attributes = attributes_of(R"({"kernel": 3})");

    SA_EXPECT_FAILURE(NN::make_conv_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 0 'conv'", "'filters'", "absent");
}

/** @brief Tests that the factory refuses a stride of zero and a kernel that is not a number. */
TEST(Unit_BlockConvNorm, TheFactoryRefusesAWindowThatIsNotAPositiveWholeNumber)
{
    const Shape input{8, 8, 3};
    const Config::Value zero_stride = attributes_of(R"({"filters": 8, "stride": 0})");
    const Config::Value text_kernel = attributes_of(R"({"filters": 8, "kernel": "3"})");

    SA_EXPECT_FAILURE(NN::make_conv_node(one_input(input), zero_stride, K_CONTEXT),
                      StatusCode::InvalidDescription, "'stride'", "positive", "0");
    SA_EXPECT_FAILURE(NN::make_conv_node(one_input(input), text_kernel, K_CONTEXT),
                      StatusCode::InvalidDescription, "'kernel'", "whole number");
}

/**
 * @brief Tests that the factory refuses an epsilon of zero or of text, and a momentum above
 *        one or below zero.
 */
TEST(Unit_BlockConvNorm, TheFactoryRefusesNormOptionsOutOfRange)
{
    const Shape input{8, 8, 3};
    const Config::Value flat_epsilon = attributes_of(R"({"filters": 8, "bn_epsilon": 0.0})");
    const Config::Value text_epsilon =
        attributes_of(R"({"filters": 8, "bn_epsilon": "0.001"})");
    const Config::Value wide_momentum = attributes_of(R"({"filters": 8, "bn_momentum": 1.5})");
    const Config::Value negative_momentum =
        attributes_of(R"({"filters": 8, "bn_momentum": -0.1})");

    SA_EXPECT_FAILURE(NN::make_conv_node(one_input(input), flat_epsilon, K_CONTEXT),
                      StatusCode::InvalidDescription, "'bn_epsilon'", "positive");
    SA_EXPECT_FAILURE(NN::make_conv_node(one_input(input), wide_momentum, K_CONTEXT),
                      StatusCode::InvalidDescription, "'bn_momentum'", "[0, 1]");
    SA_EXPECT_FAILURE(NN::make_conv_node(one_input(input), text_epsilon, K_CONTEXT),
                      StatusCode::InvalidDescription, "'bn_epsilon'", "number");
    SA_EXPECT_FAILURE(NN::make_conv_node(one_input(input), negative_momentum, K_CONTEXT),
                      StatusCode::InvalidDescription, "'bn_momentum'", "[0, 1]", "-0.1");
}

/** @brief Tests that the factory refuses two inputs and an input that is not rank 3. */
TEST(Unit_BlockConvNorm, TheFactoryRefusesInputsThatAreNotOneFeatureMap)
{
    const Shape two[] = {Shape{8, 8, 3}, Shape{8, 8, 3}};
    const Shape flat{192};
    const Config::Value attributes = attributes_of(R"({"filters": 8})");

    SA_EXPECT_FAILURE(NN::make_conv_node(NN::ShapeSpan(two), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 0 'conv'", "one input", "2");
    SA_EXPECT_FAILURE(NN::make_conv_node(one_input(flat), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "[192]", "rank-3");
}
