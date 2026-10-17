/**************************************************************************/
/* test_bottleneck.cpp                                                    */
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
 * @file test_bottleneck.cpp
 * @brief Tests Bottleneck's trace, its state names against YOLO26n row 2, and its factory.
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
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/bottleneck.hpp>

using namespace SushiAI;
using Vision::Blocks::Bottleneck;
using NN::BufferUpdateList;
using NN::NormOptions;
using NN::Trace;

namespace
{
    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 4 'bottleneck'";

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

    /** @brief Expects @p options on every BATCHNORM2D node of @p graph and every update. */
    void expect_norm_options(const Graph::Graph& graph, const BufferUpdateList& list,
                             const NormOptions& options)
    {
        // Both constants are stored as given, so they compare exactly.
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_BATCHNORM2D.id)
                EXPECT_EQ(node.attributes.values[0], options.epsilon);
        }

        for (const auto& update : list.updates())
            EXPECT_EQ(update.momentum, options.momentum);
    }

} // namespace

/** @brief Tests that the block keeps H and W and yields the output width. */
TEST(Small_BlockBottleneck, OutputShapeKeepsTheExtentsAndSetsTheWidth)
{
    const Bottleneck block(16, 24, 8, true, K_YOLO_NORM);

    EXPECT_EQ(block.output_shape(Shape{5, 7, 16}), (Shape{5, 7, 24}));
}

/**
 * @brief Tests that a shortcut over equal widths traces two Conv blocks and one ADD of the
 *        input and the second convolution's result, in that operand order.
 */
TEST(Small_BlockBottleneck, AShortcutOverEqualWidthsAddsTheInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 16}, "x");
    Bottleneck block(16, 16, 8, true, K_YOLO_NORM);
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_TRUE(block.adds_input);
    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SILU.id), 2u);
    ASSERT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);

    const Graph::Node& add = graph.nodes()[first_node(graph, Ops::K_ADD.id)];
    ASSERT_EQ(add.inputs.size(), 2u);
    EXPECT_EQ(add.inputs[0], x);
    EXPECT_EQ(graph.node(graph.value(add.inputs[1]).producer).op, Ops::K_SILU.id);
    EXPECT_EQ(add.outputs[0], y);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 16}));
}

/** @brief Tests that a block built without a shortcut traces no ADD over equal widths. */
TEST(Small_BlockBottleneck, NoShortcutTracesNoAdd)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 16}, "x");
    Bottleneck block(16, 16, 8, false, K_YOLO_NORM);
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_FALSE(block.adds_input);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 0u);
    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 2u);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 16}));
}

/** @brief Tests that a shortcut over unequal widths is dropped, so no ADD is traced. */
TEST(Small_BlockBottleneck, AShortcutOverUnequalWidthsTracesNoAdd)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 16}, "x");
    Bottleneck block(16, 24, 8, true, K_YOLO_NORM);
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_FALSE(block.adds_input);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 0u);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 24}));
}

/**
 * @brief Tests that a traced Bottleneck(16, 16, 8) yields the ten records MANIFEST.json lists
 *        under model.2.m.0, names, NHWC shapes and order.
 */
TEST(Small_BlockBottleneck, StateMatchesManifestRow2Inner)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 16}, "x");
    Bottleneck block(16, 16, 8, true, K_YOLO_NORM);
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    // MANIFEST.json of the registration fixtures, weights.sushi tensors, from line 493.
    const std::vector<std::string> expected = {
        record("model.2.m.0.cv1.conv.weight", Shape{8, 3, 3, 16}, StateKind::Parameter),
        record("model.2.m.0.cv1.bn.weight", Shape{8}, StateKind::Parameter),
        record("model.2.m.0.cv1.bn.bias", Shape{8}, StateKind::Parameter),
        record("model.2.m.0.cv1.bn.running_mean", Shape{8}, StateKind::Buffer),
        record("model.2.m.0.cv1.bn.running_var", Shape{8}, StateKind::Buffer),
        record("model.2.m.0.cv2.conv.weight", Shape{16, 3, 3, 8}, StateKind::Parameter),
        record("model.2.m.0.cv2.bn.weight", Shape{16}, StateKind::Parameter),
        record("model.2.m.0.cv2.bn.bias", Shape{16}, StateKind::Parameter),
        record("model.2.m.0.cv2.bn.running_mean", Shape{16}, StateKind::Buffer),
        record("model.2.m.0.cv2.bn.running_var", Shape{16}, StateKind::Buffer),
    };

    EXPECT_EQ(walk(block, "model.2.m.0"), expected);
}

/** @brief Tests that each kernel argument sets the window of its own convolution. */
TEST(Small_BlockBottleneck, EachKernelSetsItsOwnConvolution)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 16}, "x");
    Bottleneck block(16, 16, 8, false, K_YOLO_NORM, 1, 3);
    static_cast<void>(block.forward(Trace::evaluation(builder), x));

    EXPECT_EQ(block.cv1.conv.weight.shape(), (Shape{8, 1, 1, 16}));
    EXPECT_EQ(block.cv2.conv.weight.shape(), (Shape{16, 3, 3, 8}));
}

/**
 * @brief Tests that make_bottleneck_node builds the block of model.2.m.0 and that a row
 *        without "shortcut" adds its input.
 */
TEST(Small_BlockBottleneck, TheFactoryBuildsTheBlockAndDefaultsTheShortcutToTrue)
{
    const Shape input{4, 4, 16};
    const Config::Value attributes = attributes_of(R"({"filters": 16, "hidden": 8})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_bottleneck_node(one_input(input), attributes,
                                                                 K_CONTEXT));
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{4, 4, 16}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 16}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);

    const std::vector<std::string> state = walk(*node, "model.2.m.0");
    ASSERT_EQ(state.size(), 10u);
    EXPECT_EQ(state[0],
              record("model.2.m.0.cv1.conv.weight", Shape{8, 3, 3, 16}, StateKind::Parameter));
    EXPECT_EQ(state[5],
              record("model.2.m.0.cv2.conv.weight", Shape{16, 3, 3, 8}, StateKind::Parameter));
}

/** @brief Tests that a row stating "shortcut": false builds a block that traces no ADD. */
TEST(Small_BlockBottleneck, TheFactoryReadsTheShortcut)
{
    const Shape input{4, 4, 16};
    const Config::Value attributes =
        attributes_of(R"({"filters": 16, "hidden": 8, "shortcut": false})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_bottleneck_node(one_input(input), attributes,
                                                                 K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 16}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 0u);
}

/** @brief Tests that the factory refuses an unknown key, a missing width and a text flag. */
TEST(Small_BlockBottleneck, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{4, 4, 16};
    const Config::Value unknown = attributes_of(R"({"filters": 16, "hidden": 8, "kernel": 3})");
    const Config::Value no_hidden = attributes_of(R"({"filters": 16})");
    const Config::Value text_flag =
        attributes_of(R"({"filters": 16, "hidden": 8, "shortcut": "yes"})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_bottleneck_node(one_input(input), unknown, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 4 'bottleneck'", "'kernel'");
    SA_EXPECT_FAILURE(Vision::Blocks::make_bottleneck_node(one_input(input), no_hidden, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "absent");
    SA_EXPECT_FAILURE(Vision::Blocks::make_bottleneck_node(one_input(input), text_flag, K_CONTEXT),
                      StatusCode::InvalidDescription, "'shortcut'", "true or false");
}

/** @brief Tests that the factory refuses an input that is not one rank-3 feature map. */
TEST(Small_BlockBottleneck, TheFactoryRefusesAnInputThatIsNotAFeatureMap)
{
    const Shape flat{256};
    const Config::Value attributes = attributes_of(R"({"filters": 16, "hidden": 8})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_bottleneck_node(one_input(flat), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "[256]", "rank-3");
}

/** @brief Tests that the factory hands "bn_epsilon" and "bn_momentum" to both norms. */
TEST(Small_BlockBottleneck, TheFactoryHandsTheNormOptionsToBothNorms)
{
    const Shape input{4, 4, 16};
    const Config::Value attributes = attributes_of(
        R"({"filters": 16, "hidden": 8, "bn_epsilon": 0.001, "bn_momentum": 0.03})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_bottleneck_node(one_input(input), attributes,
                                                                 K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 16}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 2u);
    ASSERT_EQ(list.updates().size(), 4u);
    expect_norm_options(graph, list, K_YOLO_NORM);
}
