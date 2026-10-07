/**************************************************************************/
/* test_block_c3k.cpp                                                     */
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
 * @file test_block_c3k.cpp
 * @brief Tests C3k's trace, its state names against YOLO26n row 6, and its factory.
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
#include <SushiAI/nn/blocks/c3k.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include "../../common/test_status.hpp"

using namespace SushiAI;
using NN::BufferUpdateList;
using NN::C3k;
using NN::NormOptions;
using NN::Trace;

namespace
{
    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 5 'c3k'";

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
            if (node.op == Ops::BATCHNORM2D.id)
                EXPECT_EQ(node.attributes.values[0], options.epsilon);
        }

        for (const auto& update : list.updates())
            EXPECT_EQ(update.momentum, options.momentum);
    }

} // namespace

/** @brief Tests that the block keeps H and W and yields the output width. */
TEST(Unit_BlockC3k, OutputShapeKeepsTheExtentsAndSetsTheWidth)
{
    const C3k block(64, 96, 2, true, K_YOLO_NORM);

    EXPECT_EQ(block.output_shape(Shape{5, 7, 64}), (Shape{5, 7, 96}));
}

/** @brief Tests that the block holds as many bottlenecks as the repeat count asks for. */
TEST(Unit_BlockC3k, TheChainHoldsOneBottleneckPerRepeat)
{
    const C3k block(64, 64, 3, true, K_YOLO_NORM);

    EXPECT_EQ(block.m.size(), 3u);
}

/**
 * @brief Tests that C3k(64, 64, 2) with a shortcut traces seven Conv blocks, two ADDs, one
 *        CONCAT_CHANNELS and no slice.
 */
TEST(Unit_BlockC3k, ATrainingTraceEmitsSevenConvBlocksTwoAddsAndOneConcat)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 64}, "x");
    C3k block(64, 64, 2, true, K_YOLO_NORM);
    BufferUpdateList list;

    const Graph::ValueId y = block.forward(Trace::training(builder, list), x);

    EXPECT_EQ(count_nodes(graph, Ops::MATMUL.id), 7u);
    EXPECT_EQ(count_nodes(graph, Ops::BATCHNORM2D.id), 7u);
    EXPECT_EQ(count_nodes(graph, Ops::SILU.id), 7u);
    EXPECT_EQ(count_nodes(graph, Ops::ADD.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::CONCAT_CHANNELS.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::SLICE_CHANNELS.id), 0u);
    EXPECT_EQ(list.updates().size(), 14u);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 64}));
}

/**
 * @brief Tests that the join reads the chain's result first and the second branch's result
 *        second, each [N, H, W, hidden].
 */
TEST(Unit_BlockC3k, TheJoinReadsTheChainThenTheSecondBranch)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 64}, "x");
    C3k block(64, 64, 2, true, K_YOLO_NORM);
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    ASSERT_EQ(count_nodes(graph, Ops::CONCAT_CHANNELS.id), 1u);
    const Graph::Node& join = graph.nodes()[first_node(graph, Ops::CONCAT_CHANNELS.id)];
    ASSERT_EQ(join.inputs.size(), 2u);

    // The chain ends in a bottleneck's ADD; the second branch ends in cv2's SILU.
    EXPECT_EQ(graph.node(graph.value(join.inputs[0]).producer).op, Ops::ADD.id);
    EXPECT_EQ(graph.node(graph.value(join.inputs[1]).producer).op, Ops::SILU.id);
    EXPECT_EQ(graph.value(join.inputs[0]).shape, (Shape{2, 4, 4, 32}));
    EXPECT_EQ(graph.value(join.inputs[1]).shape, (Shape{2, 4, 4, 32}));
    EXPECT_EQ(graph.value(join.outputs[0]).shape, (Shape{2, 4, 4, 64}));
}

/** @brief Tests that a block built without a shortcut traces no ADD. */
TEST(Unit_BlockC3k, NoShortcutTracesNoAdd)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 64}, "x");
    C3k block(64, 64, 2, false, K_YOLO_NORM);

    static_cast<void>(block.forward(Trace::evaluation(builder), x));

    EXPECT_EQ(count_nodes(graph, Ops::ADD.id), 0u);
    EXPECT_EQ(count_nodes(graph, Ops::MATMUL.id), 7u);
}

/** @brief Tests that an odd output width leaves a hidden width of half of it, rounded down. */
TEST(Unit_BlockC3k, TheHiddenWidthIsHalfTheOutputWidthRoundedDown)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 8}, "x");
    C3k block(8, 7, 1, true, K_YOLO_NORM);

    const Graph::ValueId y = block.forward(Trace::evaluation(builder), x);

    EXPECT_EQ(block.cv1.conv.weight.shape(), (Shape{3, 1, 1, 8}));
    EXPECT_EQ(block.cv2.conv.weight.shape(), (Shape{3, 1, 1, 8}));
    EXPECT_EQ(block.cv3.conv.weight.shape(), (Shape{7, 1, 1, 6}));
    EXPECT_EQ(block.m[0].cv1.conv.weight.shape(), (Shape{3, 3, 3, 3}));
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 7}));
    EXPECT_EQ(block.output_shape(Shape{4, 4, 8}), (Shape{4, 4, 7}));
}

/**
 * @brief Tests that a traced C3k(64, 64, 2) yields the 35 records MANIFEST.json lists under
 *        model.6.m.0, names, NHWC shapes and order.
 */
TEST(Unit_BlockC3k, StateMatchesManifestRow6Inner)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 64}, "x");
    C3k block(64, 64, 2, true, K_YOLO_NORM);
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    // tests/fixtures/yolo26/MANIFEST.json, weights.sacp tensors, from line 1303.
    const std::vector<std::string> expected = {
        record("model.6.m.0.cv1.conv.weight", Shape{32, 1, 1, 64}, StateKind::Parameter),
        record("model.6.m.0.cv1.bn.weight", Shape{32}, StateKind::Parameter),
        record("model.6.m.0.cv1.bn.bias", Shape{32}, StateKind::Parameter),
        record("model.6.m.0.cv1.bn.running_mean", Shape{32}, StateKind::Buffer),
        record("model.6.m.0.cv1.bn.running_var", Shape{32}, StateKind::Buffer),
        record("model.6.m.0.cv2.conv.weight", Shape{32, 1, 1, 64}, StateKind::Parameter),
        record("model.6.m.0.cv2.bn.weight", Shape{32}, StateKind::Parameter),
        record("model.6.m.0.cv2.bn.bias", Shape{32}, StateKind::Parameter),
        record("model.6.m.0.cv2.bn.running_mean", Shape{32}, StateKind::Buffer),
        record("model.6.m.0.cv2.bn.running_var", Shape{32}, StateKind::Buffer),
        record("model.6.m.0.cv3.conv.weight", Shape{64, 1, 1, 64}, StateKind::Parameter),
        record("model.6.m.0.cv3.bn.weight", Shape{64}, StateKind::Parameter),
        record("model.6.m.0.cv3.bn.bias", Shape{64}, StateKind::Parameter),
        record("model.6.m.0.cv3.bn.running_mean", Shape{64}, StateKind::Buffer),
        record("model.6.m.0.cv3.bn.running_var", Shape{64}, StateKind::Buffer),
        record("model.6.m.0.m.0.cv1.conv.weight", Shape{32, 3, 3, 32}, StateKind::Parameter),
        record("model.6.m.0.m.0.cv1.bn.weight", Shape{32}, StateKind::Parameter),
        record("model.6.m.0.m.0.cv1.bn.bias", Shape{32}, StateKind::Parameter),
        record("model.6.m.0.m.0.cv1.bn.running_mean", Shape{32}, StateKind::Buffer),
        record("model.6.m.0.m.0.cv1.bn.running_var", Shape{32}, StateKind::Buffer),
        record("model.6.m.0.m.0.cv2.conv.weight", Shape{32, 3, 3, 32}, StateKind::Parameter),
        record("model.6.m.0.m.0.cv2.bn.weight", Shape{32}, StateKind::Parameter),
        record("model.6.m.0.m.0.cv2.bn.bias", Shape{32}, StateKind::Parameter),
        record("model.6.m.0.m.0.cv2.bn.running_mean", Shape{32}, StateKind::Buffer),
        record("model.6.m.0.m.0.cv2.bn.running_var", Shape{32}, StateKind::Buffer),
        record("model.6.m.0.m.1.cv1.conv.weight", Shape{32, 3, 3, 32}, StateKind::Parameter),
        record("model.6.m.0.m.1.cv1.bn.weight", Shape{32}, StateKind::Parameter),
        record("model.6.m.0.m.1.cv1.bn.bias", Shape{32}, StateKind::Parameter),
        record("model.6.m.0.m.1.cv1.bn.running_mean", Shape{32}, StateKind::Buffer),
        record("model.6.m.0.m.1.cv1.bn.running_var", Shape{32}, StateKind::Buffer),
        record("model.6.m.0.m.1.cv2.conv.weight", Shape{32, 3, 3, 32}, StateKind::Parameter),
        record("model.6.m.0.m.1.cv2.bn.weight", Shape{32}, StateKind::Parameter),
        record("model.6.m.0.m.1.cv2.bn.bias", Shape{32}, StateKind::Parameter),
        record("model.6.m.0.m.1.cv2.bn.running_mean", Shape{32}, StateKind::Buffer),
        record("model.6.m.0.m.1.cv2.bn.running_var", Shape{32}, StateKind::Buffer),
    };

    EXPECT_EQ(walk(block, "model.6.m.0"), expected);
}

/**
 * @brief Tests that make_c3k_node builds a block and that a row stating neither "repeats"
 *        nor "shortcut" gets one bottleneck that adds its input.
 */
TEST(Unit_BlockC3k, TheFactoryDefaultsToOneBottleneckWithAShortcut)
{
    const Shape input{4, 4, 64};
    const Config::Value attributes = attributes_of(R"({"filters": 32})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_c3k_node(one_input(input), attributes, K_CONTEXT));
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{4, 4, 32}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 64}, "x");
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{2, 4, 4, 32}));
    EXPECT_EQ(count_nodes(graph, Ops::MATMUL.id), 5u);
    EXPECT_EQ(count_nodes(graph, Ops::ADD.id), 1u);
}

/** @brief Tests that the factory reads "repeats" and "shortcut" and names state as row 6 does. */
TEST(Unit_BlockC3k, TheFactoryReadsRepeatsAndShortcut)
{
    const Shape input{4, 4, 64};
    const Config::Value attributes =
        attributes_of(R"({"filters": 64, "repeats": 2, "shortcut": false})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_c3k_node(one_input(input), attributes, K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 64}, "x");
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    EXPECT_EQ(count_nodes(graph, Ops::MATMUL.id), 7u);
    EXPECT_EQ(count_nodes(graph, Ops::ADD.id), 0u);

    const std::vector<std::string> state = walk(*node, "model.6.m.0");
    ASSERT_EQ(state.size(), 35u);
    EXPECT_EQ(state[30], record("model.6.m.0.m.1.cv2.conv.weight", Shape{32, 3, 3, 32},
                                StateKind::Parameter));
}

/** @brief Tests that the factory refuses a width of one, which leaves no hidden channel. */
TEST(Unit_BlockC3k, TheFactoryRefusesAWidthThatLeavesNoHiddenChannel)
{
    const Shape input{4, 4, 64};
    const Config::Value attributes = attributes_of(R"({"filters": 1})");

    SA_EXPECT_FAILURE(NN::make_c3k_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 5 'c3k'", "'filters'",
                      "at least 2");
}

/** @brief Tests that the factory refuses an unknown key, a missing width and zero repeats. */
TEST(Unit_BlockC3k, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{4, 4, 64};
    const Config::Value unknown = attributes_of(R"({"filters": 64, "hidden": 32})");
    const Config::Value no_filters = attributes_of(R"({"repeats": 2})");
    const Config::Value zero_repeats = attributes_of(R"({"filters": 64, "repeats": 0})");

    SA_EXPECT_FAILURE(NN::make_c3k_node(one_input(input), unknown, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 5 'c3k'", "'hidden'");
    SA_EXPECT_FAILURE(NN::make_c3k_node(one_input(input), no_filters, K_CONTEXT),
                      StatusCode::InvalidDescription, "'filters'", "absent");
    SA_EXPECT_FAILURE(NN::make_c3k_node(one_input(input), zero_repeats, K_CONTEXT),
                      StatusCode::InvalidDescription, "'repeats'", "positive");
}

/** @brief Tests that the factory refuses two inputs. */
TEST(Unit_BlockC3k, TheFactoryRefusesTwoInputs)
{
    const Shape two[] = {Shape{4, 4, 64}, Shape{4, 4, 64}};
    const Config::Value attributes = attributes_of(R"({"filters": 64})");

    SA_EXPECT_FAILURE(NN::make_c3k_node(NN::ShapeSpan(two), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "one input", "2");
}

/**
 * @brief Tests that the factory hands "bn_epsilon" and "bn_momentum" to all seven norms, the
 *        bottlenecks' included.
 */
TEST(Unit_BlockC3k, TheFactoryHandsTheNormOptionsToEveryNorm)
{
    const Shape input{4, 4, 64};
    const Config::Value attributes = attributes_of(
        R"({"filters": 64, "repeats": 2, "bn_epsilon": 0.001, "bn_momentum": 0.03})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_c3k_node(one_input(input), attributes, K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 64}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(count_nodes(graph, Ops::BATCHNORM2D.id), 7u);
    ASSERT_EQ(list.updates().size(), 14u);
    expect_norm_options(graph, list, K_YOLO_NORM);
}
