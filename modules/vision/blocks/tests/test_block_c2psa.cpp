/**************************************************************************/
/* test_block_c2psa.cpp                                                   */
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
 * @file test_block_c2psa.cpp
 * @brief Tests C2PSA's trace, its state names against YOLO26n row 10, and its factory.
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
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/c2psa.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using NN::BufferUpdateList;
using NN::NormOptions;
using NN::Trace;
using Vision::Blocks::C2PSA;

namespace
{
    /** @brief Holds the absolute tolerance of a double stored and read back unchanged. */
    constexpr double K_EXACT_TOLERANCE = 1e-15;

    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 10 'c2psa'";

    /** @brief Holds the width of each half of row 10's stem. */
    constexpr int64_t K_HIDDEN = 128;

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

    /**
     * @brief Returns the index of the @p ordinal-th node of @p graph that carries @p op, counting
     *        from zero, or node_count() when there are not that many.
     */
    std::size_t node_index(const Graph::Graph& graph, OpID op, std::size_t ordinal)
    {
        std::size_t seen = 0;
        for (std::size_t index = 0; index < graph.node_count(); ++index)
        {
            if (graph.nodes()[index].op != op)
                continue;

            if (seen == ordinal)
                return index;

            ++seen;
        }

        return graph.node_count();
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

    /** @brief Lists the 35 records MANIFEST.json holds under model.10, names, shapes and order. */
    std::vector<std::string> row10_records()
    {
        return {
            record("model.10.cv1.conv.weight", Shape{256, 1, 1, 256}, StateKind::Parameter),
            record("model.10.cv1.bn.weight", Shape{256}, StateKind::Parameter),
            record("model.10.cv1.bn.bias", Shape{256}, StateKind::Parameter),
            record("model.10.cv1.bn.running_mean", Shape{256}, StateKind::Buffer),
            record("model.10.cv1.bn.running_var", Shape{256}, StateKind::Buffer),
            record("model.10.cv2.conv.weight", Shape{256, 1, 1, 256}, StateKind::Parameter),
            record("model.10.cv2.bn.weight", Shape{256}, StateKind::Parameter),
            record("model.10.cv2.bn.bias", Shape{256}, StateKind::Parameter),
            record("model.10.cv2.bn.running_mean", Shape{256}, StateKind::Buffer),
            record("model.10.cv2.bn.running_var", Shape{256}, StateKind::Buffer),
            record("model.10.m.0.attn.qkv.conv.weight", Shape{256, 1, 1, 128},
                   StateKind::Parameter),
            record("model.10.m.0.attn.qkv.bn.weight", Shape{256}, StateKind::Parameter),
            record("model.10.m.0.attn.qkv.bn.bias", Shape{256}, StateKind::Parameter),
            record("model.10.m.0.attn.qkv.bn.running_mean", Shape{256}, StateKind::Buffer),
            record("model.10.m.0.attn.qkv.bn.running_var", Shape{256}, StateKind::Buffer),
            record("model.10.m.0.attn.proj.conv.weight", Shape{128, 1, 1, 128},
                   StateKind::Parameter),
            record("model.10.m.0.attn.proj.bn.weight", Shape{128}, StateKind::Parameter),
            record("model.10.m.0.attn.proj.bn.bias", Shape{128}, StateKind::Parameter),
            record("model.10.m.0.attn.proj.bn.running_mean", Shape{128}, StateKind::Buffer),
            record("model.10.m.0.attn.proj.bn.running_var", Shape{128}, StateKind::Buffer),
            record("model.10.m.0.attn.pe.conv.weight", Shape{3, 3, 128}, StateKind::Parameter),
            record("model.10.m.0.attn.pe.bn.weight", Shape{128}, StateKind::Parameter),
            record("model.10.m.0.attn.pe.bn.bias", Shape{128}, StateKind::Parameter),
            record("model.10.m.0.attn.pe.bn.running_mean", Shape{128}, StateKind::Buffer),
            record("model.10.m.0.attn.pe.bn.running_var", Shape{128}, StateKind::Buffer),
            record("model.10.m.0.ffn.0.conv.weight", Shape{256, 1, 1, 128}, StateKind::Parameter),
            record("model.10.m.0.ffn.0.bn.weight", Shape{256}, StateKind::Parameter),
            record("model.10.m.0.ffn.0.bn.bias", Shape{256}, StateKind::Parameter),
            record("model.10.m.0.ffn.0.bn.running_mean", Shape{256}, StateKind::Buffer),
            record("model.10.m.0.ffn.0.bn.running_var", Shape{256}, StateKind::Buffer),
            record("model.10.m.0.ffn.1.conv.weight", Shape{128, 1, 1, 256}, StateKind::Parameter),
            record("model.10.m.0.ffn.1.bn.weight", Shape{128}, StateKind::Parameter),
            record("model.10.m.0.ffn.1.bn.bias", Shape{128}, StateKind::Parameter),
            record("model.10.m.0.ffn.1.bn.running_mean", Shape{128}, StateKind::Buffer),
            record("model.10.m.0.ffn.1.bn.running_var", Shape{128}, StateKind::Buffer),
        };
    }

    /** @brief Builds row 10 of YOLO26n: C2PSA(256, 256, 128) with one PSABlock of two heads. */
    C2PSA make_row10()
    {
        return C2PSA(256, 256, K_HIDDEN, 1, 2, K_YOLO_NORM);
    }

} // namespace

/** @brief Tests that row 10 keeps H and W and yields the output width. */
TEST(Unit_BlockC2psa, OutputShapeKeepsTheExtentsAndSetsTheWidth)
{
    const C2PSA block = make_row10();

    EXPECT_EQ(block.output_shape(Shape{20, 20, 256}), (Shape{20, 20, 256}));
}

/**
 * @brief Tests that row 10 traces eight slices, three ADDs, three joins, three SILUs and seven
 *        norms: the two halves of the stem and the attention's six slices, the attention's own
 *        add and the block's two, the two joins of the attention and the C2PSA join, cv1, cv2
 *        and the feed-forward's first convolution.
 */
TEST(Unit_BlockC2psa, Row10EmitsEightSlicesThreeAddsAndThreeJoins)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    C2PSA block = make_row10();
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    // Two halves of the stem, and six slices of the attention's qkv.
    EXPECT_EQ(count_nodes(graph, Ops::K_SLICE_CHANNELS.id), 8u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 3u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 3u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SILU.id), 3u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 7u);
    EXPECT_EQ(list.updates().size(), 14u);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 256}));
    expect_norm_options(graph, list, K_YOLO_NORM);
}

/**
 * @brief Tests that the first half is kept and the second half feeds the block, and that the
 *        join reads the kept half, then the block's result.
 */
TEST(Unit_BlockC2psa, TheJoinKeepsTheFirstHalfAndChainsTheSecondOffTheSecondSlice)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    C2PSA block = make_row10();
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    // The stem's two slices come before the attention's, so they are the first two slices.
    const std::size_t kept_index = node_index(graph, Ops::K_SLICE_CHANNELS.id, 0);
    const std::size_t chained_index = node_index(graph, Ops::K_SLICE_CHANNELS.id, 1);
    ASSERT_LT(chained_index, graph.node_count());
    const Graph::Node& kept = graph.nodes()[kept_index];
    const Graph::Node& chained = graph.nodes()[chained_index];

    // A slice stores its offset as a whole number in slot 0, so it compares exactly.
    EXPECT_NEAR(kept.attributes.values[0], 0.0, K_EXACT_TOLERANCE);
    EXPECT_NEAR(chained.attributes.values[0], static_cast<double>(K_HIDDEN), K_EXACT_TOLERANCE);
    EXPECT_EQ(kept.inputs[0], chained.inputs[0]);

    // The block's first residual add is the second ADD, after the attention's own.
    const Graph::Node& first_residual = graph.nodes()[node_index(graph, Ops::K_ADD.id, 1)];
    EXPECT_EQ(first_residual.inputs[0], chained.outputs[0]);

    // The join is the third concatenation: the attention's two come before it.
    const Graph::Node& join = graph.nodes()[node_index(graph, Ops::K_CONCAT_CHANNELS.id, 2)];
    const Graph::Node& last_residual = graph.nodes()[node_index(graph, Ops::K_ADD.id, 2)];
    ASSERT_EQ(join.inputs.size(), 2u);
    EXPECT_EQ(join.inputs[0], kept.outputs[0]);
    EXPECT_EQ(join.inputs[1], last_residual.outputs[0]);
    EXPECT_EQ(graph.value(join.outputs[0]).shape, (Shape{2, 4, 4, 256}));
}

/**
 * @brief Tests that with two repeats each block reads the output of the one before it.
 */
TEST(Unit_BlockC2psa, EachRepeatReadsTheResultOfThePreviousOne)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    C2PSA block(256, 256, K_HIDDEN, 2, 2, K_YOLO_NORM);

    static_cast<void>(block.forward(Trace::evaluation(builder), x));

    // A block emits three ADDs, so block one ends at ADD 2 and block two's residual is ADD 4.
    EXPECT_EQ(block.m.size(), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 6u);
    const Graph::Node& first_block_last = graph.nodes()[node_index(graph, Ops::K_ADD.id, 2)];
    const Graph::Node& second_block_first = graph.nodes()[node_index(graph, Ops::K_ADD.id, 4)];
    EXPECT_EQ(second_block_first.inputs[0], first_block_last.outputs[0]);
}

/**
 * @brief Tests that the state of traced row 10 yields the 35 records MANIFEST.json lists under
 *        model.10, names, NHWC shapes and order.
 */
TEST(Unit_BlockC2psa, StateMatchesManifestRow10)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    C2PSA block = make_row10();
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    EXPECT_EQ(walk(block, "model.10"), row10_records());
}

/**
 * @brief Tests that make_c2psa_node builds row 10 from its file row: the port's shape, the 35
 *        manifest records and the row's norm options on all seven norms.
 */
TEST(Unit_BlockC2psa, TheFactoryBuildsRow10)
{
    const Shape input{20, 20, 256};
    const Config::Value attributes = attributes_of(
        R"({"filters": 256, "hidden": 128, "repeats": 1, "heads": 2,
            "bn_epsilon": 0.001, "bn_momentum": 0.03})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c2psa_node(one_input(input), attributes,
                                                            K_CONTEXT));
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{20, 20, 256}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 7u);
    EXPECT_EQ(list.updates().size(), 14u);
    expect_norm_options(graph, list, K_YOLO_NORM);
    EXPECT_EQ(walk(*node, "model.10"), row10_records());
}

/**
 * @brief Tests that a row stating neither "repeats" nor "heads" gets one PSABlock of
 *        max(hidden / 64, 1) heads, which is two for a hidden width of 128.
 */
TEST(Unit_BlockC2psa, TheFactoryDefaultsToOneBlockAndHiddenOver64Heads)
{
    const Shape input{20, 20, 256};
    const Config::Value attributes = attributes_of(R"({"filters": 256, "hidden": 128})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c2psa_node(one_input(input), attributes,
                                                            K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    EXPECT_EQ(count_nodes(graph, Ops::K_LOG_SOFTMAX.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 3u);
    EXPECT_EQ(walk(*node, "").size(), 35u);
}

/**
 * @brief Tests that a hidden width under 128 gets a single head, as max(hidden / 64, 1) gives.
 */
TEST(Unit_BlockC2psa, TheFactoryGivesOneHeadForAHiddenWidthUnder128)
{
    const Shape input{4, 4, 64};
    const Config::Value attributes = attributes_of(R"({"filters": 64, "hidden": 64})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c2psa_node(one_input(input), attributes,
                                                            K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 64}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    EXPECT_EQ(count_nodes(graph, Ops::K_LOG_SOFTMAX.id), 1u);
}

/** @brief Tests that a "filters" other than the input width is refused, naming both widths. */
TEST(Unit_BlockC2psa, TheFactoryRefusesAFilterWidthThatIsNotTheInput)
{
    const Shape input{20, 20, 256};
    const Config::Value attributes = attributes_of(R"({"filters": 128, "hidden": 128})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c2psa_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "'filters'", "128", "256 channels");
}

/**
 * @brief Tests that a hidden width the heads do not split into parts of two or more channels is
 *        refused.
 */
TEST(Unit_BlockC2psa, TheFactoryRefusesAHiddenWidthTheHeadsDoNotSplit)
{
    const Shape input{20, 20, 256};
    const Config::Value indivisible =
        attributes_of(R"({"filters": 256, "hidden": 128, "heads": 3})");
    const Config::Value one_channel =
        attributes_of(R"({"filters": 256, "hidden": 128, "heads": 128})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c2psa_node(one_input(input), indivisible, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "parts of at least 2 channels");
    SA_EXPECT_FAILURE(Vision::Blocks::make_c2psa_node(one_input(input), one_channel, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "parts of at least 2 channels");
}

/** @brief Tests that the factory refuses an unknown key, a missing width and a flat input. */
TEST(Unit_BlockC2psa, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{20, 20, 256};
    const Shape flat{512};
    const Config::Value unknown =
        attributes_of(R"({"filters": 256, "hidden": 128, "inner": "bottleneck"})");
    const Config::Value no_hidden = attributes_of(R"({"filters": 256})");
    const Config::Value fits = attributes_of(R"({"filters": 256, "hidden": 128})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c2psa_node(one_input(input), unknown, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 10 'c2psa'", "'inner'");
    SA_EXPECT_FAILURE(Vision::Blocks::make_c2psa_node(one_input(input), no_hidden, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "absent");
    SA_EXPECT_FAILURE(Vision::Blocks::make_c2psa_node(one_input(flat), fits, K_CONTEXT),
                      StatusCode::InvalidDescription, "[512]", "rank-3");
}
