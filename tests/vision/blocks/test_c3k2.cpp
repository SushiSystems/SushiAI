/**************************************************************************/
/* test_c3k2.cpp                                                          */
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
 * @file test_c3k2.cpp
 * @brief Tests C3k2's trace, its state names against YOLO26n rows 2, 6 and 22, and its factories.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
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
#include <SushiAI/nn/module_list.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/bottleneck.hpp>
#include <SushiAI/vision/blocks/c3k.hpp>
#include <SushiAI/vision/blocks/c3k2.hpp>

using namespace SushiAI;
using Vision::Blocks::Bottleneck;
using NN::BufferUpdateList;
using Vision::Blocks::C3k;
using Vision::Blocks::C3k2;
using NN::NormOptions;
using NN::Trace;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Holds the absolute tolerance of a double stored and read back unchanged. */
    constexpr double K_EXACT_TOLERANCE = 1e-15;

    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 2 'c3k2'";

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
        return require_value(Config::parse_json(text, "<test>"));
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

namespace
{
    /** @brief Lists the 20 records MANIFEST.json holds under model.2, from line 331. */
    std::vector<std::string> row2_records()
    {
        return {
            record("model.2.cv1.conv.weight", Shape{32, 1, 1, 32}, StateKind::Parameter),
            record("model.2.cv1.bn.weight", Shape{32}, StateKind::Parameter),
            record("model.2.cv1.bn.bias", Shape{32}, StateKind::Parameter),
            record("model.2.cv1.bn.running_mean", Shape{32}, StateKind::Buffer),
            record("model.2.cv1.bn.running_var", Shape{32}, StateKind::Buffer),
            record("model.2.cv2.conv.weight", Shape{64, 1, 1, 48}, StateKind::Parameter),
            record("model.2.cv2.bn.weight", Shape{64}, StateKind::Parameter),
            record("model.2.cv2.bn.bias", Shape{64}, StateKind::Parameter),
            record("model.2.cv2.bn.running_mean", Shape{64}, StateKind::Buffer),
            record("model.2.cv2.bn.running_var", Shape{64}, StateKind::Buffer),
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
    }

    /** @brief Lists the 45 records MANIFEST.json holds under model.6, from line 1141. */
    std::vector<std::string> row6_records()
    {
        return {
            record("model.6.cv1.conv.weight", Shape{128, 1, 1, 128}, StateKind::Parameter),
            record("model.6.cv1.bn.weight", Shape{128}, StateKind::Parameter),
            record("model.6.cv1.bn.bias", Shape{128}, StateKind::Parameter),
            record("model.6.cv1.bn.running_mean", Shape{128}, StateKind::Buffer),
            record("model.6.cv1.bn.running_var", Shape{128}, StateKind::Buffer),
            record("model.6.cv2.conv.weight", Shape{128, 1, 1, 192}, StateKind::Parameter),
            record("model.6.cv2.bn.weight", Shape{128}, StateKind::Parameter),
            record("model.6.cv2.bn.bias", Shape{128}, StateKind::Parameter),
            record("model.6.cv2.bn.running_mean", Shape{128}, StateKind::Buffer),
            record("model.6.cv2.bn.running_var", Shape{128}, StateKind::Buffer),
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
    }

    /** @brief Lists the 45 records MANIFEST.json holds under model.22, from line 5757. */
    std::vector<std::string> row22_records()
    {
        return {
            record("model.22.cv1.conv.weight", Shape{256, 1, 1, 384}, StateKind::Parameter),
            record("model.22.cv1.bn.weight", Shape{256}, StateKind::Parameter),
            record("model.22.cv1.bn.bias", Shape{256}, StateKind::Parameter),
            record("model.22.cv1.bn.running_mean", Shape{256}, StateKind::Buffer),
            record("model.22.cv1.bn.running_var", Shape{256}, StateKind::Buffer),
            record("model.22.cv2.conv.weight", Shape{256, 1, 1, 384}, StateKind::Parameter),
            record("model.22.cv2.bn.weight", Shape{256}, StateKind::Parameter),
            record("model.22.cv2.bn.bias", Shape{256}, StateKind::Parameter),
            record("model.22.cv2.bn.running_mean", Shape{256}, StateKind::Buffer),
            record("model.22.cv2.bn.running_var", Shape{256}, StateKind::Buffer),
            record("model.22.m.0.0.cv1.conv.weight", Shape{64, 3, 3, 128}, StateKind::Parameter),
            record("model.22.m.0.0.cv1.bn.weight", Shape{64}, StateKind::Parameter),
            record("model.22.m.0.0.cv1.bn.bias", Shape{64}, StateKind::Parameter),
            record("model.22.m.0.0.cv1.bn.running_mean", Shape{64}, StateKind::Buffer),
            record("model.22.m.0.0.cv1.bn.running_var", Shape{64}, StateKind::Buffer),
            record("model.22.m.0.0.cv2.conv.weight", Shape{128, 3, 3, 64}, StateKind::Parameter),
            record("model.22.m.0.0.cv2.bn.weight", Shape{128}, StateKind::Parameter),
            record("model.22.m.0.0.cv2.bn.bias", Shape{128}, StateKind::Parameter),
            record("model.22.m.0.0.cv2.bn.running_mean", Shape{128}, StateKind::Buffer),
            record("model.22.m.0.0.cv2.bn.running_var", Shape{128}, StateKind::Buffer),
            record("model.22.m.0.1.attn.qkv.conv.weight", Shape{256, 1, 1, 128},
                   StateKind::Parameter),
            record("model.22.m.0.1.attn.qkv.bn.weight", Shape{256}, StateKind::Parameter),
            record("model.22.m.0.1.attn.qkv.bn.bias", Shape{256}, StateKind::Parameter),
            record("model.22.m.0.1.attn.qkv.bn.running_mean", Shape{256}, StateKind::Buffer),
            record("model.22.m.0.1.attn.qkv.bn.running_var", Shape{256}, StateKind::Buffer),
            record("model.22.m.0.1.attn.proj.conv.weight", Shape{128, 1, 1, 128},
                   StateKind::Parameter),
            record("model.22.m.0.1.attn.proj.bn.weight", Shape{128}, StateKind::Parameter),
            record("model.22.m.0.1.attn.proj.bn.bias", Shape{128}, StateKind::Parameter),
            record("model.22.m.0.1.attn.proj.bn.running_mean", Shape{128}, StateKind::Buffer),
            record("model.22.m.0.1.attn.proj.bn.running_var", Shape{128}, StateKind::Buffer),
            record("model.22.m.0.1.attn.pe.conv.weight", Shape{3, 3, 128}, StateKind::Parameter),
            record("model.22.m.0.1.attn.pe.bn.weight", Shape{128}, StateKind::Parameter),
            record("model.22.m.0.1.attn.pe.bn.bias", Shape{128}, StateKind::Parameter),
            record("model.22.m.0.1.attn.pe.bn.running_mean", Shape{128}, StateKind::Buffer),
            record("model.22.m.0.1.attn.pe.bn.running_var", Shape{128}, StateKind::Buffer),
            record("model.22.m.0.1.ffn.0.conv.weight", Shape{256, 1, 1, 128}, StateKind::Parameter),
            record("model.22.m.0.1.ffn.0.bn.weight", Shape{256}, StateKind::Parameter),
            record("model.22.m.0.1.ffn.0.bn.bias", Shape{256}, StateKind::Parameter),
            record("model.22.m.0.1.ffn.0.bn.running_mean", Shape{256}, StateKind::Buffer),
            record("model.22.m.0.1.ffn.0.bn.running_var", Shape{256}, StateKind::Buffer),
            record("model.22.m.0.1.ffn.1.conv.weight", Shape{128, 1, 1, 256}, StateKind::Parameter),
            record("model.22.m.0.1.ffn.1.bn.weight", Shape{128}, StateKind::Parameter),
            record("model.22.m.0.1.ffn.1.bn.bias", Shape{128}, StateKind::Parameter),
            record("model.22.m.0.1.ffn.1.bn.running_mean", Shape{128}, StateKind::Buffer),
            record("model.22.m.0.1.ffn.1.bn.running_var", Shape{128}, StateKind::Buffer),
        };
    }

    /** @brief Builds row 2 of YOLO26n: C3k2(32, 64, 16) over one Bottleneck(16, 16, 8). */
    C3k2<Bottleneck> make_row2()
    {
        NN::ModuleList<Bottleneck> inner;
        inner.push_back(Bottleneck(16, 16, 8, true, K_YOLO_NORM));

        return C3k2<Bottleneck>(32, 64, 16, std::move(inner), K_YOLO_NORM);
    }

    /** @brief Builds row 6 of YOLO26n: C3k2(128, 128, 64) over one C3k(64, 64, 2). */
    C3k2<C3k> make_row6()
    {
        NN::ModuleList<C3k> inner;
        inner.push_back(C3k(64, 64, 2, true, K_YOLO_NORM));

        return C3k2<C3k>(128, 128, 64, std::move(inner), K_YOLO_NORM);
    }

} // namespace

/** @brief Tests that row 2 keeps H and W and yields the output width. */
TEST(Small_BlockC3k2, OutputShapeKeepsTheExtentsAndSetsTheWidth)
{
    const C3k2<Bottleneck> block = make_row2();

    EXPECT_EQ(block.output_shape(Shape{160, 160, 32}).value(), (Shape{160, 160, 64}));
}

/**
 * @brief Tests that row 2 traces four Conv blocks, two slices, one ADD and one
 *        CONCAT_CHANNELS.
 */
TEST(Small_BlockC3k2, Row2EmitsFourConvBlocksTwoSlicesAndOneConcat)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 32}, "x");
    C3k2<Bottleneck> block = make_row2();
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 4u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 4u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SILU.id), 4u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SLICE_CHANNELS.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
    EXPECT_EQ(list.updates().size(), 8u);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 64}));
    expect_norm_options(graph, list, K_YOLO_NORM);
}

/**
 * @brief Tests that the join reads the stem's first half, its second half, then the inner
 *        block's result, and that the inner block reads the second half.
 */
TEST(Small_BlockC3k2, TheJoinReadsBothHalvesThenTheChainOffTheSecondHalf)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 32}, "x");
    C3k2<Bottleneck> block = make_row2();
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    ASSERT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
    const Graph::Node& join = graph.nodes()[first_node(graph, Ops::K_CONCAT_CHANNELS.id)];
    ASSERT_EQ(join.inputs.size(), 3u);

    const Graph::Node& first_half = graph.node(graph.value(join.inputs[0]).producer);
    const Graph::Node& second_half = graph.node(graph.value(join.inputs[1]).producer);
    const Graph::Node& chained = graph.node(graph.value(join.inputs[2]).producer);
    ASSERT_EQ(first_half.op, Ops::K_SLICE_CHANNELS.id);
    ASSERT_EQ(second_half.op, Ops::K_SLICE_CHANNELS.id);
    ASSERT_EQ(chained.op, Ops::K_ADD.id);

    // A slice stores its offset as a whole number in slot 0, so it compares exactly.
    EXPECT_NEAR(first_half.attributes.values[0], 0.0, K_EXACT_TOLERANCE);
    EXPECT_NEAR(second_half.attributes.values[0], 16.0, K_EXACT_TOLERANCE);
    EXPECT_EQ(first_half.inputs[0], second_half.inputs[0]);
    EXPECT_EQ(graph.value(join.inputs[0]).shape, (Shape{2, 4, 4, 16}));
    EXPECT_EQ(graph.value(join.inputs[1]).shape, (Shape{2, 4, 4, 16}));

    // The bottleneck adds its own input, which is the second half.
    EXPECT_EQ(chained.inputs[0], join.inputs[1]);
    EXPECT_EQ(graph.value(join.outputs[0]).shape, (Shape{2, 4, 4, 48}));
}

/** @brief Tests that each inner block reads the one before it and that all are joined. */
TEST(Small_BlockC3k2, TwoInnerBlocksAreChainedAndBothJoined)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 32}, "x");
    NN::ModuleList<Bottleneck> inner;
    inner.push_back(Bottleneck(16, 16, 8, true, K_YOLO_NORM));
    inner.push_back(Bottleneck(16, 16, 8, true, K_YOLO_NORM));
    C3k2<Bottleneck> block(32, 64, 16, std::move(inner), K_YOLO_NORM);

    static_cast<void>(block.forward(Trace::evaluation(builder), x));

    ASSERT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
    const Graph::Node& join = graph.nodes()[first_node(graph, Ops::K_CONCAT_CHANNELS.id)];
    ASSERT_EQ(join.inputs.size(), 4u);
    EXPECT_EQ(graph.node(graph.value(join.inputs[2]).producer).inputs[0], join.inputs[1]);
    EXPECT_EQ(graph.node(graph.value(join.inputs[3]).producer).inputs[0], join.inputs[2]);
    EXPECT_EQ(block.cv2.conv.weight.shape(), (Shape{64, 1, 1, 64}));
    EXPECT_EQ(block.m[1].cv1.conv.weight.shape(), (Shape{8, 3, 3, 16}));
}

/**
 * @brief Tests that row 6, whose inner block is a C3k, traces nine Conv blocks, two slices,
 *        two ADDs and two CONCAT_CHANNELS.
 */
TEST(Small_BlockC3k2, Row6EmitsNineConvBlocksAndTwoConcats)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 128}, "x");
    C3k2<C3k> block = make_row6();
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 9u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 9u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SILU.id), 9u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SLICE_CHANNELS.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 2u);
    EXPECT_EQ(list.updates().size(), 18u);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 128}));
    expect_norm_options(graph, list, K_YOLO_NORM);
    EXPECT_EQ(block.output_shape(Shape{40, 40, 128}).value(), (Shape{40, 40, 128}));
}

/**
 * @brief Tests that traced row 2 yields the 20 records MANIFEST.json lists under model.2,
 *        names, NHWC shapes and order.
 */
TEST(Small_BlockC3k2, StateMatchesManifestRow2)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 32}, "x");
    C3k2<Bottleneck> block = make_row2();
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    EXPECT_EQ(walk(block, "model.2"), row2_records());
}

/**
 * @brief Tests that traced row 6 yields the 45 records MANIFEST.json lists under model.6,
 *        names, NHWC shapes and order.
 */
TEST(Small_BlockC3k2, StateMatchesManifestRow6)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 128}, "x");
    C3k2<C3k> block = make_row6();
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    EXPECT_EQ(walk(block, "model.6"), row6_records());
}

/**
 * @brief Tests that K_C3K2_INNER lists the bottleneck, the C3k and the bottleneck_psa pair,
 *        each with a factory.
 */
TEST(Small_BlockC3k2, TheInnerTableListsEachInnerBlockWithItsFactory)
{
    ASSERT_EQ(Vision::Blocks::K_C3K2_INNER.size(), 3u);
    EXPECT_EQ(Vision::Blocks::K_C3K2_INNER[0].name, "bottleneck");
    EXPECT_EQ(Vision::Blocks::K_C3K2_INNER[0].make, &Vision::Blocks::make_c3k2_of_bottleneck);
    EXPECT_EQ(Vision::Blocks::K_C3K2_INNER[1].name, "c3k");
    EXPECT_EQ(Vision::Blocks::K_C3K2_INNER[1].make, &Vision::Blocks::make_c3k2_of_c3k);
    EXPECT_EQ(Vision::Blocks::K_C3K2_INNER[2].name, "bottleneck_psa");
    EXPECT_EQ(Vision::Blocks::K_C3K2_INNER[2].make, &Vision::Blocks::make_c3k2_of_bottleneck_psa);
}

/**
 * @brief Tests that make_c3k2_node builds row 22 from its file row, whose inner block is a
 *        Bottleneck and a PSABlock: the port's shape, the 45 manifest records, the 18 buffer
 *        updates and the row's norm options on all nine norms.
 */
TEST(Small_BlockC3k2, TheFactoryBuildsRow22)
{
    const Shape input{20, 20, 384};
    const Config::Value attributes = attributes_of(
        R"({"filters": 256, "hidden": 128, "repeats": 1, "inner": "bottleneck_psa",
            "shortcut": true, "bn_epsilon": 0.001, "bn_momentum": 0.03})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c3k2_node(one_input(input), attributes,
                                                           K_CONTEXT));
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{20, 20, 256}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 384}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 9u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SILU.id), 5u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 4u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SLICE_CHANNELS.id), 8u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 3u);
    EXPECT_EQ(list.updates().size(), 18u);
    expect_norm_options(graph, list, K_YOLO_NORM);
    EXPECT_EQ(walk(*node, "model.22"), row22_records());
}

/**
 * @brief Tests that the factory refuses a hidden width whose head count does not divide it:
 *        200 channels take max(200 / 64, 1) = 3 heads, and 3 does not divide 200.
 */
TEST(Small_BlockC3k2, TheFactoryRefusesAPSAHeadCountThatDoesNotDivideHidden)
{
    const Shape input{4, 4, 192};
    const Config::Value attributes =
        attributes_of(R"({"filters": 256, "hidden": 200, "inner": "bottleneck_psa"})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "must split into 3 parts");
}

/**
 * @brief Tests that make_c3k2_node builds row 2 from its file row: the port's shape, the 20
 *        manifest records and the row's norm options on all four norms.
 */
TEST(Small_BlockC3k2, TheFactoryBuildsRow2)
{
    const Shape input{160, 160, 32};
    const Config::Value attributes = attributes_of(
        R"({"filters": 64, "hidden": 16, "repeats": 1, "inner": "bottleneck",
            "shortcut": true, "bn_epsilon": 0.001, "bn_momentum": 0.03})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c3k2_node(one_input(input), attributes,
                                                           K_CONTEXT));
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{160, 160, 64}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 32}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 4u);
    EXPECT_EQ(list.updates().size(), 8u);
    expect_norm_options(graph, list, K_YOLO_NORM);
    EXPECT_EQ(walk(*node, "model.2"), row2_records());
}

/**
 * @brief Tests that make_c3k2_node builds row 6 from its file row, whose inner block is a
 *        C3k of two bottlenecks that add their input: the port's shape, the 45 manifest
 *        records and the row's norm options on all nine norms.
 */
TEST(Small_BlockC3k2, TheFactoryBuildsRow6)
{
    const Shape input{40, 40, 128};
    const Config::Value attributes = attributes_of(
        R"({"filters": 128, "hidden": 64, "repeats": 1, "inner": "c3k", "shortcut": true,
            "bn_epsilon": 0.001, "bn_momentum": 0.03})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c3k2_node(one_input(input), attributes,
                                                           K_CONTEXT));
    EXPECT_EQ(node->ports()[0].shape, (Shape{40, 40, 128}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 128}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 9u);
    EXPECT_EQ(list.updates().size(), 18u);
    expect_norm_options(graph, list, K_YOLO_NORM);
    EXPECT_EQ(walk(*node, "model.6"), row6_records());
}

/** @brief Tests that "shortcut": false reaches the bottlenecks of a C3k inner block. */
TEST(Small_BlockC3k2, TheFactoryReadsTheShortcutOfAC3kInnerBlock)
{
    const Shape input{4, 4, 128};
    const Config::Value attributes = attributes_of(
        R"({"filters": 128, "hidden": 64, "inner": "c3k", "shortcut": false})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c3k2_node(one_input(input), attributes,
                                                           K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 128}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 9u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 0u);
}

/**
 * @brief Tests that a row stating neither "inner", "repeats" nor "shortcut" gets one
 *        bottleneck that adds its input.
 */
TEST(Small_BlockC3k2, TheFactoryDefaultsToOneBottleneckWithAShortcut)
{
    const Shape input{4, 4, 32};
    const Config::Value attributes = attributes_of(R"({"filters": 64, "hidden": 16})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c3k2_node(one_input(input), attributes,
                                                           K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 32}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 4u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
}

/** @brief Tests that "repeats" sets the number of inner blocks and "shortcut" their add. */
TEST(Small_BlockC3k2, TheFactoryReadsRepeatsAndShortcut)
{
    const Shape input{4, 4, 32};
    const Config::Value attributes =
        attributes_of(R"({"filters": 64, "hidden": 16, "repeats": 3, "shortcut": false})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c3k2_node(one_input(input), attributes,
                                                           K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 32}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 8u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 0u);

    const std::vector<std::string> state = walk(*node, "");
    ASSERT_EQ(state.size(), 40u);
    EXPECT_EQ(state[5], record("cv2.conv.weight", Shape{64, 1, 1, 80}, StateKind::Parameter));
    EXPECT_EQ(state[30], record("m.2.cv1.conv.weight", Shape{8, 3, 3, 16}, StateKind::Parameter));
}

/** @brief Tests that an inner name outside the table fails and lists the table's names. */
TEST(Small_BlockC3k2, TheFactoryRefusesAnInnerBlockItDoesNotList)
{
    const Shape input{4, 4, 32};
    const Config::Value unlisted =
        attributes_of(R"({"filters": 64, "hidden": 16, "inner": "c2psa"})");
    const Config::Value numbered = attributes_of(R"({"filters": 64, "hidden": 16, "inner": 2})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(input), unlisted, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 2 'c3k2'", "'c2psa'",
                      "bottleneck, c3k, bottleneck_psa");
    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(input), numbered, K_CONTEXT),
                      StatusCode::InvalidDescription, "'inner'", "string");
}

/** @brief Tests that the factory refuses a hidden width of one, which no inner block halves. */
TEST(Small_BlockC3k2, TheFactoryRefusesAHiddenWidthOfOne)
{
    const Shape input{4, 4, 32};
    const Config::Value bottleneck = attributes_of(R"({"filters": 64, "hidden": 1})");
    const Config::Value c3k = attributes_of(R"({"filters": 64, "hidden": 1, "inner": "c3k"})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(input), bottleneck, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "at least 2");
    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(input), c3k, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "at least 2");
}

/** @brief Tests that the factory refuses an unknown key, a missing width and a flat input. */
TEST(Small_BlockC3k2, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{4, 4, 32};
    const Shape flat{512};
    const Config::Value unknown = attributes_of(R"({"filters": 64, "hidden": 16, "heads": 2})");
    const Config::Value no_hidden = attributes_of(R"({"filters": 64})");
    const Config::Value fits = attributes_of(R"({"filters": 64, "hidden": 16})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(input), unknown, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 2 'c3k2'", "'heads'");
    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(input), no_hidden, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "absent");
    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(flat), fits, K_CONTEXT),
                      StatusCode::InvalidDescription, "[512]", "rank-3");
}

/**
 * @brief Tests that each inner factory, called directly, refuses a row whose "inner" names
 *        the other block or is not a string, and accepts its own name.
 */
TEST(Small_BlockC3k2, AnInnerFactoryRefusesARowThatNamesAnotherInnerBlock)
{
    const Shape input{4, 4, 32};
    const Config::Value of_c3k = attributes_of(R"({"filters": 64, "hidden": 16, "inner": "c3k"})");
    const Config::Value of_bottleneck =
        attributes_of(R"({"filters": 64, "hidden": 16, "inner": "bottleneck"})");
    const Config::Value numbered = attributes_of(R"({"filters": 64, "hidden": 16, "inner": 2})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_of_bottleneck(one_input(input), of_c3k, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 2 'c3k2'", "'inner' is 'c3k'",
                      "builds 'bottleneck'");
    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_of_c3k(one_input(input), of_bottleneck, K_CONTEXT),
                      StatusCode::InvalidDescription, "'inner' is 'bottleneck'", "builds 'c3k'");
    SA_EXPECT_FAILURE(
        Vision::Blocks::make_c3k2_of_bottleneck(one_input(input), numbered, K_CONTEXT),
        StatusCode::InvalidDescription, "'inner'", "string");
    SA_EXPECT_OK(
        Vision::Blocks::make_c3k2_of_bottleneck(one_input(input), of_bottleneck, K_CONTEXT));
    SA_EXPECT_OK(Vision::Blocks::make_c3k2_of_c3k(one_input(input), of_c3k, K_CONTEXT));
}

/**
 * @brief Tests that the bottleneck_psa factory refuses a row whose "inner" names the C3k, and
 *        accepts a row that names its own block.
 */
TEST(Small_BlockC3k2, TheBottleneckPSAFactoryRefusesARowThatNamesAnotherInnerBlock)
{
    const Shape input{4, 4, 32};
    const Config::Value of_c3k = attributes_of(R"({"filters": 64, "hidden": 16, "inner": "c3k"})");
    const Config::Value of_psa =
        attributes_of(R"({"filters": 64, "hidden": 16, "inner": "bottleneck_psa"})");

    SA_EXPECT_FAILURE(
        Vision::Blocks::make_c3k2_of_bottleneck_psa(one_input(input), of_c3k, K_CONTEXT),
        StatusCode::InvalidDescription, "'inner' is 'c3k'", "builds 'bottleneck_psa'");
    SA_EXPECT_OK(
        Vision::Blocks::make_c3k2_of_bottleneck_psa(one_input(input), of_psa, K_CONTEXT));
}
