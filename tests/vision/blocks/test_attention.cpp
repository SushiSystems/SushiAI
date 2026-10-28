/**************************************************************************/
/* test_attention.cpp                                                     */
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
 * @file test_attention.cpp
 * @brief Tests Attention's trace for two heads and its state names against YOLO26n row 10.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/attention.hpp>

using namespace SushiAI;
using Vision::Blocks::Attention;
using NN::BufferUpdateList;
using NN::NormOptions;
using NN::Trace;

namespace
{
    /** @brief Holds the absolute tolerance of a double computed once and read back. */
    constexpr double K_EXACT_TOLERANCE = 1e-15;

    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Holds the width of the attention of row 10. */
    constexpr int64_t K_CHANNELS = 128;

    /** @brief Holds the head count of the attention of row 10. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Holds the query and key width of one head of row 10. */
    constexpr int64_t K_KEY_DIM = 32;

    /** @brief Holds the value width of one head of row 10. */
    constexpr int64_t K_HEAD_DIM = 64;

    /** @brief Holds the channels one head takes of qkv's result: queries, keys, values. */
    constexpr int64_t K_HEAD_WIDTH = 2 * K_KEY_DIM + K_HEAD_DIM;

    /** @brief Holds the batch extent of the traced input. */
    constexpr int64_t K_BATCH = 2;

    /** @brief Holds the H and the W of the traced input. */
    constexpr int64_t K_SIDE = 4;

    /** @brief Holds the positions of the traced input, K_SIDE squared. */
    constexpr int64_t K_TOKENS = K_SIDE * K_SIDE;

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

    /** @brief Appends the five records of a convolution and its norm under @p prefix. */
    void append_conv_norm(std::vector<std::string>& records, const std::string& prefix,
                          const Shape& weight, int64_t width)
    {
        records.push_back(record(prefix + ".conv.weight", weight, StateKind::Parameter));
        records.push_back(record(prefix + ".bn.weight", Shape{width}, StateKind::Parameter));
        records.push_back(record(prefix + ".bn.bias", Shape{width}, StateKind::Parameter));
        records.push_back(record(prefix + ".bn.running_mean", Shape{width}, StateKind::Buffer));
        records.push_back(record(prefix + ".bn.running_var", Shape{width}, StateKind::Buffer));
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

    /** @brief Returns the node of @p graph that produced @p value. */
    const Graph::Node& producer_of(const Graph::Graph& graph, Graph::ValueID value)
    {
        return graph.node(graph.value(value).producer);
    }

    /** @brief Lists the MATMUL nodes of @p graph over rank-3 operands, in emission order. */
    std::vector<const Graph::Node*> batched_products(const Graph::Graph& graph)
    {
        std::vector<const Graph::Node*> products;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_MATMUL.id && graph.value(node.inputs[0]).shape.rank() == 3)
                products.push_back(&node);
        }

        return products;
    }

    /**
     * @brief Expects @p tokens to be a RESHAPE to @p shape of a slice of @p source that starts
     *        at @p offset and holds @p count channels.
     */
    void expect_tokens_of_slice(const Graph::Graph& graph, Graph::ValueID tokens,
                                const Shape& shape, Graph::ValueID source, int64_t offset,
                                int64_t count)
    {
        const Graph::Node& reshape = producer_of(graph, tokens);
        ASSERT_EQ(reshape.op, Ops::K_RESHAPE.id);
        EXPECT_EQ(graph.value(tokens).shape, shape);

        const Graph::Node& slice = producer_of(graph, reshape.inputs[0]);
        ASSERT_EQ(slice.op, Ops::K_SLICE_CHANNELS.id);
        EXPECT_EQ(slice.inputs[0], source);

        // A slice stores its first channel as a whole number in slot 0.
        EXPECT_NEAR(slice.attributes.values[0], static_cast<double>(offset), K_EXACT_TOLERANCE);
        EXPECT_EQ(graph.value(slice.outputs[0]).shape,
                  (Shape{K_BATCH, K_SIDE, K_SIDE, count}));
    }

    /** @brief Holds a graph with the attention of row 10 traced over it in training mode. */
    struct TracedAttention
    {
        /** @brief Holds the traced graph. */
        Graph::Graph graph;

        /** @brief Holds the block, with its state declared by the trace. */
        Attention block{K_CHANNELS, K_HEADS, K_YOLO_NORM};

        /** @brief Holds the running-statistics updates the three norms recorded. */
        BufferUpdateList list;

        /** @brief Holds the [K_BATCH, K_SIDE, K_SIDE, K_CHANNELS] input's id. */
        Graph::ValueID x = Graph::K_NO_VALUE;

        /** @brief Holds the id forward() returned. */
        Graph::ValueID y = Graph::K_NO_VALUE;

        /** @brief Declares the input and traces the block over it. */
        TracedAttention()
        {
            const Graph::GraphBuilder builder(graph);
            x = builder.input(Shape{K_BATCH, K_SIDE, K_SIDE, K_CHANNELS}, "x");
            y = block.forward(Trace::training(builder, list), x);
        }

        /** @brief Returns the id of qkv's normalised result, which every slice reads. */
        Graph::ValueID mixed() const
        {
            return graph.nodes()[first_node(graph, Ops::K_BATCHNORM2D.id)].outputs[0];
        }
    };

} // namespace

/** @brief Tests that 128 channels over two heads give a head of 64 and a key of 32. */
TEST(Small_BlockAttention, TwoHeadsOver128ChannelsGiveAHeadOf64AndAKeyOf32)
{
    const Attention block(K_CHANNELS, K_HEADS, K_YOLO_NORM);

    EXPECT_EQ(block.heads, K_HEADS);
    EXPECT_EQ(block.head_dim, K_HEAD_DIM);
    EXPECT_EQ(block.key_dim, K_KEY_DIM);
    EXPECT_EQ(block.qkv.conv.out_channels, K_CHANNELS + 2 * K_HEADS * K_KEY_DIM);
    EXPECT_EQ(block.pe.conv.geometry.kernel_h, 3);
    EXPECT_EQ(block.pe.conv.geometry.pad_top, 1);
}

/** @brief Tests that the block keeps the per-sample shape. */
TEST(Small_BlockAttention, OutputShapeIsTheInputShape)
{
    const Attention block(K_CHANNELS, K_HEADS, K_YOLO_NORM);

    EXPECT_EQ(block.output_shape(Shape{20, 20, K_CHANNELS}).value(), (Shape{20, 20, K_CHANNELS}));
}

/**
 * @brief Tests that two heads trace 6 slices, 2 scales, 4 batched products, 2 softmax groups,
 *        1 add, 1 depthwise convolution and 2 joins, and keep the input's shape.
 */
TEST(Small_BlockAttention, TwoHeadsEmitSixSlicesTwoScalesFourProductsTwoSoftmaxesAndOneAdd)
{
    const TracedAttention traced;
    const Graph::Graph& graph = traced.graph;

    EXPECT_EQ(count_nodes(graph, Ops::K_SLICE_CHANNELS.id), 6u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SCALE.id), 2u);
    EXPECT_EQ(batched_products(graph).size(), 4u);
    EXPECT_EQ(count_nodes(graph, Ops::K_LOG_SOFTMAX.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_EXP.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_DEPTHWISE_CONV2D.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 3u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SILU.id), 0u);
    EXPECT_EQ(graph.value(traced.y).shape, (Shape{K_BATCH, K_SIDE, K_SIDE, K_CHANNELS}));
}

/**
 * @brief Tests that each head's score product reads its scaled queries, then its keys with
 *        the second operand transposed, and yields [B, N, N].
 */
TEST(Small_BlockAttention, TheScoreProductReadsScaledQueriesThenTransposedKeys)
{
    const TracedAttention traced;
    const Graph::Graph& graph = traced.graph;
    const std::vector<const Graph::Node*> products = batched_products(graph);
    ASSERT_EQ(products.size(), 4u);

    const Shape key_tokens{K_BATCH, K_TOKENS, K_KEY_DIM};
    for (int64_t head = 0; head < K_HEADS; ++head)
    {
        SCOPED_TRACE("head " + std::to_string(head));
        const Graph::Node& scores = *products[static_cast<std::size_t>(2 * head)];
        ASSERT_EQ(scores.inputs.size(), 2u);

        // A product stores transpose_a and transpose_b as 0 or 1 in slots 0 and 1.
        EXPECT_EQ(scores.attributes.values[0], 0.0);
        EXPECT_EQ(scores.attributes.values[1], 1.0);
        EXPECT_EQ(graph.value(scores.outputs[0]).shape, (Shape{K_BATCH, K_TOKENS, K_TOKENS}));

        const Graph::Node& scale = producer_of(graph, scores.inputs[0]);
        ASSERT_EQ(scale.op, Ops::K_SCALE.id);
        expect_tokens_of_slice(graph, scale.inputs[0], key_tokens, traced.mixed(),
                               head * K_HEAD_WIDTH, K_KEY_DIM);
        expect_tokens_of_slice(graph, scores.inputs[1], key_tokens, traced.mixed(),
                               head * K_HEAD_WIDTH + K_KEY_DIM, K_KEY_DIM);
    }
}

/**
 * @brief Tests that each head's value product reads the softmax of its scores, then its
 *        values, with neither operand transposed, and yields [B, N, head_dim].
 */
TEST(Small_BlockAttention, TheValueProductReadsTheSoftmaxOfTheScoresThenTheValues)
{
    const TracedAttention traced;
    const Graph::Graph& graph = traced.graph;
    const std::vector<const Graph::Node*> products = batched_products(graph);
    ASSERT_EQ(products.size(), 4u);

    const Shape value_tokens{K_BATCH, K_TOKENS, K_HEAD_DIM};
    for (int64_t head = 0; head < K_HEADS; ++head)
    {
        SCOPED_TRACE("head " + std::to_string(head));
        const Graph::Node& scores = *products[static_cast<std::size_t>(2 * head)];
        const Graph::Node& weighed = *products[static_cast<std::size_t>(2 * head + 1)];
        ASSERT_EQ(weighed.inputs.size(), 2u);

        EXPECT_EQ(weighed.attributes.values[0], 0.0);
        EXPECT_EQ(weighed.attributes.values[1], 0.0);
        EXPECT_EQ(graph.value(weighed.outputs[0]).shape, value_tokens);

        // softmax() emits RESHAPE, LOG_SOFTMAX, EXP, RESHAPE; the chain is walked backwards.
        const Graph::Node& unflatten = producer_of(graph, weighed.inputs[0]);
        ASSERT_EQ(unflatten.op, Ops::K_RESHAPE.id);
        const Graph::Node& exponent = producer_of(graph, unflatten.inputs[0]);
        ASSERT_EQ(exponent.op, Ops::K_EXP.id);
        const Graph::Node& log_softmax = producer_of(graph, exponent.inputs[0]);
        ASSERT_EQ(log_softmax.op, Ops::K_LOG_SOFTMAX.id);
        EXPECT_EQ(graph.value(log_softmax.inputs[0]).shape,
                  (Shape{K_BATCH * K_TOKENS, K_TOKENS}));
        const Graph::Node& flatten = producer_of(graph, log_softmax.inputs[0]);
        ASSERT_EQ(flatten.op, Ops::K_RESHAPE.id);
        EXPECT_EQ(flatten.inputs[0], scores.outputs[0]);

        expect_tokens_of_slice(graph, weighed.inputs[1], value_tokens, traced.mixed(),
                               head * K_HEAD_WIDTH + 2 * K_KEY_DIM, K_HEAD_DIM);
    }
}

/** @brief Tests that both scales carry 1 / sqrt(32), to 1e-15 absolute. */
TEST(Small_BlockAttention, TheQueriesAreScaledByTheInverseRootOfTheKeyWidth)
{
    const TracedAttention traced;
    std::size_t scales = 0;

    for (const Graph::Node& node : traced.graph.nodes())
    {
        if (node.op != Ops::K_SCALE.id)
            continue;

        EXPECT_NEAR(node.attributes.values[0], 1.0 / std::sqrt(32.0), K_EXACT_TOLERANCE);
        EXPECT_NEAR(node.attributes.values[0], 0.17677669529663687, K_EXACT_TOLERANCE);
        ++scales;
    }

    EXPECT_EQ(scales, 2u);
}

/**
 * @brief Tests that the add reads the attended heads joined, then pe's result over the values
 *        of both heads joined, and that proj reads the add.
 */
TEST(Small_BlockAttention, TheAddReadsTheAttendedHeadsThenThePositionalConvolution)
{
    const TracedAttention traced;
    const Graph::Graph& graph = traced.graph;
    const std::vector<const Graph::Node*> products = batched_products(graph);
    ASSERT_EQ(products.size(), 4u);
    ASSERT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);
    const Graph::Node& add = graph.nodes()[first_node(graph, Ops::K_ADD.id)];
    ASSERT_EQ(add.inputs.size(), 2u);

    const Graph::Node& attended = producer_of(graph, add.inputs[0]);
    ASSERT_EQ(attended.op, Ops::K_CONCAT_CHANNELS.id);
    ASSERT_EQ(attended.inputs.size(), 2u);

    // pe is a DWConvLinear, so its result comes out of its BATCHNORM2D.
    const Graph::Node& norm = producer_of(graph, add.inputs[1]);
    ASSERT_EQ(norm.op, Ops::K_BATCHNORM2D.id);
    const Graph::Node& depthwise = producer_of(graph, norm.inputs[0]);
    ASSERT_EQ(depthwise.op, Ops::K_DEPTHWISE_CONV2D.id);
    const Graph::Node& values = producer_of(graph, depthwise.inputs[0]);
    ASSERT_EQ(values.op, Ops::K_CONCAT_CHANNELS.id);
    ASSERT_EQ(values.inputs.size(), 2u);
    EXPECT_EQ(graph.value(values.outputs[0]).shape,
              (Shape{K_BATCH, K_SIDE, K_SIDE, K_CHANNELS}));

    for (int64_t head = 0; head < K_HEADS; ++head)
    {
        SCOPED_TRACE("head " + std::to_string(head));
        const std::size_t part = static_cast<std::size_t>(head);

        const Graph::Node& map = producer_of(graph, attended.inputs[part]);
        ASSERT_EQ(map.op, Ops::K_RESHAPE.id);
        EXPECT_EQ(map.inputs[0], products[2 * part + 1]->outputs[0]);
        EXPECT_EQ(graph.value(attended.inputs[part]).shape,
                  (Shape{K_BATCH, K_SIDE, K_SIDE, K_HEAD_DIM}));

        const Graph::Node& slice = producer_of(graph, values.inputs[part]);
        ASSERT_EQ(slice.op, Ops::K_SLICE_CHANNELS.id);
        EXPECT_NEAR(slice.attributes.values[0],
                    static_cast<double>(head * K_HEAD_WIDTH + 2 * K_KEY_DIM), K_EXACT_TOLERANCE);
    }

    // proj is a pointwise ConvLinear: it flattens the add, and its norm yields the result.
    const Graph::Node& last = producer_of(graph, traced.y);
    EXPECT_EQ(last.op, Ops::K_BATCHNORM2D.id);
    std::size_t readers = 0;
    for (const Graph::Node& node : graph.nodes())
    {
        if (node.op == Ops::K_RESHAPE.id && node.inputs[0] == add.outputs[0])
            ++readers;
    }
    EXPECT_EQ(readers, 1u);
}

/**
 * @brief Tests that a traced Attention(128, 2) yields the fifteen records MANIFEST.json lists
 *        under model.10.m.0.attn, records 175 to 189: names, shapes and order.
 */
TEST(Small_BlockAttention, StateMatchesManifestRecords175To189)
{
    TracedAttention traced;

    // MANIFEST.json of the registration fixtures, weights.sushi tensors, from line 3004.
    const std::string prefix = "model.10.m.0.attn";
    std::vector<std::string> expected;
    append_conv_norm(expected, prefix + ".qkv", Shape{256, 1, 1, 128}, 256);
    append_conv_norm(expected, prefix + ".proj", Shape{128, 1, 1, 128}, 128);
    append_conv_norm(expected, prefix + ".pe", Shape{3, 3, 128}, 128);

    EXPECT_EQ(walk(traced.block, prefix), expected);
}

/** @brief Tests that an evaluation trace folds the three norms and keeps the four products. */
TEST(Small_BlockAttention, AnEvaluationTraceFoldsTheThreeNorms)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_BATCH, K_SIDE, K_SIDE, K_CHANNELS}, "x");
    Attention block(K_CHANNELS, K_HEADS, K_YOLO_NORM);

    const Graph::ValueID y = block.forward(Trace::evaluation(builder), x);

    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 0u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM_FOLD.id), 3u);
    EXPECT_EQ(batched_products(graph).size(), 4u);
    EXPECT_EQ(graph.value(y).shape, (Shape{K_BATCH, K_SIDE, K_SIDE, K_CHANNELS}));
}

/** @brief Tests that a width the head count does not divide is refused, naming both. */
TEST(Small_BlockAttention, AWidthTheHeadCountDoesNotDivideIsRefused)
{
    SA_EXPECT_ERROR(static_cast<void>(Attention(100, 3, K_YOLO_NORM)), "head count divides",
                    "100 channels", "3 heads");
    SA_EXPECT_ERROR(static_cast<void>(Attention(128, 0, K_YOLO_NORM)), "head count divides",
                    "128 channels", "0 heads");
}

/** @brief Tests that one channel per head is refused, since it leaves no key channel. */
TEST(Small_BlockAttention, OneChannelPerHeadIsRefused)
{
    SA_EXPECT_ERROR(static_cast<void>(Attention(2, 2, K_YOLO_NORM)),
                    "at least 2 channels per head", "2 channels", "2 heads");
}
