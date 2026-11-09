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
 * @file test_attention.cpp
 * @brief Tests Attention's trace for two heads, its head check and its state names against YOLO26n row 10.
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
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/attention.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Vision::Blocks::Attention;
using Vision::Blocks::check_attention_heads;
using NN::BufferUpdateList;
using NN::NormOptions;
using NN::Trace;

namespace
{
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

    /** @brief Holds the batch extent of the traced input. */
    constexpr int64_t K_BATCH = 2;

    /** @brief Holds the H and the W of the traced input. */
    constexpr int64_t K_SIDE = 4;

    /** @brief Holds the positions of the traced input, K_SIDE squared. */
    constexpr int64_t K_TOKENS = K_SIDE * K_SIDE;

    /** @brief Holds the norms of a block of row 10: qkv, proj and pe. */
    constexpr std::size_t K_NORMS = 3;

    /** @brief Holds the buffer updates one norm requests: its running mean and its variance. */
    constexpr std::size_t K_UPDATES_PER_NORM = 2;

    /** @brief Holds 1 / sqrt(32), the double nearest to it, written out as a second oracle. */
    constexpr double K_INVERSE_ROOT_OF_KEY = 0.17677669529663687;

    /** @brief Names which of the two kinds of state a record is. */
    enum class StateKind : std::uint8_t
    {
        Parameter, /**< A learnable tensor. */
        Buffer,    /**< A stored tensor no gradient reaches. */
    };

    /** @brief Holds one head layout the check refuses and the words its message names. */
    struct RefusedHeads
    {
        std::string name;     /**< The row's name. */
        int64_t channels;     /**< The width given. */
        int64_t heads;        /**< The head count given. */
        std::string rule;     /**< The rule the message states. */
        std::string channels_named; /**< The width as the message names it. */
        std::string heads_named;    /**< The head count as the message names it. */
    };

    /** @brief Runs one refused head layout. */
    class Small_BlockAttentionHeads : public ::testing::TestWithParam<RefusedHeads>
    {

    };

    /** @brief Runs the process-stopping cases in a child process. */
    class Medium_BlockAttentionDeath : public ::testing::Test
    {

    };

    /** @brief Names a row by its name field. */
    std::string row_name(const ::testing::TestParamInfo<RefusedHeads>& info)
    {
        return info.param.name;
    }

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

    /** @brief Counts the MATMUL nodes of @p graph over rank-3 operands that yield @p shape. */
    std::size_t count_products_of(const Graph::Graph& graph, const Shape& shape)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_MATMUL.id && graph.value(node.inputs[0]).shape.rank() == 3
                && graph.value(node.outputs[0]).shape == shape)
                ++count;
        }

        return count;
    }

    /** @brief Lists the factor each SCALE node of @p graph carries, in emission order. */
    std::vector<double> scale_factors(const Graph::Graph& graph)
    {
        std::vector<double> factors;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_SCALE.id)
                factors.push_back(node.attributes.values[0]);
        }

        return factors;
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
    };

} // namespace

/** @brief Tests that 128 channels over two heads give a head of 64 and a key of 32. */
TEST(Small_BlockAttention, TwoHeadsOver128ChannelsGiveAHeadOf64AndAKeyOf32)
{
    const Attention block(K_CHANNELS, K_HEADS, K_YOLO_NORM);

    EXPECT_EQ(block.heads, K_HEADS);
    EXPECT_EQ(block.head_dim, K_HEAD_DIM);
    EXPECT_EQ(block.key_dim, K_KEY_DIM);
}

/** @brief Tests that the block keeps the per-sample shape. */
TEST(Small_BlockAttention, OutputShapeIsTheInputShape)
{
    const Attention block(K_CHANNELS, K_HEADS, K_YOLO_NORM);

    EXPECT_EQ(block.output_shape(Shape{20, 20, K_CHANNELS}).value(), (Shape{20, 20, K_CHANNELS}));
}

/**
 * @brief Tests that a training trace keeps the input's shape and requests the running-statistics
 *        updates of the three norms, a mean and a variance each.
 */
TEST(Small_BlockAttention, ATrainingTraceKeepsTheInputShapeAndUpdatesTheThreeNorms)
{
    const TracedAttention traced;

    EXPECT_EQ(traced.graph.value(traced.y).shape, (Shape{K_BATCH, K_SIDE, K_SIDE, K_CHANNELS}));
    EXPECT_EQ(traced.list.updates().size(), K_NORMS * K_UPDATES_PER_NORM);
}

/** @brief Tests that each head scores every position against every other, as [B, N, N]. */
TEST(Small_BlockAttention, EachHeadScoresEveryPositionAgainstEveryOther)
{
    const TracedAttention traced;

    EXPECT_EQ(count_products_of(traced.graph, Shape{K_BATCH, K_TOKENS, K_TOKENS}),
              static_cast<std::size_t>(K_HEADS));
}

/** @brief Tests that each head weighs its values by the scores, as [B, N, head_dim]. */
TEST(Small_BlockAttention, EachHeadWeighsItsValuesAsABatchOfTokenRows)
{
    const TracedAttention traced;

    EXPECT_EQ(count_products_of(traced.graph, Shape{K_BATCH, K_TOKENS, K_HEAD_DIM}),
              static_cast<std::size_t>(K_HEADS));
}

/** @brief Tests that every head scales its queries by 1 / sqrt(key width). */
TEST(Small_BlockAttention, TheQueriesAreScaledByTheInverseRootOfTheKeyWidth)
{
    const TracedAttention traced;
    const std::vector<double> factors = scale_factors(traced.graph);

    EXPECT_EQ(factors, std::vector<double>(static_cast<std::size_t>(K_HEADS),
                                           1.0 / std::sqrt(static_cast<double>(K_KEY_DIM))));
    ASSERT_FALSE(factors.empty());
    // One rounding of the square root and one of the quotient separate the two oracles.
    EXPECT_TRUE(TestSupport::is_within(
        TestSupport::measure_relative_error<double>(factors.front(), K_INVERSE_ROOT_OF_KEY, 2)));
}

/**
 * @brief Tests that the sum of the attended heads and the positional convolution has the shape
 *        of the input, which proj then keeps.
 */
TEST(Small_BlockAttention, TheAttendedHeadsAndThePositionalConvolutionAddUpToTheInputShape)
{
    const TracedAttention traced;
    const Graph::Graph& graph = traced.graph;
    const std::size_t add = first_node(graph, Ops::K_ADD.id);
    ASSERT_LT(add, graph.node_count());

    EXPECT_EQ(graph.value(graph.nodes()[add].outputs[0]).shape,
              (Shape{K_BATCH, K_SIDE, K_SIDE, K_CHANNELS}));
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

/** @brief Tests that an evaluation trace leaves no running-statistics norm and keeps the shape. */
TEST(Small_BlockAttention, AnEvaluationTraceLeavesNoRunningStatisticsNode)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_BATCH, K_SIDE, K_SIDE, K_CHANNELS}, "x");
    Attention block(K_CHANNELS, K_HEADS, K_YOLO_NORM);

    const Graph::ValueID y = block.forward(Trace::evaluation(builder), x);

    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 0u);
    EXPECT_EQ(graph.value(y).shape, (Shape{K_BATCH, K_SIDE, K_SIDE, K_CHANNELS}));
}

/** @brief Tests that the head check accepts the layout of row 10. */
TEST(Small_BlockAttention, TheHeadsOfRow10AreAccepted)
{
    SA_EXPECT_OK(check_attention_heads(K_CHANNELS, K_HEADS));
}

/** @brief Tests that a head layout the block cannot take is refused naming the counts. */
TEST_P(Small_BlockAttentionHeads, ALayoutThatDoesNotFitIsRefusedNamingTheCounts)
{
    const RefusedHeads& row = GetParam();

    SA_EXPECT_FAILURE(check_attention_heads(row.channels, row.heads),
                      StatusCode::InvalidDescription, row.rule, row.channels_named,
                      row.heads_named);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockAttentionHeads,
    ::testing::Values(
        RefusedHeads{"AWidthTheHeadCountDoesNotDivide", 100, 3, "head count divides",
                     "100 channels", "3 heads"},
        RefusedHeads{"NoHeads", 128, 0, "head count divides", "128 channels", "0 heads"},
        RefusedHeads{"OneChannelPerHead", 2, 2, "at least 2 channels per head", "2 channels",
                     "2 heads"}),
    row_name);

/** @brief Tests that the constructor stops the process on the refusal of the head check. */
TEST_F(Medium_BlockAttentionDeath, AWidthTheHeadCountDoesNotDivideIsFatal)
{
    SA_EXPECT_FATAL(static_cast<void>(Attention(100, 3, K_YOLO_NORM)), "head count divides",
                    "100 channels", "3 heads");
}

/** @brief Tests that the constructor stops the process when a head would hold one channel. */
TEST_F(Medium_BlockAttentionDeath, OneChannelPerHeadIsFatal)
{
    SA_EXPECT_FATAL(static_cast<void>(Attention(2, 2, K_YOLO_NORM)),
                    "at least 2 channels per head", "2 channels", "2 heads");
}
