/**************************************************************************/
/* test_psa_block.cpp                                                     */
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
 * @file test_psa_block.cpp
 * @brief Tests PSABlock's two residual adds, its shape rule and its state names against the block
 *        at YOLO26n row 10.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/psa_block.hpp>

using namespace SushiAI;
using NN::BufferUpdateList;
using NN::NormOptions;
using NN::Trace;
using Vision::Blocks::PSABlock;

namespace
{
    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Holds the width of row 10's attention: 128 channels over 2 heads. */
    constexpr int64_t K_CHANNELS = 128;

    /** @brief Holds the head count of row 10's attention. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Holds the prefix MANIFEST.json gives the one PSABlock of row 10. */
    constexpr std::string_view K_PREFIX = "model.10.m.0";

    /** @brief Runs the process-stopping case in a child process. */
    class Medium_BlockPsaBlockDeath : public ::testing::Test
    {

    };

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

    /** @brief Counts the norms a walk lists, one running mean each. */
    std::size_t count_norms(const std::vector<std::string>& records)
    {
        std::size_t norms = 0;
        for (const std::string& line : records)
        {
            if (line.find(".bn.running_mean ") != std::string::npos)
                ++norms;
        }

        return norms;
    }

    /** @brief Counts the ADD nodes of @p graph that read @p value as their first operand. */
    std::size_t count_adds_reading_first(const Graph::Graph& graph, Graph::ValueID value)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_ADD.id && node.inputs[0] == value)
                ++count;
        }

        return count;
    }

    /** @brief Lists the 25 records MANIFEST.json holds under model.10.m.0, in walk order. */
    std::vector<std::string> psa_records()
    {
        return {
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

} // namespace

/** @brief Tests that the block keeps the per-sample shape. */
TEST(Small_BlockPsaBlock, OutputShapeIsTheInputShape)
{
    const PSABlock block(K_CHANNELS, K_HEADS, K_YOLO_NORM);

    EXPECT_EQ(block.output_shape(Shape{20, 20, K_CHANNELS}).value(), (Shape{20, 20, K_CHANNELS}));
}

/**
 * @brief Tests that a training trace keeps the input's shape and requests a mean and a
 *        variance update for each norm of its state.
 */
TEST(Small_BlockPsaBlock, ATrainingTraceKeepsTheInputShapeAndUpdatesEveryNorm)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, K_CHANNELS}, "x");
    PSABlock block(K_CHANNELS, K_HEADS, K_YOLO_NORM);
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, K_CHANNELS}));
    EXPECT_EQ(list.updates().size(), 2 * count_norms(walk(block, K_PREFIX)));
}

/**
 * @brief Tests that the input is added back once, as the first operand of a residual add, and
 *        that the block's result is produced by an add.
 */
TEST(Small_BlockPsaBlock, TheInputIsAddedBackOnceAndTheResultIsAResidualSum)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, K_CHANNELS}, "x");
    PSABlock block(K_CHANNELS, K_HEADS, K_YOLO_NORM);
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_EQ(count_adds_reading_first(graph, x), 1u);
    EXPECT_EQ(graph.node(graph.value(y).producer).op, Ops::K_ADD.id);
}

/**
 * @brief Tests that the state of the block yields the 25 records MANIFEST.json lists under
 *        model.10.m.0, names, NHWC shapes and order.
 */
TEST(Small_BlockPsaBlock, StateMatchesManifestRow10Block)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, K_CHANNELS}, "x");
    PSABlock block(K_CHANNELS, K_HEADS, K_YOLO_NORM);
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    EXPECT_EQ(walk(block, K_PREFIX), psa_records());
}

/** @brief Tests that a width the head count does not divide stops the process, as Attention does. */
TEST_F(Medium_BlockPsaBlockDeath, AWidthTheHeadCountDoesNotDivideIsFatal)
{
    SA_EXPECT_FATAL(static_cast<void>(PSABlock(100, 3, K_YOLO_NORM)), "head count divides",
                    "100 channels", "3 heads");
}
