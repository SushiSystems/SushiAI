/**************************************************************************/
/* test_block_sppf.cpp                                                    */
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
 * @file test_block_sppf.cpp
 * @brief Tests SPPF's trace, its state names against YOLO26n row 9, and its factory.
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
#include <SushiAI/nn/blocks/sppf.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include "../../common/test_status.hpp"

using namespace SushiAI;
using NN::BufferUpdateList;
using NN::NormOptions;
using NN::SPPF;
using NN::Trace;

namespace
{
    /** @brief Holds the absolute tolerance of a double stored and read back unchanged. */
    constexpr double K_EXACT_TOLERANCE = 1e-15;

    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 9 'sppf'";

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
TEST(Unit_BlockSppf, OutputShapeKeepsTheExtentsAndSetsTheWidth)
{
    const SPPF row9(256, 256, 5, 3, true, K_YOLO_NORM);
    const SPPF widening(64, 96, 5, 3, true, K_YOLO_NORM);

    EXPECT_EQ(row9.output_shape(Shape{20, 20, 256}), (Shape{20, 20, 256}));
    EXPECT_EQ(widening.output_shape(Shape{5, 7, 64}), (Shape{5, 7, 96}));
}

/** @brief Tests that the pool is a stride-1 window of the kernel given, padded by half of it. */
TEST(Unit_BlockSppf, ThePoolKeepsTheExtents)
{
    const SPPF block(256, 256, 5, 3, true, K_YOLO_NORM);

    EXPECT_EQ(block.pool.geometry.kernel_h, 5);
    EXPECT_EQ(block.pool.geometry.kernel_w, 5);
    EXPECT_EQ(block.pool.geometry.stride_h, 1);
    EXPECT_EQ(block.pool.geometry.stride_w, 1);
    EXPECT_EQ(block.pool.geometry.pad_top, 2);
    EXPECT_EQ(block.pool.geometry.pad_left, 2);
    EXPECT_EQ(block.pool.output_shape(Shape{20, 20, 128}), (Shape{20, 20, 128}));
}

/**
 * @brief Tests that row 9 traces two convolutions, three MAXPOOL2D, one CONCAT_CHANNELS, one
 *        ADD and a single SILU, which is cv2's.
 */
TEST(Unit_BlockSppf, Row9EmitsThreePoolsOneConcatAndOneAdd)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 256}, "x");
    SPPF block(256, 256, 5, 3, true, K_YOLO_NORM);
    BufferUpdateList list;

    const Graph::ValueId y = block.forward(Trace::training(builder, list), x);

    EXPECT_EQ(count_nodes(graph, Ops::K_MATMUL.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 2u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SILU.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_MAXPOOL2D.id), 3u);
    EXPECT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);
    EXPECT_EQ(list.updates().size(), 4u);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 256}));
}

/**
 * @brief Tests that the join reads cv1's normalised result, with no SILU after it, then each
 *        pool, and that each pool reads the stage before it.
 */
TEST(Unit_BlockSppf, TheJoinReadsTheNarrowedValueThenEachPoolInAChain)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 256}, "x");
    SPPF block(256, 256, 5, 3, true, K_YOLO_NORM);
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    ASSERT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
    const Graph::Node& join = graph.nodes()[first_node(graph, Ops::K_CONCAT_CHANNELS.id)];
    ASSERT_EQ(join.inputs.size(), 4u);

    // cv1 is a ConvLinear, so the value the pools start from comes out of its BATCHNORM2D.
    EXPECT_EQ(graph.node(graph.value(join.inputs[0]).producer).op, Ops::K_BATCHNORM2D.id);
    for (std::size_t stage = 1; stage < join.inputs.size(); ++stage)
    {
        const Graph::Node& pool = graph.node(graph.value(join.inputs[stage]).producer);
        ASSERT_EQ(pool.op, Ops::K_MAXPOOL2D.id) << "stage " << stage;
        EXPECT_EQ(pool.inputs[0], join.inputs[stage - 1]) << "stage " << stage;
        EXPECT_EQ(graph.value(join.inputs[stage]).shape, (Shape{2, 4, 4, 128}))
            << "stage " << stage;
    }

    EXPECT_EQ(graph.value(join.outputs[0]).shape, (Shape{2, 4, 4, 512}));
}

/** @brief Tests that the closing ADD reads cv2's result first and the block's input second. */
TEST(Unit_BlockSppf, TheClosingAddReadsTheResultThenTheInput)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 256}, "x");
    SPPF block(256, 256, 5, 3, true, K_YOLO_NORM);
    BufferUpdateList list;

    const Graph::ValueId y = block.forward(Trace::training(builder, list), x);

    EXPECT_TRUE(block.adds_input);
    ASSERT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);
    const Graph::Node& add = graph.nodes()[first_node(graph, Ops::K_ADD.id)];
    ASSERT_EQ(add.inputs.size(), 2u);
    EXPECT_EQ(graph.node(graph.value(add.inputs[0]).producer).op, Ops::K_SILU.id);
    EXPECT_EQ(add.inputs[1], x);
    EXPECT_EQ(add.outputs[0], y);
}

/** @brief Tests that no ADD is traced without a shortcut, or with one over unequal widths. */
TEST(Unit_BlockSppf, NoAddWithoutAShortcutOrOverUnequalWidths)
{
    Graph::Graph plain_graph;
    const Graph::GraphBuilder plain_builder(plain_graph);
    SPPF plain(64, 64, 5, 3, false, K_YOLO_NORM);
    static_cast<void>(plain.forward(Trace::evaluation(plain_builder),
                                    plain_builder.input(Shape{2, 4, 4, 64}, "x")));

    Graph::Graph widening_graph;
    const Graph::GraphBuilder widening_builder(widening_graph);
    SPPF widening(64, 96, 5, 3, true, K_YOLO_NORM);
    static_cast<void>(widening.forward(Trace::evaluation(widening_builder),
                                       widening_builder.input(Shape{2, 4, 4, 64}, "x")));

    EXPECT_FALSE(plain.adds_input);
    EXPECT_EQ(count_nodes(plain_graph, Ops::K_ADD.id), 0u);
    EXPECT_FALSE(widening.adds_input);
    EXPECT_EQ(count_nodes(widening_graph, Ops::K_ADD.id), 0u);
}

/** @brief Tests that the pool count sets the number of pools and the width cv2 reads. */
TEST(Unit_BlockSppf, ThePoolCountSetsThePoolsAndTheJoinedWidth)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 64}, "x");
    SPPF block(64, 64, 3, 1, false, K_YOLO_NORM);

    static_cast<void>(block.forward(Trace::evaluation(builder), x));

    EXPECT_EQ(count_nodes(graph, Ops::K_MAXPOOL2D.id), 1u);
    EXPECT_EQ(block.cv1.conv.weight.shape(), (Shape{32, 1, 1, 64}));
    EXPECT_EQ(block.cv2.conv.weight.shape(), (Shape{64, 1, 1, 64}));
}

/**
 * @brief Tests that a traced SPPF(256, 256, 5, 3) yields the ten records MANIFEST.json lists
 *        under model.9, names, NHWC shapes and order.
 */
TEST(Unit_BlockSppf, StateMatchesManifestRow9)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 256}, "x");
    SPPF block(256, 256, 5, 3, true, K_YOLO_NORM);
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    // tests/fixtures/yolo26/MANIFEST.json, weights.sacp tensors, from line 2680.
    const std::vector<std::string> expected = {
        record("model.9.cv1.conv.weight", Shape{128, 1, 1, 256}, StateKind::Parameter),
        record("model.9.cv1.bn.weight", Shape{128}, StateKind::Parameter),
        record("model.9.cv1.bn.bias", Shape{128}, StateKind::Parameter),
        record("model.9.cv1.bn.running_mean", Shape{128}, StateKind::Buffer),
        record("model.9.cv1.bn.running_var", Shape{128}, StateKind::Buffer),
        record("model.9.cv2.conv.weight", Shape{256, 1, 1, 512}, StateKind::Parameter),
        record("model.9.cv2.bn.weight", Shape{256}, StateKind::Parameter),
        record("model.9.cv2.bn.bias", Shape{256}, StateKind::Parameter),
        record("model.9.cv2.bn.running_mean", Shape{256}, StateKind::Buffer),
        record("model.9.cv2.bn.running_var", Shape{256}, StateKind::Buffer),
    };

    EXPECT_EQ(walk(block, "model.9"), expected);
}

/**
 * @brief Tests that make_sppf_node builds row 9 from its file row: the port's shape, three
 *        pools, the closing add and the state names.
 */
TEST(Unit_BlockSppf, TheFactoryBuildsRow9)
{
    const Shape input{20, 20, 256};
    const Config::Value attributes = attributes_of(
        R"({"filters": 256, "kernel": 5, "pools": 3, "shortcut": true})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_sppf_node(one_input(input), attributes, K_CONTEXT));
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{20, 20, 256}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 256}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_MAXPOOL2D.id), 3u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);

    const std::vector<std::string> state = walk(*node, "model.9");
    ASSERT_EQ(state.size(), 10u);
    EXPECT_EQ(state[0],
              record("model.9.cv1.conv.weight", Shape{128, 1, 1, 256}, StateKind::Parameter));
    EXPECT_EQ(state[5],
              record("model.9.cv2.conv.weight", Shape{256, 1, 1, 512}, StateKind::Parameter));
}

/**
 * @brief Tests that a row stating only "filters" gets a 5x5 pool applied three times and no
 *        closing add.
 */
TEST(Unit_BlockSppf, TheFactoryDefaultsToThreePoolsOfFiveAndNoShortcut)
{
    const Shape input{4, 4, 64};
    const Config::Value attributes = attributes_of(R"({"filters": 64})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_sppf_node(one_input(input), attributes, K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 64}, "x");
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    EXPECT_EQ(count_nodes(graph, Ops::K_MAXPOOL2D.id), 3u);
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 0u);

    const std::vector<std::string> state = walk(*node, "");
    ASSERT_EQ(state.size(), 10u);
    EXPECT_EQ(state[5], record("cv2.conv.weight", Shape{64, 1, 1, 128}, StateKind::Parameter));
}

/** @brief Tests that the factory refuses an even kernel, which would change H and W. */
TEST(Unit_BlockSppf, TheFactoryRefusesAnEvenKernel)
{
    const Shape input{4, 4, 64};
    const Config::Value attributes = attributes_of(R"({"filters": 64, "kernel": 4})");

    SA_EXPECT_FAILURE(NN::make_sppf_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 9 'sppf'", "'kernel'", "odd", "4");
}

/** @brief Tests that the factory refuses a one-channel input, which leaves no hidden channel. */
TEST(Unit_BlockSppf, TheFactoryRefusesAnInputOfOneChannel)
{
    const Shape input{4, 4, 1};
    const Config::Value attributes = attributes_of(R"({"filters": 64})");

    SA_EXPECT_FAILURE(NN::make_sppf_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "[4, 4, 1]", "at least 2 channels");
}

/** @brief Tests that the factory refuses an unknown key, a missing width and zero pools. */
TEST(Unit_BlockSppf, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{4, 4, 64};
    const Config::Value unknown = attributes_of(R"({"filters": 64, "hidden": 32})");
    const Config::Value no_filters = attributes_of(R"({"kernel": 5})");
    const Config::Value zero_pools = attributes_of(R"({"filters": 64, "pools": 0})");

    SA_EXPECT_FAILURE(NN::make_sppf_node(one_input(input), unknown, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 9 'sppf'", "'hidden'");
    SA_EXPECT_FAILURE(NN::make_sppf_node(one_input(input), no_filters, K_CONTEXT),
                      StatusCode::InvalidDescription, "'filters'", "absent");
    SA_EXPECT_FAILURE(NN::make_sppf_node(one_input(input), zero_pools, K_CONTEXT),
                      StatusCode::InvalidDescription, "'pools'", "positive");
}

/** @brief Tests that the factory hands "bn_epsilon" and "bn_momentum" to both norms. */
TEST(Unit_BlockSppf, TheFactoryHandsTheNormOptionsToBothNorms)
{
    const Shape input{4, 4, 256};
    const Config::Value attributes = attributes_of(
        R"({"filters": 256, "shortcut": true, "bn_epsilon": 0.001, "bn_momentum": 0.03})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_sppf_node(one_input(input), attributes, K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 256}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 2u);
    ASSERT_EQ(list.updates().size(), 4u);
    expect_norm_options(graph, list, K_YOLO_NORM);
}

/**
 * @brief Tests that the factory reads "kernel" and "pools": one 3x3 pool padded by one, and a
 *        cv2 that reads two parts.
 */
TEST(Unit_BlockSppf, TheFactoryReadsKernelAndPools)
{
    const Shape input{4, 4, 64};
    const Config::Value attributes =
        attributes_of(R"({"filters": 64, "kernel": 3, "pools": 1})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NN::make_sppf_node(one_input(input), attributes, K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{2, 4, 4, 64}, "x");
    std::vector<Graph::ValueId> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(count_nodes(graph, Ops::K_MAXPOOL2D.id), 1u);
    const Graph::Node& pool = graph.nodes()[first_node(graph, Ops::K_MAXPOOL2D.id)];

    // A pool stores kernel_h, kernel_w and pad_top as whole numbers in slots 0, 1 and 6.
    EXPECT_NEAR(pool.attributes.values[0], 3.0, K_EXACT_TOLERANCE);
    EXPECT_NEAR(pool.attributes.values[1], 3.0, K_EXACT_TOLERANCE);
    EXPECT_NEAR(pool.attributes.values[6], 1.0, K_EXACT_TOLERANCE);

    const std::vector<std::string> state = walk(*node, "");
    ASSERT_EQ(state.size(), 10u);
    EXPECT_EQ(state[5], record("cv2.conv.weight", Shape{64, 1, 1, 64}, StateKind::Parameter));
}
