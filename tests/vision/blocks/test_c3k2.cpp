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
 * @file test_c3k2.cpp
 * @brief Tests C3k2's trace, its state names against YOLO26n rows 2, 6 and 22, and its factories.
 * @author Mustafa Garip
 */

#include <algorithm>
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
    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 2 'c3k2'";

    /** @brief Holds one inner kind whose hidden width of one the factory refuses. */
    struct HiddenRow
    {
        std::string name;       /**< The row's name. */
        std::string attributes; /**< The row's attributes. */
    };

    /** @brief Runs one row whose hidden width of one is refused. */
    class Small_BlockC3k2Hidden : public ::testing::TestWithParam<HiddenRow>
    {

    };

    /** @brief Holds one row handed to the bottleneck_psa factory and what it answers. */
    struct PsaInnerRow
    {
        std::string name;   /**< The row's name. */
        std::string inner;  /**< The "inner" the row states. */
        bool accepted;      /**< Whether the factory builds the block. */
        std::string needle; /**< A part of the refusal's message, or empty when accepted. */
    };

    /** @brief Runs one row of the bottleneck_psa factory. */
    class Small_BlockC3k2PsaInner : public ::testing::TestWithParam<PsaInnerRow>
    {

    };

    /** @brief Names a row by its name field. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return info.param.name;
    }

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

    /** @brief Reports whether @p records holds @p wanted. */
    bool contains(const std::vector<std::string>& records, const std::string& wanted)
    {
        return std::find(records.begin(), records.end(), wanted) != records.end();
    }

    /** @brief Lists the offset each SLICE_CHANNELS node of @p graph stores, in emission order. */
    std::vector<double> slice_offsets(const Graph::Graph& graph)
    {
        std::vector<double> offsets;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_SLICE_CHANNELS.id)
                offsets.push_back(node.attributes.values[0]);
        }

        return offsets;
    }

    /** @brief Lists the shape of each input of @p node, in order. */
    std::vector<Shape> input_shapes(const Graph::Graph& graph, const Graph::Node& node)
    {
        std::vector<Shape> shapes;
        for (const Graph::ValueID input : node.inputs)
            shapes.push_back(graph.value(input).shape);

        return shapes;
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
 * @brief Tests that a training trace of row 2 yields the output width, requests a mean and a
 *        variance update for each norm of its state and hands them the row's options.
 */
TEST(Small_BlockC3k2, Row2KeepsTheShapeUpdatesEveryNormAndKeepsTheOptions)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 32}, "x");
    C3k2<Bottleneck> block = make_row2();
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 64}));
    EXPECT_EQ(list.updates().size(), 2 * count_norms(walk(block, "model.2")));
    expect_norm_options(graph, list, K_YOLO_NORM);
}

/** @brief Tests that the stem is cut into two halves of 16 channels, at channels 0 and 16. */
TEST(Small_BlockC3k2, TheStemIsCutIntoTwoHalvesAtChannelsZeroAndSixteen)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 32}, "x");
    C3k2<Bottleneck> block = make_row2();
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    // A slice stores its offset as a whole number, so the copy compares exactly.
    EXPECT_EQ(slice_offsets(graph), (std::vector<double>{0.0, 16.0}));
}

/** @brief Tests that the join takes both halves and the inner block's result, 16 channels each. */
TEST(Small_BlockC3k2, TheJoinTakesBothHalvesAndTheInnerBlockResult)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 32}, "x");
    C3k2<Bottleneck> block = make_row2();
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    ASSERT_EQ(count_nodes(graph, Ops::K_CONCAT_CHANNELS.id), 1u);
    const Graph::Node& join = graph.nodes()[first_node(graph, Ops::K_CONCAT_CHANNELS.id)];

    EXPECT_EQ(input_shapes(graph, join), std::vector<Shape>(3, Shape{2, 4, 4, 16}));
    EXPECT_EQ(graph.value(join.outputs[0]).shape, (Shape{2, 4, 4, 48}));
}

/** @brief Tests that two inner blocks are all joined and listed in the state. */
TEST(Small_BlockC3k2, TwoInnerBlocksAreBothJoinedAndBothListedInTheState)
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
    EXPECT_EQ(input_shapes(graph, join), std::vector<Shape>(4, Shape{2, 4, 4, 16}));

    const std::vector<std::string> state = walk(block, "b");
    EXPECT_TRUE(contains(state, record("b.cv2.conv.weight", Shape{64, 1, 1, 64},
                                       StateKind::Parameter)));
    EXPECT_TRUE(contains(state, record("b.m.1.cv1.conv.weight", Shape{8, 3, 3, 16},
                                       StateKind::Parameter)));
}

/**
 * @brief Tests that a training trace of row 6, whose inner block is a C3k, keeps the input's
 *        shape, requests a mean and a variance update for each norm and hands them the options.
 */
TEST(Small_BlockC3k2, Row6KeepsTheShapeUpdatesEveryNormAndKeepsTheOptions)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 128}, "x");
    C3k2<C3k> block = make_row6();
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 128}));
    EXPECT_EQ(list.updates().size(), 2 * count_norms(walk(block, "model.6")));
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
 * @brief Tests that make_c3k2_node builds row 22 from its file row, whose inner block is a
 *        Bottleneck and a PSABlock: the port's shape, the 45 manifest records and the row's
 *        norm options.
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

/** @brief Tests that make_c3k2_node builds row 2 from its file row: the port and the options. */
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
    expect_norm_options(graph, list, K_YOLO_NORM);
}

/**
 * @brief Tests that make_c3k2_node builds row 6 from its file row, whose inner block is a
 *        C3k of two bottlenecks that add their input: the port and the options.
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

    expect_norm_options(graph, list, K_YOLO_NORM);
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

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{2, 4, 4, 64}));
    EXPECT_EQ(count_nodes(graph, Ops::K_ADD.id), 1u);
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

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(input), unlisted, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 2 'c3k2'", "'c2psa'",
                      "bottleneck, c3k, bottleneck_psa");
}

/** @brief Tests that the factory refuses an inner block given as a number. */
TEST(Small_BlockC3k2, TheFactoryRefusesAnInnerBlockThatIsNotAString)
{
    const Shape input{4, 4, 32};
    const Config::Value numbered = attributes_of(R"({"filters": 64, "hidden": 16, "inner": 2})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(input), numbered, K_CONTEXT),
                      StatusCode::InvalidDescription, "'inner'", "string");
}

/** @brief Tests that the factory refuses a hidden width of one, which no inner block halves. */
TEST_P(Small_BlockC3k2Hidden, AHiddenWidthOfOneIsRefused)
{
    const Shape input{4, 4, 32};
    const Config::Value attributes = attributes_of(GetParam().attributes);

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "at least 2");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockC3k2Hidden,
    ::testing::Values(HiddenRow{"InnerBottleneck", R"({"filters": 64, "hidden": 1})"},
                      HiddenRow{"InnerC3k", R"({"filters": 64, "hidden": 1, "inner": "c3k"})"}),
    row_name<HiddenRow>);

/** @brief Tests that the factory refuses a key the row does not take. */
TEST(Small_BlockC3k2, TheFactoryRefusesAnUnknownKey)
{
    const Shape input{4, 4, 32};
    const Config::Value unknown = attributes_of(R"({"filters": 64, "hidden": 16, "heads": 2})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(input), unknown, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 2 'c3k2'", "'heads'");
}

/** @brief Tests that the factory refuses a row without its hidden width. */
TEST(Small_BlockC3k2, TheFactoryRefusesAnAbsentHiddenWidth)
{
    const Shape input{4, 4, 32};
    const Config::Value no_hidden = attributes_of(R"({"filters": 64})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(input), no_hidden, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "absent");
}

/** @brief Tests that the factory refuses an input that is not one rank-3 feature map. */
TEST(Small_BlockC3k2, TheFactoryRefusesAFlatInput)
{
    const Shape flat{512};
    const Config::Value fits = attributes_of(R"({"filters": 64, "hidden": 16})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_node(one_input(flat), fits, K_CONTEXT),
                      StatusCode::InvalidDescription, "[512]", "rank-3");
}

/** @brief Tests that the bottleneck factory refuses a row whose "inner" names the C3k. */
TEST(Small_BlockC3k2, TheBottleneckFactoryRefusesTheNameOfTheC3k)
{
    const Shape input{4, 4, 32};
    const Config::Value of_c3k = attributes_of(R"({"filters": 64, "hidden": 16, "inner": "c3k"})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_of_bottleneck(one_input(input), of_c3k, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 2 'c3k2'", "'inner' is 'c3k'",
                      "builds 'bottleneck'");
}

/** @brief Tests that the bottleneck factory refuses an "inner" that is not a string. */
TEST(Small_BlockC3k2, TheBottleneckFactoryRefusesAnInnerThatIsNotAString)
{
    const Shape input{4, 4, 32};
    const Config::Value numbered = attributes_of(R"({"filters": 64, "hidden": 16, "inner": 2})");

    SA_EXPECT_FAILURE(
        Vision::Blocks::make_c3k2_of_bottleneck(one_input(input), numbered, K_CONTEXT),
        StatusCode::InvalidDescription, "'inner'", "string");
}

/** @brief Tests that the bottleneck factory accepts a row that names its own block. */
TEST(Small_BlockC3k2, TheBottleneckFactoryAcceptsItsOwnName)
{
    const Shape input{4, 4, 32};
    const Config::Value of_bottleneck =
        attributes_of(R"({"filters": 64, "hidden": 16, "inner": "bottleneck"})");

    SA_EXPECT_OK(
        Vision::Blocks::make_c3k2_of_bottleneck(one_input(input), of_bottleneck, K_CONTEXT));
}

/** @brief Tests that the C3k factory refuses a row whose "inner" names the bottleneck. */
TEST(Small_BlockC3k2, TheC3kFactoryRefusesTheNameOfTheBottleneck)
{
    const Shape input{4, 4, 32};
    const Config::Value of_bottleneck =
        attributes_of(R"({"filters": 64, "hidden": 16, "inner": "bottleneck"})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c3k2_of_c3k(one_input(input), of_bottleneck, K_CONTEXT),
                      StatusCode::InvalidDescription, "'inner' is 'bottleneck'", "builds 'c3k'");
}

/** @brief Tests that the C3k factory accepts a row that names its own block. */
TEST(Small_BlockC3k2, TheC3kFactoryAcceptsItsOwnName)
{
    const Shape input{4, 4, 32};
    const Config::Value of_c3k = attributes_of(R"({"filters": 64, "hidden": 16, "inner": "c3k"})");

    SA_EXPECT_OK(Vision::Blocks::make_c3k2_of_c3k(one_input(input), of_c3k, K_CONTEXT));
}

/** @brief Tests that the bottleneck_psa factory refuses another block's name, accepts its own. */
TEST_P(Small_BlockC3k2PsaInner, TheBottleneckPSAFactoryAnswersARowByTheInnerItNames)
{
    const Shape input{4, 4, 32};
    const PsaInnerRow& row = GetParam();
    const Config::Value attributes = attributes_of(
        R"({"filters": 64, "hidden": 16, "inner": ")" + row.inner + R"("})");

    const auto made =
        Vision::Blocks::make_c3k2_of_bottleneck_psa(one_input(input), attributes, K_CONTEXT);

    EXPECT_EQ(made.status().is_ok(), row.accepted);
    EXPECT_NE(std::string(made.status().message()).find(row.needle), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockC3k2PsaInner,
    ::testing::Values(PsaInnerRow{"RefusesTheNameOfTheC3k", "c3k", false,
                                  "'inner' is 'c3k'"},
                      PsaInnerRow{"AcceptsItsOwnName", "bottleneck_psa", true, ""}),
    row_name<PsaInnerRow>);
