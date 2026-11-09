/**************************************************************************/
/* test_c2psa.cpp                                                         */
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
 * @file test_c2psa.cpp
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
#include <SushiAI/vision/blocks/c2psa.hpp>

using namespace SushiAI;
using NN::BufferUpdateList;
using NN::NormOptions;
using NN::Trace;
using Vision::Blocks::C2PSA;
using Vision::Blocks::default_attention_heads;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Holds the epsilon and the momentum MANIFEST.json records for YOLO26n. */
    constexpr NormOptions K_YOLO_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 10 'c2psa'";

    /** @brief Holds the width of each half of row 10's stem. */
    constexpr int64_t K_HIDDEN = 128;

    /** @brief Holds one head count that cannot split the hidden width of 128. */
    struct HeadRow
    {
        std::string name; /**< The row's name. */
        int64_t heads;    /**< The head count the row states. */
    };

    /** @brief Runs one head count the hidden width cannot split into. */
    class Small_BlockC2psaHeads : public ::testing::TestWithParam<HeadRow>
    {

    };

    /** @brief Holds one hidden width and the head count a row that states none gets. */
    struct DefaultHeadRow
    {
        std::string name; /**< The row's name. */
        int64_t hidden;   /**< The hidden width of the row. */
        int64_t heads;    /**< The head count the row gets. */
    };

    /** @brief Runs one hidden width of a row that states no head count. */
    class Small_BlockC2psaDefaultHeads : public ::testing::TestWithParam<DefaultHeadRow>
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

    /** @brief Lists the offset of each SLICE_CHANNELS node whose result holds @p channels. */
    std::vector<double> slice_offsets_yielding(const Graph::Graph& graph, int64_t channels)
    {
        std::vector<double> offsets;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == Ops::K_SLICE_CHANNELS.id
                && graph.value(node.outputs[0]).shape[3] == channels)
                offsets.push_back(node.attributes.values[0]);
        }

        return offsets;
    }

    /** @brief Lists the nodes of @p graph that carry @p op and yield @p shape. */
    std::vector<const Graph::Node*> nodes_yielding(const Graph::Graph& graph, OpID op,
                                                    const Shape& shape)
    {
        std::vector<const Graph::Node*> found;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op && graph.value(node.outputs[0]).shape == shape)
                found.push_back(&node);
        }

        return found;
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
TEST(Small_BlockC2psa, OutputShapeKeepsTheExtentsAndSetsTheWidth)
{
    const C2PSA block = make_row10();

    EXPECT_EQ(block.output_shape(Shape{20, 20, 256}).value(), (Shape{20, 20, 256}));
}

/**
 * @brief Tests that a training trace of row 10 keeps the input's shape, requests a mean and a
 *        variance update for each norm of its state and hands them the row's options.
 */
TEST(Small_BlockC2psa, Row10KeepsTheShapeUpdatesEveryNormAndKeepsTheOptions)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    C2PSA block = make_row10();
    BufferUpdateList list;

    const Graph::ValueID y = block.forward(Trace::training(builder, list), x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 256}));
    EXPECT_EQ(list.updates().size(), 2 * count_norms(walk(block, "model.10")));
    expect_norm_options(graph, list, K_YOLO_NORM);
}

/** @brief Tests that the stem is cut into two halves at channels 0 and 128. */
TEST(Small_BlockC2psa, TheStemIsCutIntoTwoHalvesAtChannelsZeroAndOneHundredTwentyEight)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    C2PSA block = make_row10();
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    // A slice stores its offset as a whole number, so the copy compares exactly.
    EXPECT_EQ(slice_offsets_yielding(graph, K_HIDDEN), (std::vector<double>{0.0, 128.0}));
}

/** @brief Tests that the join takes two halves of 128 channels and yields the 256 of the row. */
TEST(Small_BlockC2psa, TheJoinTakesTwoHalvesOf128ChannelsAndYieldsTheOutputWidth)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    C2PSA block = make_row10();
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    const std::vector<const Graph::Node*> joins =
        nodes_yielding(graph, Ops::K_CONCAT_CHANNELS.id, Shape{2, 4, 4, 256});
    ASSERT_EQ(joins.size(), 1u);

    ASSERT_EQ(joins.front()->inputs.size(), 2u);
    EXPECT_EQ(graph.value(joins.front()->inputs[0]).shape, (Shape{2, 4, 4, K_HIDDEN}));
    EXPECT_EQ(graph.value(joins.front()->inputs[1]).shape, (Shape{2, 4, 4, K_HIDDEN}));
}

/** @brief Tests that two repeats hold two blocks and keep the input's shape. */
TEST(Small_BlockC2psa, TwoRepeatsHoldTwoBlocksAndKeepTheInputShape)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    C2PSA block(256, 256, K_HIDDEN, 2, 2, K_YOLO_NORM);

    const Graph::ValueID y = block.forward(Trace::evaluation(builder), x);

    EXPECT_EQ(block.m.size(), 2u);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 256}));
}

/**
 * @brief Tests that the state of traced row 10 yields the 35 records MANIFEST.json lists under
 *        model.10, names, NHWC shapes and order.
 */
TEST(Small_BlockC2psa, StateMatchesManifestRow10)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    C2PSA block = make_row10();
    BufferUpdateList list;
    static_cast<void>(block.forward(Trace::training(builder, list), x));

    EXPECT_EQ(walk(block, "model.10"), row10_records());
}

/** @brief Tests that make_c2psa_node builds a node with one port of the input's shape. */
TEST(Small_BlockC2psa, TheFactoryBuildsOnePortOfTheInputShape)
{
    const Shape input{20, 20, 256};
    const Config::Value attributes = attributes_of(
        R"({"filters": 256, "hidden": 128, "repeats": 1, "heads": 2})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c2psa_node(one_input(input), attributes,
                                                            K_CONTEXT));

    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{20, 20, 256}));
}

/** @brief Tests that the factory hands the row's "bn_epsilon" and "bn_momentum" to every norm. */
TEST(Small_BlockC2psa, TheFactoryHandsTheNormOptionsToEveryNorm)
{
    const Shape input{20, 20, 256};
    const Config::Value attributes = attributes_of(
        R"({"filters": 256, "hidden": 128, "bn_epsilon": 0.001, "bn_momentum": 0.03})");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_c2psa_node(one_input(input), attributes,
                                                            K_CONTEXT));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 256}, "x");
    BufferUpdateList list;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::training(builder, list), NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    expect_norm_options(graph, list, K_YOLO_NORM);
}

/** @brief Tests that a row stating neither "repeats" nor "heads" holds one PSABlock. */
TEST(Small_BlockC2psa, TheFactoryDefaultsToOneBlock)
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

    // One block is the 35 records of row 10: the stem's two convolutions and one PSABlock.
    EXPECT_EQ(walk(*node, "").size(), 35u);
}

/** @brief Tests that a row that states no head count gets max(hidden / 64, 1) heads. */
TEST_P(Small_BlockC2psaDefaultHeads, ARowThatStatesNoHeadCountGetsTheHiddenWidthOver64OrOne)
{
    const DefaultHeadRow& row = GetParam();

    EXPECT_EQ(default_attention_heads(row.hidden), row.heads);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockC2psaDefaultHeads,
    ::testing::Values(DefaultHeadRow{"Hidden128GivesTwo", 128, 2},
                      DefaultHeadRow{"Hidden192GivesThree", 192, 3},
                      DefaultHeadRow{"Hidden127GivesOne", 127, 1},
                      DefaultHeadRow{"Hidden64GivesOne", 64, 1}),
    row_name<DefaultHeadRow>);

/** @brief Tests that a "filters" other than the input width is refused, naming both widths. */
TEST(Small_BlockC2psa, TheFactoryRefusesAFilterWidthThatIsNotTheInput)
{
    const Shape input{20, 20, 256};
    const Config::Value attributes = attributes_of(R"({"filters": 128, "hidden": 128})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c2psa_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "'filters'", "128", "256 channels");
}

/** @brief Tests that a hidden width the heads do not split into parts of two or more is refused. */
TEST_P(Small_BlockC2psaHeads, AHiddenWidthTheHeadsDoNotSplitIsRefused)
{
    const Shape input{20, 20, 256};
    const Config::Value attributes = attributes_of(
        R"({"filters": 256, "hidden": 128, "heads": )" + std::to_string(GetParam().heads) + "}");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c2psa_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "parts of at least 2 channels");
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_BlockC2psaHeads,
                         ::testing::Values(HeadRow{"HeadsThatDoNotDivide", 3},
                                           HeadRow{"OneChannelPerHead", 128}),
                         row_name<HeadRow>);

/** @brief Tests that the factory refuses a key the row does not take. */
TEST(Small_BlockC2psa, TheFactoryRefusesAnUnknownKey)
{
    const Shape input{20, 20, 256};
    const Config::Value unknown =
        attributes_of(R"({"filters": 256, "hidden": 128, "inner": "bottleneck"})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c2psa_node(one_input(input), unknown, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 10 'c2psa'", "'inner'");
}

/** @brief Tests that the factory refuses a row without its hidden width. */
TEST(Small_BlockC2psa, TheFactoryRefusesAnAbsentHiddenWidth)
{
    const Shape input{20, 20, 256};
    const Config::Value no_hidden = attributes_of(R"({"filters": 256})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c2psa_node(one_input(input), no_hidden, K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "absent");
}

/** @brief Tests that the factory refuses an input that is not one rank-3 feature map. */
TEST(Small_BlockC2psa, TheFactoryRefusesAFlatInput)
{
    const Shape flat{512};
    const Config::Value fits = attributes_of(R"({"filters": 256, "hidden": 128})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_c2psa_node(one_input(flat), fits, K_CONTEXT),
                      StatusCode::InvalidDescription, "[512]", "rank-3");
}
