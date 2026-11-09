/**************************************************************************/
/* test_transformer_block.cpp                                             */
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
 * @file test_transformer_block.cpp
 * @brief Tests the trace, the state, the incremental form and the node factory of
 *        TransformerBlock.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using Language::Blocks::AttentionMask;
using Language::Blocks::TransformerBlock;
using NN::Trace;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Holds the width of the sequence every case reads. */
    constexpr int64_t K_DIM = 8;

    /** @brief Holds the head count of every case. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Holds the width of the feed-forward of every case. */
    constexpr int64_t K_HIDDEN = 16;

    /** @brief Returns a block over the three widths above with the mask given. */
    TransformerBlock make_block(AttentionMask mask = AttentionMask::Causal)
    {
        return TransformerBlock(K_DIM, K_HEADS, K_HIDDEN, mask, NN::K_LAYER_NORM_EPSILON);
    }

    /** @brief Counts the nodes of @p graph that carry @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        return static_cast<std::size_t>(
            std::count_if(graph.nodes().begin(), graph.nodes().end(),
                          [op](const Graph::Node& node) { return node.op == op; }));
    }

    /** @brief Counts the LAYER_NORM nodes of @p graph that read @p a or @p b. */
    std::size_t count_norms_reading(const Graph::Graph& graph, Graph::ValueID a,
                                    Graph::ValueID b)
    {
        return static_cast<std::size_t>(std::count_if(
            graph.nodes().begin(), graph.nodes().end(), [a, b](const Graph::Node& node)
            {
                return node.op == Ops::K_LAYER_NORM.id
                    && (node.inputs[0] == a || node.inputs[0] == b);
            }));
    }

    /** @brief Holds one factory row the factory refuses and two words of its message. */
    struct FactoryRow
    {
        std::string name;   /**< The row's name. */
        std::string json;   /**< The row's attributes. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A second word of the message, or empty. */
    };

    /** @brief Runs one refused factory row. */
    class Small_BlockTransformerBlockFactory : public ::testing::TestWithParam<FactoryRow>
    {

    };

    /** @brief Names a factory row by its name field. */
    std::string row_name(const ::testing::TestParamInfo<FactoryRow>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that a trace keeps the shape and adds each normalised branch to what it read. */
TEST(Small_BlockTransformerBlock, ATraceAddsANormalisedAttentionThenANormalisedFeedForward)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    TransformerBlock block = make_block();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_DIM}));
    const Graph::Node& last = graph.node(graph.value(y).producer);
    ASSERT_EQ(last.op, Ops::K_ADD.id);
    const Graph::Node& first = graph.node(graph.value(last.inputs[0]).producer);
    ASSERT_EQ(first.op, Ops::K_ADD.id);
    EXPECT_EQ(first.inputs[0], x);
    EXPECT_EQ(count_norms_reading(graph, x, first.outputs[0]), 2u);
}

/** @brief Tests that the mask option reaches the attention: no CAUSAL_MASK without it. */
TEST(Small_BlockTransformerBlock, ABlockWithoutAMaskEmitsNone)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    TransformerBlock block = make_block(AttentionMask::None);

    const Trace trace = Trace::evaluation(builder);
    static_cast<void>(block.forward(trace, builder.input(Shape{2, 5, K_DIM}, "x")));

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(count_nodes(graph, Ops::K_CAUSAL_MASK.id), 0u);
}

/** @brief Tests that the walk names the twelve parameters under the four children, in order. */
TEST(Small_BlockTransformerBlock, TheWalkYieldsTheFourChildren)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    TransformerBlock block = make_block();
    static_cast<void>(block.forward(Trace::evaluation(builder),
                                    builder.input(Shape{2, 5, K_DIM}, "x")));

    std::vector<std::string> names;
    const auto on_parameter = [&names](const std::string& name, NN::Parameter&)
    {
        names.push_back(name);
    };
    const auto on_buffer = [&names](const std::string& name, NN::Buffer&)
    {
        names.push_back(name);
    };
    const NN::StateVisitor visitor(on_parameter, on_buffer);
    NN::walk_state(block, "h", visitor);

    EXPECT_EQ(names,
              (std::vector<std::string>{
                  "h.norm1.weight", "h.norm1.bias", "h.attn.qkv.weight", "h.attn.qkv.bias",
                  "h.attn.proj.weight", "h.attn.proj.bias", "h.norm2.weight", "h.norm2.bias",
                  "h.ff.expand.weight", "h.ff.expand.bias", "h.ff.project.weight",
                  "h.ff.project.bias"}));
}

/** @brief Tests that an input of another width fails once, with the first child's message. */
TEST(Small_BlockTransformerBlock, AnInputThatDoesNotFitFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID wide = builder.input(Shape{2, 5, K_DIM + 1}, "wide");
    TransformerBlock block = make_block();

    const Trace trace = Trace::evaluation(builder);
    EXPECT_EQ(block.forward(trace, wide), wide);

    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "LayerNorm(8)",
                      "[2, 5, 9]");
    EXPECT_EQ(count_nodes(graph, Ops::K_LAYER_NORM.id), 0u);
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that a row of heads alone takes the defaults of the other keys. */
TEST(Small_BlockTransformerBlock, TheFactoryTakesTheDefaultsOfAbsentKeys)
{
    const Shape input{5, K_DIM};
    const Config::Value plain = require_value(Config::parse_json(R"({"heads": 2})", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&input, 1), plain,
                                                      "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<TransformerBlock>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().attn.heads, 2);
    EXPECT_EQ(wrapped->layer().ff.hidden, 4 * K_DIM);
    EXPECT_EQ(wrapped->layer().attn.mask, AttentionMask::Causal);
    EXPECT_EQ(wrapped->layer().norm1.epsilon, NN::K_LAYER_NORM_EPSILON);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, K_DIM}));
}

/** @brief Tests that stated keys override the defaults. */
TEST(Small_BlockTransformerBlock, TheFactoryReadsTheStatedKeys)
{
    const Shape input{5, K_DIM};
    const Config::Value stated = require_value(Config::parse_json(
        R"({"heads": 4, "hidden": 16, "causal": false, "ln_epsilon": 0.001})", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&input, 1), stated,
                                                      "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<TransformerBlock>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().attn.heads, 4);
    EXPECT_EQ(wrapped->layer().ff.hidden, 16);
    EXPECT_EQ(wrapped->layer().attn.mask, AttentionMask::None);
    EXPECT_EQ(wrapped->layer().norm2.epsilon, 0.001);
}

/** @brief Tests that the factory refuses a stray key, no heads and heads that do not divide. */
TEST_P(Small_BlockTransformerBlockFactory, TheFactoryRefusesARowThatDoesNotFit)
{
    const FactoryRow& row = GetParam();
    const Shape input{5, K_DIM};
    const Config::Value attributes = require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&input, 1),
                                                                    attributes, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockTransformerBlockFactory,
    ::testing::Values(FactoryRow{"StrayKey", R"({"heads": 2, "depth": 3})", "'depth'", ""},
                      FactoryRow{"NoHeads", R"({"hidden": 16})", "'heads'", "absent"},
                      FactoryRow{"UnevenHeads", R"({"heads": 5})", "dim=8 heads=5", ""}),
    row_name);

/** @brief Tests that one position of a block is fed two leaves and carries the two caches. */
TEST(Small_BlockTransformerBlock, TheIncrementalFormTracesEachChildOverOnePosition)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    TransformerBlock block = make_block();
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y = block.forward_incremental(trace, x, Shape{5, K_DIM});

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_DIM}));
    EXPECT_EQ(fed.leaves().size(), 2u);
    EXPECT_EQ(carried.values().size(), 2u);
}

/** @brief Tests that the incremental form declares the parameters the ordinary form declares. */
TEST(Small_BlockTransformerBlock, BothFormsDeclareTheSameParameters)
{
    TransformerBlock whole = make_block();
    TransformerBlock sliced = make_block();

    Graph::Graph whole_graph;
    const Graph::GraphBuilder whole_builder(whole_graph);
    static_cast<void>(whole.forward(Trace::evaluation(whole_builder),
                                    whole_builder.input(Shape{1, 5, K_DIM}, "x")));

    Graph::Graph sliced_graph;
    const Graph::GraphBuilder sliced_builder(sliced_graph);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(sliced_builder, fed, carried);
    static_cast<void>(sliced.forward_incremental(
        trace, sliced_builder.input(Shape{1, 1, K_DIM}, "x"), Shape{5, K_DIM}));

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(NN::parameter_count(sliced), NN::parameter_count(whole));
    EXPECT_EQ(NN::buffer_count(sliced), 0u);
}

/** @brief Tests that a block without a mask fails the trace through its attention. */
TEST(Small_BlockTransformerBlock, AnUnmaskedBlockHasNoIncrementalForm)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    TransformerBlock block = make_block(AttentionMask::None);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(builder, fed, carried);

    EXPECT_EQ(block.forward_incremental(trace, x, Shape{5, K_DIM}), x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "no mask");
}

/** @brief Tests that the factory hands "positions" to the attention. */
TEST(Small_BlockTransformerBlock, TheFactoryPassesThePositionsToItsAttention)
{
    const Shape input{5, K_DIM};
    const Config::Value plain = require_value(Config::parse_json(R"({"heads": 2})", "<test>"));
    const Config::Value rotary =
        require_value(Config::parse_json(R"({"heads": 2, "positions": "rotary"})", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> learned,
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&input, 1), plain,
                                                      "layer 0"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> turned,
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&input, 1), rotary,
                                                      "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<TransformerBlock>*>(learned.get());
    const auto* wrapped_turned =
        dynamic_cast<const NN::ModuleNode<TransformerBlock>*>(turned.get());
    ASSERT_NE(wrapped, nullptr);
    ASSERT_NE(wrapped_turned, nullptr);
    EXPECT_EQ(wrapped->layer().attn.positions,
              Language::Blocks::AttentionPositions::Learned);
    EXPECT_EQ(wrapped_turned->layer().attn.positions,
              Language::Blocks::AttentionPositions::Rotary);
}

/** @brief Tests that rotary positions over an odd head width are refused. */
TEST(Small_BlockTransformerBlock, RotaryPositionsOverAnOddHeadWidthAreRefused)
{
    const Shape odd{5, 6};
    const Config::Value rotary =
        require_value(Config::parse_json(R"({"heads": 2, "positions": "rotary"})", "<test>"));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_transformer_block_node(NN::ShapeSpan(&odd, 1), rotary,
                                                      "layer 3"),
        StatusCode::InvalidDescription, "layer 3", "rotary positions", "3 channels");
}
