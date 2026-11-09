/**************************************************************************/
/* test_expert_transformer_block.cpp                                      */
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
 * @file test_expert_transformer_block.cpp
 * @brief Tests the trace, the state, the row and the incremental form of
 *        ExpertTransformerBlock.
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
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/expert_trace.hpp>
#include <SushiAI/language/blocks/expert_transformer_block.hpp>
#include <SushiAI/language/blocks/moe_feed_forward.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/nn/host_table.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
using Language::Blocks::AttentionMask;
using Language::Blocks::AttentionPositions;
using Language::Blocks::ExpertTrace;
using Language::Blocks::ExpertTransformerBlock;
using Language::Blocks::MoEFeedForward;
using NN::Trace;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Holds the width of the sequence every case reads. */
    constexpr int64_t K_DIM = 8;

    /** @brief Holds the head count of every case. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Holds the hidden width of one expert. */
    constexpr int64_t K_HIDDEN = 16;

    /** @brief Holds the number of experts of every case. */
    constexpr int64_t K_EXPERTS = 4;

    /** @brief Returns a causal block of K_EXPERTS experts that keeps @p top_k of them. */
    ExpertTransformerBlock make_block(int64_t top_k = 0)
    {
        return ExpertTransformerBlock(K_HEADS,
                                      MoEFeedForward(K_DIM, K_HIDDEN, K_EXPERTS, top_k),
                                      AttentionMask::Causal, NN::K_LAYER_NORM_EPSILON);
    }

    /** @brief Counts the nodes of @p graph that carry @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        return static_cast<std::size_t>(
            std::count_if(graph.nodes().begin(), graph.nodes().end(),
                          [op](const Graph::Node& node) { return node.op == op; }));
    }

    /** @brief Builds the node of a "transformer_block" row over a [5, K_DIM] sample. */
    Result<std::unique_ptr<NN::Node>> make_row(const std::string& text)
    {
        const Shape input{5, K_DIM};
        return Language::Blocks::make_transformer_block_node(
            NN::ShapeSpan(&input, 1), require_value(Config::parse_json(text, "<test>")), "layer 3");
    }

    /** @brief Returns the shape of the parameter the state walk of @p node names @p name. */
    Shape parameter_shape(NN::Node& node, const std::string& name)
    {
        Shape found{};
        const auto on_parameter = [&](const std::string& walked, NN::Parameter& parameter)
        {
            found = walked == name ? parameter.shape() : found;
        };
        const auto on_buffer = [](const std::string&, NN::Buffer&)
        {

        };
        const NN::StateVisitor visitor(on_parameter, on_buffer);
        NN::walk_state(node, "row", visitor);
        return found;
    }

    /** @brief Holds one row whose node has the single port. */
    struct OnePortRow
    {
        std::string name; /**< The row's name. */
        std::string json; /**< The row's attributes. */
    };

    /** @brief Runs one row of a single port. */
    class Small_BlockExpertTransformerBlockOnePort : public ::testing::TestWithParam<OnePortRow>
    {

    };

    /** @brief Holds one row the factory refuses and two words of its message. */
    struct FactoryRow
    {
        std::string name;   /**< The row's name. */
        std::string json;   /**< The row's attributes. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A second word of the message, or empty. */
    };

    /** @brief Runs one refused row. */
    class Small_BlockExpertTransformerBlockFactory : public ::testing::TestWithParam<FactoryRow>
    {

    };

    /** @brief Names a row by its name field. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that a trace yields its three values and adds the mixture to the first sum. */
TEST(Small_BlockExpertTransformerBlock, ATraceAddsANormalisedAttentionThenANormalisedMixture)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    ExpertTransformerBlock block = make_block();

    const Trace trace = Trace::evaluation(builder);
    SA_ASSERT_OK_AND_ASSIGN(const ExpertTrace traced, block.trace_experts(trace, x));

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(traced.output).shape, (Shape{2, 5, K_DIM}));
    EXPECT_EQ(graph.value(traced.balance).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(traced.usage).shape, (Shape{1, K_EXPERTS}));
    const Graph::Node& last = graph.node(graph.value(traced.output).producer);
    ASSERT_EQ(last.op, Ops::K_ADD.id);
    const Graph::Node& first = graph.node(graph.value(last.inputs[0]).producer);
    ASSERT_EQ(first.op, Ops::K_ADD.id);
    EXPECT_EQ(first.inputs[0], x);
}

/** @brief Tests that the walk names the fourteen parameters under the four children. */
TEST(Small_BlockExpertTransformerBlock, TheWalkYieldsTheFourChildren)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    ExpertTransformerBlock block = make_block();
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
                  "h.ff.router.weight", "h.ff.router.bias", "h.ff.expand.weight",
                  "h.ff.expand.bias", "h.ff.project.weight", "h.ff.project.bias"}));
}

/** @brief Tests that an input of another width fails once, with the first child's message. */
TEST(Small_BlockExpertTransformerBlock, AnInputThatDoesNotFitFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID wide = builder.input(Shape{2, 5, K_DIM + 1}, "wide");
    ExpertTransformerBlock block = make_block();

    const Trace trace = Trace::evaluation(builder);
    EXPECT_EQ(block.forward(trace, wide), wide);

    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "LayerNorm(8)",
                      "[2, 5, 9]");
    EXPECT_EQ(count_nodes(graph, Ops::K_LAYER_NORM.id), 0u);
    EXPECT_FALSE(block.ff.router.weight.declared());
}

/** @brief Tests that a row with experts builds a node of the output, balance and usage ports. */
TEST(Small_BlockExpertTransformerBlock, TheTransformerRowWithExpertsHasThreePorts)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> routed,
                            make_row(R"({"heads": 2, "experts": 4, "top_k": 1})"));

    ASSERT_EQ(routed->ports().size(), 3u);
    EXPECT_EQ(routed->ports()[0].name, NN::K_SINGLE_PORT_NAME);
    EXPECT_EQ(routed->ports()[0].shape, (Shape{5, K_DIM}));
    EXPECT_EQ(routed->ports()[1].name, Language::Blocks::K_BALANCE_PORT);
    EXPECT_EQ(routed->ports()[2].name, Language::Blocks::K_USAGE_PORT);
    EXPECT_EQ(routed->ports()[2].shape, (Shape{K_EXPERTS}));
}

/** @brief Tests that a row without experts, or with none, builds a node of one port. */
TEST_P(Small_BlockExpertTransformerBlockOnePort, ARowWithoutExpertsHasOnePort)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node, make_row(GetParam().json));

    EXPECT_EQ(node->ports().size(), 1u);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockExpertTransformerBlockOnePort,
    ::testing::Values(OnePortRow{"NoExpertsKey", R"({"heads": 2})"},
                      OnePortRow{"ZeroExperts", R"({"heads": 2, "experts": 0})"}),
    row_name<OnePortRow>);

/** @brief Tests that the node of a row with experts writes three outputs of their shapes. */
TEST(Small_BlockExpertTransformerBlock, TheExpertRowWritesThreeOutputsOfTheirShapes)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> routed,
                            make_row(R"({"heads": 2, "experts": 4, "top_k": 1})"));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    const Trace trace = Trace::evaluation(builder);
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_OK(routed->forward(trace, NN::ValueSpan(&x, 1), outputs));

    ASSERT_EQ(outputs.size(), 3u);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{2, 5, K_DIM}));
    EXPECT_EQ(graph.value(outputs[1]).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(outputs[2]).shape, (Shape{1, K_EXPERTS}));
}

/** @brief Tests that a row leaving out "hidden" gives each expert four times the width. */
TEST(Small_BlockExpertTransformerBlock, AnExpertRowTakesFourTimesTheWidthAsHidden)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> routed,
                            make_row(R"({"heads": 2, "experts": 4, "top_k": 1})"));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(routed->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));

    const Shape expand = parameter_shape(*routed, "row.ff.expand.weight");

    EXPECT_EQ(expand, (Shape{K_EXPERTS * 4 * K_DIM, K_DIM}));
}

/** @brief Tests that the row refuses a top_k without experts, above them and negative experts. */
TEST_P(Small_BlockExpertTransformerBlockFactory, TheTransformerRowRefusesCountsThatDoNotFit)
{
    const FactoryRow& row = GetParam();

    SA_EXPECT_FAILURE(make_row(row.json), StatusCode::InvalidDescription, "layer 3", row.first,
                      row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockExpertTransformerBlockFactory,
    ::testing::Values(FactoryRow{"TopKWithoutExperts", R"({"heads": 2, "top_k": 1})",
                                 "'top_k'", "no 'experts'"},
                      FactoryRow{"TopKAboveExperts", R"({"heads": 2, "experts": 2, "top_k": 3})",
                                 "top_k=3", ""},
                      FactoryRow{"NegativeExperts", R"({"heads": 2, "experts": -1})",
                                 "'experts'", "zero or more"}),
    row_name<FactoryRow>);

/** @brief Tests that one position yields its output and usage and carries the two caches. */
TEST(Small_BlockExpertTransformerBlock, TheIncrementalFormTracesEachChildOverOnePosition)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    ExpertTransformerBlock block = make_block(1);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    SA_ASSERT_OK_AND_ASSIGN(const ExpertTrace traced,
                            block.trace_experts_incremental(trace, x, Shape{5, K_DIM}));

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(traced.output).shape, (Shape{1, 1, K_DIM}));
    EXPECT_EQ(graph.value(traced.usage).shape, (Shape{1, K_EXPERTS}));
    EXPECT_EQ(carried.values().size(), 2u);
}

/** @brief Tests that the node of an expert row writes three outputs over one position. */
TEST(Small_BlockExpertTransformerBlock, TheIncrementalRowWritesThreeOutputs)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            make_row(R"({"heads": 2, "experts": 4})"));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID slice = builder.input(Shape{1, 1, K_DIM}, "x");
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(builder, fed, carried);
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_OK(node->forward_incremental(trace, NN::ValueSpan(&slice, 1), outputs));

    ASSERT_EQ(outputs.size(), 3u);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{1, 1, K_DIM}));
    EXPECT_EQ(carried.values().size(), 2u);
}

/** @brief Tests that the block hands learned or rotary positions to its attention. */
TEST(Small_BlockExpertTransformerBlock, TheBlockPassesThePositionsToTheAttention)
{
    const ExpertTransformerBlock learned = make_block();
    const ExpertTransformerBlock turned(K_HEADS, MoEFeedForward(K_DIM, K_HIDDEN, K_EXPERTS, 0),
                                        AttentionMask::Causal, NN::K_LAYER_NORM_EPSILON,
                                        AttentionPositions::Rotary);

    EXPECT_EQ(learned.attn.positions, AttentionPositions::Learned);
    EXPECT_EQ(turned.attn.positions, AttentionPositions::Rotary);
}

/** @brief Tests that the one attention of a rotary expert row declares two host tables. */
TEST(Small_BlockExpertTransformerBlock, ARotaryExpertRowDeclaresTwoTables)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> routed,
        make_row(R"({"heads": 2, "experts": 4, "positions": "rotary"})"));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_OK(routed->forward(trace, NN::ValueSpan(&x, 1), outputs));

    EXPECT_EQ(tables.tables().size(), 2u);
}

/** @brief Tests that a rotary expert row over an odd head width is refused. */
TEST(Small_BlockExpertTransformerBlock, ARotaryExpertRowOverAnOddHeadWidthIsRefused)
{
    SA_EXPECT_FAILURE(make_row(R"({"heads": 8, "experts": 4, "positions": "rotary"})"),
                      StatusCode::InvalidDescription, "layer 3", "rotary positions",
                      "1 channels");
}
