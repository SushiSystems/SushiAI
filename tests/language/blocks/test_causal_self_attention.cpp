/**************************************************************************/
/* test_causal_self_attention.cpp                                         */
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
 * @file test_causal_self_attention.cpp
 * @brief Tests causal self-attention: its trace, its mask and gradients on an engine, and a
 *        transformer block over it.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/nn/host_table.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>
#include "blocks_test_support.hpp"

using namespace SushiAI;
using Language::Blocks::AttentionMask;
using Language::Blocks::AttentionPositions;
using Language::Blocks::CausalSelfAttention;
using NN::Trace;
// SaTest stays reachable inside a TEST_F body; see tests/language/blocks/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::write_tensor;
using Language::Blocks::TransformerBlock;

namespace
{
    /** @brief Holds the width every case attends over. */
    constexpr int64_t K_DIM = 8;

    /** @brief Holds the head count every case splits K_DIM into. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Holds T, the rows of the caches of every incremental case. */
    constexpr int64_t K_WINDOW = 5;

    /** @brief Holds the pairs of a head of K_DIM / K_HEADS channels. */
    constexpr int64_t K_HEAD_PAIRS = 2;

    /** @brief Holds the positions of the sequence the numeric case reads. */
    constexpr int64_t K_POSITIONS = 3;

    /** @brief Holds the operations a value of the numeric case sums, an upper bound. */
    constexpr std::size_t K_REFERENCE_OPERATIONS = 512;

    /** @brief Returns the per-sample [T, dim] shape the block reads in its ordinary form. */
    Shape window_sample()
    {
        return Shape{K_WINDOW, K_DIM};
    }

    /** @brief Returns a causal block of K_HEADS heads over K_DIM with rotary positions. */
    CausalSelfAttention rotary_block()
    {
        return CausalSelfAttention(K_DIM, K_HEADS, AttentionMask::Causal,
                                   AttentionPositions::Rotary);
    }

    /** @brief Counts the nodes of @p graph that carry @p op. */
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

    /** @brief Traces @p block over @p x, expects @p x back, and returns what was recorded. */
    Status refused(CausalSelfAttention& block, const Graph::GraphBuilder& builder,
                   Graph::ValueID x)
    {
        const Trace trace = Trace::evaluation(builder);
        EXPECT_EQ(block.forward(trace, x), x);
        return trace.status();
    }

    /** @brief Returns the names of the state @p block yields under "attn", in walk order. */
    std::vector<std::string> walked_names(CausalSelfAttention& block)
    {
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
        NN::walk_state(block, "attn", visitor);
        return names;
    }

    /** @brief Returns @p count rows of @p rows mapped through the linear layer weight and bias. */
    std::vector<double> linear_on_host(const std::vector<double>& rows, std::size_t count,
                                       const std::vector<double>& weight,
                                       const std::vector<double>& bias)
    {
        const std::size_t out = bias.size();
        const std::size_t in = weight.size() / out;
        std::vector<double> result(count * out);
        for (std::size_t row = 0; row < count; ++row)
        {
            for (std::size_t column = 0; column < out; ++column)
            {
                double sum = bias[column];
                for (std::size_t i = 0; i < in; ++i)
                    sum += rows[row * in + i] * weight[column * in + i];

                result[row * out + column] = sum;
            }
        }

        return result;
    }

    /**
     * @brief Returns the output of an unmasked attention of K_HEADS heads over the K_POSITIONS rows
     *        of @p x, with the state write_state() gives: scores scaled by one over the square
     *        root of the head width, softmax over the positions, the values mixed, then proj.
     */
    std::vector<double> attention_on_host(const std::vector<double>& x)
    {
        const std::size_t dim = static_cast<std::size_t>(K_DIM);
        const std::size_t length = static_cast<std::size_t>(K_POSITIONS);
        const std::size_t head_dim = dim / static_cast<std::size_t>(K_HEADS);
        const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
        const std::vector<double> mixed = linear_on_host(
            x, length, SaTest::eighths(3 * K_DIM * K_DIM, 0), SaTest::eighths(3 * K_DIM, 1));

        std::vector<double> joined(length * dim);
        for (std::size_t head = 0; head < static_cast<std::size_t>(K_HEADS); ++head)
        {
            const std::size_t offset = head * head_dim;
            for (std::size_t row = 0; row < length; ++row)
            {
                std::vector<double> weights(length);
                double total = 0.0;
                for (std::size_t other = 0; other < length; ++other)
                {
                    double score = 0.0;
                    for (std::size_t i = 0; i < head_dim; ++i)
                        score += mixed[row * 3 * dim + offset + i]
                               * mixed[other * 3 * dim + dim + offset + i];

                    weights[other] = std::exp(score * scale);
                    total += weights[other];
                }

                for (std::size_t i = 0; i < head_dim; ++i)
                {
                    double sum = 0.0;
                    for (std::size_t other = 0; other < length; ++other)
                        sum += weights[other] / total
                             * mixed[other * 3 * dim + 2 * dim + offset + i];

                    joined[row * dim + offset + i] = sum;
                }
            }
        }

        return linear_on_host(joined, length, SaTest::eighths(K_DIM * K_DIM, 2),
                              SaTest::eighths(K_DIM, 3));
    }

    /** @brief Expects every element of @p computed to be within the bound of @p reference. */
    void expect_close(const std::vector<double>& computed, const std::vector<double>& reference)
    {
        ASSERT_EQ(computed.size(), reference.size());
        for (std::size_t i = 0; i < computed.size(); ++i)
        {
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<double>(
                computed[i], reference[i], 1.0, K_REFERENCE_OPERATIONS)))
                << "at element " << i;
        }
    }

    /** @brief Names a row by its name field. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return info.param.name;
    }

    /** @brief Holds an input the ordinary trace refuses and the words of its message. */
    struct InputRow
    {
        std::string name;   /**< The row's name. */
        Shape shape;        /**< The input's shape. */
        DType dtype;        /**< The input's element type. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A word of the message, or empty. */
        std::string third;  /**< A word of the message, or empty. */
    };

    /** @brief Runs one input the trace refuses. */
    class Small_BlockCausalSelfAttentionInput : public ::testing::TestWithParam<InputRow>
    {

    };

    /** @brief Holds a block description the trace refuses and the words of its message. */
    struct DescriptionRow
    {
        std::string name;   /**< The row's name. */
        int64_t dim;        /**< The width the block states. */
        int64_t heads;      /**< The head count the block states. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A word of the message, or empty. */
    };

    /** @brief Runs one description the trace refuses. */
    class Small_BlockCausalSelfAttentionDescription
        : public ::testing::TestWithParam<DescriptionRow>
    {

    };

    /** @brief Holds a factory row the factory refuses and four words of its message. */
    struct FactoryRow
    {
        std::string name;   /**< The row's name. */
        std::string json;   /**< The row's attributes. */
        Shape input;        /**< The input the factory reads. */
        std::string first;  /**< A word of the message, after the context. */
        std::string second; /**< A word of the message, or empty. */
        std::string third;  /**< A word of the message, or empty. */
        std::string fourth; /**< A word of the message, or empty. */
    };

    /** @brief Runs one factory row the factory refuses. */
    class Small_BlockCausalSelfAttentionFactory : public ::testing::TestWithParam<FactoryRow>
    {

    };

    /** @brief Runs one factory row that misreads the positions. */
    class Small_BlockCausalSelfAttentionPositions : public ::testing::TestWithParam<FactoryRow>
    {

    };

    /** @brief Holds an incremental trace the block refuses and the words of its message. */
    struct CacheRow
    {
        std::string name;   /**< The row's name. */
        AttentionMask mask; /**< The mask of the block. */
        Shape input;        /**< The shape of the one position. */
        bool carrying;      /**< Whether the trace is a carrying one. */
        Shape sample;       /**< The per-sample shape of the cache. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A word of the message. */
    };

    /** @brief Runs one incremental trace the block refuses. */
    class Small_BlockCausalSelfAttentionCache : public ::testing::TestWithParam<CacheRow>
    {

    };

    /** @brief Holds a rotary block the trace refuses and the words of its message. */
    struct RotaryRow
    {
        std::string name;   /**< The row's name. */
        int64_t dim;        /**< The width of the block. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A word of the message. */
    };

    /** @brief Runs one rotary block the trace refuses. */
    class Small_BlockCausalSelfAttentionRotary : public ::testing::TestWithParam<RotaryRow>
    {

    };
} // namespace

/** @brief Tests that a causal trace keeps the [N, T, dim] shape and records no failure. */
TEST(Small_BlockCausalSelfAttention, ACausalTraceKeepsTheSequenceShape)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_DIM}));
}

/**
 * @brief Tests that the scores of each head are scaled by one over the square root of the head
 *        width: 0.5 for a width of 8 and two heads.
 *
 * The block has no mask, so the host evaluator runs it. The reference is the attention worked on
 * the host from the same state and the same rows.
 */
TEST(Small_BlockCausalSelfAttention, TheScoresOfEachHeadAreScaledByTheRootOfItsWidth)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, K_POSITIONS, K_DIM}, "x");
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::None);
    const Graph::ValueID y = block.forward(Trace::evaluation(builder), x);
    const std::vector<double> rows = SaTest::eighths(K_POSITIONS * K_DIM, 4);
    Graph::HostEvaluator run(graph);
    SA_EXPECT_OK(run.write(x, rows));
    SA_EXPECT_OK(SaTest::write_state(run, graph, block));

    SA_EXPECT_OK(run.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> got, run.read(y));
    expect_close(got, attention_on_host(rows));
}

/** @brief Tests that a block without a mask adds no CAUSAL_MASK node. */
TEST(Small_BlockCausalSelfAttention, ATraceWithoutAMaskEmitsNone)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::None);

    const Trace trace = Trace::evaluation(builder);
    static_cast<void>(block.forward(trace, x));

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(count_nodes(graph, Ops::K_CAUSAL_MASK.id), 0u);
}

/** @brief Tests that qkv is [3 * dim, dim], proj [dim, dim], and the walk names the four. */
TEST(Small_BlockCausalSelfAttention, TheWalkYieldsTheTwoProjections)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
    static_cast<void>(block.forward(Trace::evaluation(builder),
                                    builder.input(Shape{2, 5, K_DIM}, "x")));

    EXPECT_EQ(walked_names(block),
              (std::vector<std::string>{"attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
                                        "attn.proj.bias"}));
    EXPECT_EQ(block.qkv.weight.shape(), (Shape{3 * K_DIM, K_DIM}));
    EXPECT_EQ(block.proj.weight.shape(), (Shape{K_DIM, K_DIM}));
}

/** @brief Tests that a rank-2 input, another width and an integer input fail and emit nothing. */
TEST_P(Small_BlockCausalSelfAttentionInput, AnInputThatDoesNotFitFails)
{
    const InputRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(row.shape, "x", row.dtype);
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(refused(block, builder, x), StatusCode::InvalidDescription, row.first,
                      row.second, row.third);

    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_FALSE(block.qkv.weight.declared());
    EXPECT_TRUE(graph.has_refusal());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockCausalSelfAttentionInput,
    ::testing::Values(InputRow{"RankTwo", Shape{5, K_DIM}, DType::Float32, "CausalSelfAttention",
                               "rank-3 [N, T, 8] input", "[5, 8]"},
                      InputRow{"AnotherWidth", Shape{2, 5, K_DIM + 2}, DType::Float32,
                               "CausalSelfAttention", "[2, 5, 10]", ""},
                      InputRow{"Int32", Shape{2, 5, K_DIM}, DType::Int32,
                               "half, float32 or float64 input", "int32", ""}),
    row_name<InputRow>);

/** @brief Tests that a head count that does not divide the width, and no width, each fail. */
TEST_P(Small_BlockCausalSelfAttentionDescription, ABlockDescribedWrongFails)
{
    const DescriptionRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    CausalSelfAttention block(row.dim, row.heads, AttentionMask::Causal);

    SA_EXPECT_FAILURE(refused(block, builder, x), StatusCode::InvalidDescription, row.first,
                      row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockCausalSelfAttentionDescription,
    ::testing::Values(DescriptionRow{"UnevenHeads", K_DIM, 3, "head count divides",
                                     "dim=8 heads=3"},
                      DescriptionRow{"EmptyBlock", 0, 0, "dim=0 heads=0", ""}),
    row_name<DescriptionRow>);

/** @brief Tests that the factory takes dim from the input and reads "heads" and "causal". */
TEST(Small_BlockCausalSelfAttention, TheFactoryReadsTheHeadsAndTheMask)
{
    const Shape input{5, K_DIM};
    const Config::Value plain =
        SaTest::require_value(Config::parse_json(R"({"heads": 4})", "<test>"));
    const Config::Value open =
        SaTest::require_value(Config::parse_json(R"({"heads": 2, "causal": false})", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), plain,
                                                          "layer 0"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> encoder,
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), open,
                                                          "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<CausalSelfAttention>*>(node.get());
    const auto* wrapped_open =
        dynamic_cast<const NN::ModuleNode<CausalSelfAttention>*>(encoder.get());
    ASSERT_NE(wrapped, nullptr);
    ASSERT_NE(wrapped_open, nullptr);
    EXPECT_EQ(wrapped->layer().dim, K_DIM);
    EXPECT_EQ(wrapped->layer().heads, 4);
    EXPECT_EQ(wrapped->layer().mask, AttentionMask::Causal);
    EXPECT_EQ(wrapped_open->layer().mask, AttentionMask::None);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, K_DIM}));
}

/** @brief Tests that the factory refuses a stray key, no heads, uneven heads and a bad input. */
TEST_P(Small_BlockCausalSelfAttentionFactory, TheFactoryRefusesARowThatDoesNotFit)
{
    const FactoryRow& row = GetParam();
    const Config::Value attributes =
        SaTest::require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&row.input, 1),
                                                          attributes, "layer 3"),
        StatusCode::InvalidDescription, "layer 3", row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockCausalSelfAttentionFactory,
    ::testing::Values(FactoryRow{"StrayKey", R"({"heads": 2, "dropout": 0.1})", Shape{5, K_DIM},
                                 "'dropout'", "", "", ""},
                      FactoryRow{"NoHeads", "{}", Shape{5, K_DIM}, "'heads'", "absent", "", ""},
                      FactoryRow{"UnevenHeads", R"({"heads": 3})", Shape{5, K_DIM},
                                 "dim=8 heads=3", "", "", ""},
                      FactoryRow{"RankThreeInput", R"({"heads": 2})", Shape{4, 4, K_DIM},
                                 "rank-2 [T, C] input", "[4, 4, 8]", "", ""}),
    row_name<FactoryRow>);

/** @brief Tests that one position keeps the output shape and records no failure. */
TEST(Small_BlockCausalSelfAttention, TheIncrementalFormWritesOneRowPerCacheAndReadsEveryRow)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y = block.forward_incremental(trace, x, window_sample());

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_DIM}));
}

/** @brief Tests that the block declares the position, the mask and two carried [T, dim] leaves. */
TEST(Small_BlockCausalSelfAttention, TheIncrementalFormDeclaresTwoFedLeavesAndTwoCarriedValues)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    static_cast<void>(block.forward_incremental(trace, x, window_sample()));

    SA_EXPECT_OK(trace.status());
    ASSERT_EQ(fed.leaves().size(), 2u);
    EXPECT_EQ(fed.leaves()[0].name, Language::Blocks::K_POSITION_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[0].value).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(fed.leaves()[0].value).dtype, DType::Int32);
    EXPECT_EQ(fed.leaves()[1].name, Language::Blocks::K_ATTENTION_MASK_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[1].value).shape, (Shape{1, K_WINDOW}));

    ASSERT_EQ(carried.values().size(), 2u);
    for (const NN::CarriedValue& kept : carried.values())
    {
        EXPECT_EQ(graph.value(kept.leaf).shape, (Shape{K_WINDOW, K_DIM}));
        EXPECT_EQ(graph.value(kept.next).shape, (Shape{K_WINDOW, K_DIM}));
        EXPECT_EQ(graph.value(kept.leaf).lifetime, Graph::Lifetime::Input);
    }
}

/** @brief Tests that two blocks of one trace share the fed leaves and carry four values. */
TEST(Small_BlockCausalSelfAttention, TwoBlocksOfOneTraceShareTheirFedLeaves)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    CausalSelfAttention first(K_DIM, K_HEADS, AttentionMask::Causal);
    CausalSelfAttention second(K_DIM, K_HEADS, AttentionMask::Causal);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID between = first.forward_incremental(trace, x, window_sample());
    static_cast<void>(second.forward_incremental(trace, between, window_sample()));

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(fed.leaves().size(), 2u);
    EXPECT_EQ(carried.values().size(), 4u);
}

/** @brief Tests that the incremental form declares the state the ordinary form declares. */
TEST(Small_BlockCausalSelfAttention, TheIncrementalFormDeclaresTheParametersOfTheOrdinaryForm)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    CausalSelfAttention block(K_DIM, K_HEADS, AttentionMask::Causal);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    static_cast<void>(block.forward_incremental(trace, x, window_sample()));

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(NN::parameter_count(block), 4u);
    EXPECT_EQ(NN::buffer_count(block), 0u);
}

/** @brief Tests that an unmasked block, a longer slice, a plain trace and a flat sample fail. */
TEST_P(Small_BlockCausalSelfAttentionCache, TheIncrementalFormRefusesWhatItCannotCache)
{
    const CacheRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(row.input, "x");
    CausalSelfAttention block(K_DIM, K_HEADS, row.mask);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = row.carrying ? Trace::carrying(builder, fed, carried)
                                     : Trace::evaluation(builder);

    EXPECT_EQ(block.forward_incremental(trace, x, row.sample), x);

    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockCausalSelfAttentionCache,
    ::testing::Values(CacheRow{"NoMask", AttentionMask::None, Shape{1, 1, K_DIM}, true,
                               Shape{K_WINDOW, K_DIM}, "no mask", "no incremental form"},
                      CacheRow{"ThreePositions", AttentionMask::Causal, Shape{1, 3, K_DIM}, true,
                               Shape{K_WINDOW, K_DIM}, "one position", "[1, 3, 8]"},
                      CacheRow{"PlainTrace", AttentionMask::Causal, Shape{1, 1, K_DIM}, false,
                               Shape{K_WINDOW, K_DIM}, "'position'", "carrying trace"},
                      CacheRow{"FlatSample", AttentionMask::Causal, Shape{1, 1, K_DIM}, true,
                               Shape{K_WINDOW}, "per-sample", "[5]"}),
    row_name<CacheRow>);

/** @brief Tests that rotary positions keep the shape and fill one table of cosines and of sines. */
TEST(Small_BlockCausalSelfAttention, ARotaryTraceTurnsTheQueriesAndTheKeysOnce)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, K_DIM}, "x");
    CausalSelfAttention block = rotary_block();

    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    const Graph::ValueID y = block.forward(trace, x);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_DIM}));

    // One table of cosines and one of sines, a row per position and a column per pair.
    ASSERT_EQ(tables.tables().size(), 2u);
    for (const NN::HostTable& table : tables.tables())
    {
        EXPECT_EQ(graph.value(table.leaf).shape, (Shape{5, K_HEAD_PAIRS}));
        EXPECT_EQ(graph.value(table.leaf).lifetime, Graph::Lifetime::Buffer);
        EXPECT_FALSE(graph.value(table.leaf).requires_grad);
        EXPECT_EQ(table.values.size(), static_cast<std::size_t>(5 * K_HEAD_PAIRS));
    }
    EXPECT_DOUBLE_EQ(tables.tables()[0].values[0], 1.0);
    EXPECT_DOUBLE_EQ(tables.tables()[1].values[0], 0.0);
}

/** @brief Tests that the walk of a rotary block yields the two projections and no table. */
TEST(Small_BlockCausalSelfAttention, ARotaryBlockHoldsTheStateOfALearnedOne)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    CausalSelfAttention block = rotary_block();
    NN::HostTableList tables;
    Trace trace = Trace::evaluation(builder);
    trace.collect_tables(tables);
    static_cast<void>(block.forward(trace, builder.input(Shape{2, 5, K_DIM}, "x")));
    SA_EXPECT_OK(trace.status());

    EXPECT_EQ(walked_names(block),
              (std::vector<std::string>{"attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
                                        "attn.proj.bias"}));
}

/** @brief Tests that a trace no step fills tables for and an odd head are each refused. */
TEST_P(Small_BlockCausalSelfAttentionRotary, ARotaryTraceRefusesWhatItCannotTurn)
{
    const RotaryRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 5, row.dim}, "x");
    CausalSelfAttention block(row.dim, K_HEADS, AttentionMask::Causal,
                              AttentionPositions::Rotary);

    SA_EXPECT_FAILURE(refused(block, builder, x), StatusCode::InvalidDescription, row.first,
                      row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockCausalSelfAttentionRotary,
    ::testing::Values(RotaryRow{"NoTables", K_DIM, "'rotary_cosine'", "collect_tables"},
                      RotaryRow{"OddHeadWidth", 6, "rotary positions", "3 channels"}),
    row_name<RotaryRow>);

/** @brief Tests that the factory reads "positions", learned or rotary, into the layer. */
TEST(Small_BlockCausalSelfAttention, TheFactoryReadsThePositions)
{
    const Shape input{5, K_DIM};
    const Config::Value plain =
        SaTest::require_value(Config::parse_json(R"({"heads": 2})", "<test>"));
    const Config::Value rotary =
        SaTest::require_value(Config::parse_json(R"({"heads": 2, "positions": "rotary"})",
                                                 "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> learned,
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), plain,
                                                          "layer 0"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> turned,
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&input, 1), rotary,
                                                          "layer 0"));

    const auto* wrapped =
        dynamic_cast<const NN::ModuleNode<CausalSelfAttention>*>(learned.get());
    const auto* wrapped_turned =
        dynamic_cast<const NN::ModuleNode<CausalSelfAttention>*>(turned.get());
    ASSERT_NE(wrapped, nullptr);
    ASSERT_NE(wrapped_turned, nullptr);
    EXPECT_EQ(wrapped->layer().positions, AttentionPositions::Learned);
    EXPECT_EQ(wrapped_turned->layer().positions, AttentionPositions::Rotary);
}

/** @brief Tests that the factory refuses an unknown word, a non-string and an odd head width. */
TEST_P(Small_BlockCausalSelfAttentionPositions, TheFactoryRefusesWhatItCannotRead)
{
    const FactoryRow& row = GetParam();
    const Config::Value attributes =
        SaTest::require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_causal_self_attention_node(NN::ShapeSpan(&row.input, 1),
                                                          attributes, "layer 3"),
        StatusCode::InvalidDescription, "layer 3", row.first, row.second, row.third,
        row.fourth);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockCausalSelfAttentionPositions,
    ::testing::Values(FactoryRow{"UnknownWord", R"({"heads": 2, "positions": "sinusoidal"})",
                                 Shape{5, K_DIM}, "'positions'", "'sinusoidal'", "'learned'",
                                 "'rotary'"},
                      FactoryRow{"NotAString", R"({"heads": 2, "positions": 1})", Shape{5, K_DIM},
                                 "'positions'", "must be a string", "", ""},
                      FactoryRow{"OddHeadWidth", R"({"heads": 2, "positions": "rotary"})",
                                 Shape{5, 6}, "rotary positions", "3 channels", "", ""}),
    row_name<FactoryRow>);

/** @brief Tests that one rotary position keeps the output shape and carries two windows. */
TEST(Small_BlockCausalSelfAttention, TheRotaryIncrementalFormReplacesARowOfEachRing)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    CausalSelfAttention block = rotary_block();
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y = block.forward_incremental(trace, x, window_sample());

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_DIM}));
    EXPECT_EQ(carried.values().size(), 2u);
}

/** @brief Tests that a rotary position is fed its slot, the mask and its angles, no position. */
TEST(Small_BlockCausalSelfAttention, TheRotaryIncrementalFormIsFedASlotAndTheAngles)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    CausalSelfAttention block = rotary_block();
    CausalSelfAttention second = rotary_block();
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y = block.forward_incremental(trace, x, window_sample());
    static_cast<void>(second.forward_incremental(trace, y, window_sample()));

    SA_EXPECT_OK(trace.status());
    ASSERT_EQ(fed.leaves().size(), 4u);
    EXPECT_EQ(fed.leaves()[0].name, Language::Blocks::K_CACHE_SLOT_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[0].value).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(fed.leaves()[0].value).dtype, DType::Int32);
    EXPECT_EQ(fed.leaves()[1].name, Language::Blocks::K_ATTENTION_MASK_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[1].value).shape, (Shape{1, K_WINDOW}));
    EXPECT_EQ(fed.leaves()[2].name, Language::Blocks::K_ROTARY_COSINE_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[2].value).shape, (Shape{1, K_HEAD_PAIRS}));
    EXPECT_EQ(fed.leaves()[3].name, Language::Blocks::K_ROTARY_SINE_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[3].value).shape, (Shape{1, K_HEAD_PAIRS}));
    EXPECT_EQ(carried.values().size(), 4u);
}

namespace
{
    /** @brief Provides the Engine every gradient case records on. */
    class Medium_AttentionGradient : public SaTest::EngineTest
    {

    };

    /**
     * @brief Holds the absolute Bounds the distance of a FLOAT64 gradient from its closed form on
     *        the host: 16384 times the double unit roundoff, the smallest power of two not tighter
     *        than the earlier 1e-12.
     */
    constexpr double K_CLOSED_FORM_TOLERANCE = 16384.0 * TestSupport::unit_roundoff<double>();

    /** @brief Holds T, the number of positions of the sequence the block cases read. */
    constexpr int64_t K_GRADIENT_LENGTH = 4;

    /** @brief Holds the width of the sequence the block cases read. */
    constexpr int64_t K_GRADIENT_DIM = 4;

    /** @brief Returns @p count values of the deterministic host stream seeded with @p seed. */
    std::vector<double> noise(std::size_t count, std::uint64_t seed)
    {
        std::vector<double> values(count);
        SaTest::HostNoise(seed).fill(values, 1.0);
        return values;
    }

    /** @brief Names the leaf a probe differentiates at and the [N, C] value its loss reads. */
    struct Subject
    {
        ValueID variable = Graph::K_NO_VALUE; /**< The leaf the gradient is taken at. */
        ValueID logits = Graph::K_NO_VALUE;   /**< The [N, C] value the cross-entropy reads. */
    };

    /**
     * @brief Holds a FLOAT64 graph whose loss and gradient can both be evaluated at a point.
     *
     * The loss is the cross-entropy of the logits the build function names, against targets
     * whose row r names class r mod C. Every other leaf is filled from the host stream.
     */
    class Probe
    {
        public:
            /** @brief Builds the graph, differentiates it at its variable and compiles it. */
            template<typename BuildFn>
            Probe(SushiAI::ExecutionBackend& backend, std::uint64_t seed, BuildFn&& build)
                : pool_(backend.engine())
            {
                const GraphBuilder builder(graph_);
                const Subject subject = build(builder);
                variable_ = subject.variable;

                const Shape logit_shape = graph_.value(subject.logits).shape;
                targets_ = builder.input(logit_shape, "targets", DType::Float64);
                // A constructor cannot ASSERT; a Subject's logits are a well-formed [N, C].
                loss_ = builder.nll_loss(builder.log_softmax(subject.logits), targets_).value();

                const std::vector<ValueID> wrt = {variable_};
                // The graph is fixed here, so a refusal is a defect: value() stops the test.
                gradient_ = Autograd::differentiate(
                    graph_, loss_,
                    SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).value().at(0);
                EXPECT_NE(gradient_, Graph::K_NO_VALUE);

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                fill_other_leaves(seed, logit_shape);

                Result<std::unique_ptr<CompiledStep>> compiled =
                    CompiledStep::compile(backend, graph_, pool_);
                SA_EXPECT_OK(compiled);
                if (compiled.has_value())
                    step_ = std::move(compiled).value();
            }

            /** @brief Returns the graph the probe built, with its gradient nodes. */
            const Graph::Graph& graph() const noexcept
            {
                return graph_;
            }

            /** @brief Returns how many elements the variable holds. */
            std::size_t variable_size() const
            {
                return static_cast<std::size_t>(graph_.value(variable_).shape.num_elements());
            }

            /** @brief Returns the loss at @p point, or zero when the step did not compile. */
            double value(SushiRuntime::span<const double> point)
            {
                if (step_ == nullptr)
                    return 0.0;

                write_tensor(pool_.tensor(variable_), point);
                SA_EXPECT_OK(step_->run());
                return read_tensor(pool_.tensor(loss_))[0];
            }

            /** @brief Writes the analytic gradient at @p point into @p out. */
            void gradient(SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
            {
                if (step_ == nullptr)
                    return;

                write_tensor(pool_.tensor(variable_), point);
                SA_EXPECT_OK(step_->run());

                const std::vector<double> values = read_tensor(pool_.tensor(gradient_));
                ASSERT_EQ(values.size(), out.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                    out[i] = values[i];
            }

        private:
            /** @brief Writes the one-hot targets, and noise into each leaf but the variable. */
            void fill_other_leaves(std::uint64_t seed, const Shape& logit_shape)
            {
                std::vector<double> one_hot(
                    static_cast<std::size_t>(logit_shape.num_elements()), 0.0);
                for (int64_t row = 0; row < logit_shape[0]; ++row)
                    one_hot[static_cast<std::size_t>(row * logit_shape[1]
                                                     + row % logit_shape[1])] = 1.0;
                write_tensor(pool_.tensor(targets_), one_hot);

                for (ValueID id = 0; id < graph_.value_count(); ++id)
                {
                    const Graph::Lifetime lifetime = graph_.value(id).lifetime;
                    const bool is_leaf = lifetime == Graph::Lifetime::Parameter
                                      || lifetime == Graph::Lifetime::Input;
                    if (!is_leaf || id == variable_ || id == targets_)
                        continue;

                    const std::size_t count =
                        static_cast<std::size_t>(graph_.value(id).shape.num_elements());
                    write_tensor(pool_.tensor(id), noise(count, seed + id));
                }
            }

            Graph::Graph graph_;
            TensorPool pool_;
            std::unique_ptr<CompiledStep> step_;
            ValueID variable_ = Graph::K_NO_VALUE;
            ValueID targets_ = Graph::K_NO_VALUE;
            ValueID loss_ = Graph::K_NO_VALUE;
            ValueID gradient_ = Graph::K_NO_VALUE;
    };

    /** @brief Runs the full-Jacobian check of @p probe at a point drawn from @p seed. */
    void expect_gradient_matches(Probe& probe, std::uint64_t seed)
    {
        std::vector<double> point = noise(probe.variable_size(), seed);
        const Autograd::GradCheckResult result = Autograd::check_full_jacobian(
            [&probe](SushiRuntime::span<const double> at)
            {
                return probe.value(at);
            },
            [&probe](SushiRuntime::span<const double> at, SushiRuntime::span<double> out)
            {
                probe.gradient(at, out);
            },
            point);

        EXPECT_TRUE(result.passed) << result.to_string();
    }

    /** @brief Names which leaf of an attention probe the gradient is taken at. */
    enum class AttentionVariable : std::uint8_t
    {
        Input,            /**< The [1, K_GRADIENT_LENGTH, K_GRADIENT_DIM] sequence the block reads. */
        ProjectionWeight, /**< The weight of qkv, which every score and every value reads. */
    };

    /**
     * @brief Traces @p block over a FLOAT64 [1, K_GRADIENT_LENGTH, K_GRADIENT_DIM] sequence and names its
     *        [K_GRADIENT_LENGTH, K_GRADIENT_DIM] output rows as the logits.
     */
    Subject attention_subject(const GraphBuilder& builder, CausalSelfAttention& block,
                              AttentionVariable variable)
    {
        const Shape sequence{1, K_GRADIENT_LENGTH, K_GRADIENT_DIM};
        const ValueID x = variable == AttentionVariable::Input
            ? builder.parameter(sequence, "x", DType::Float64)
            : builder.input(sequence, "x", DType::Float64);

        const NN::Trace trace = NN::Trace::evaluation(builder);
        const ValueID y = block.forward(trace, x);
        SA_EXPECT_OK(trace.status());
        EXPECT_FALSE(builder.graph().has_refusal());

        const ValueID at = variable == AttentionVariable::Input
            ? x
            : block.qkv.weight.id_in(builder.graph());
        // The trace succeeded above, so y is the [1, T, dim] the reshape reads.
        return Subject{at,
                       builder.reshape(y, Shape{K_GRADIENT_LENGTH, K_GRADIENT_DIM}).value()};
    }

    /** @brief Checks the gradient of a block with @p mask at @p variable. */
    void expect_attention_gradient_matches(SushiAI::ExecutionBackend& backend, AttentionMask mask,
                                           AttentionVariable variable, std::uint64_t seed)
    {
        CausalSelfAttention block(K_GRADIENT_DIM, K_HEADS, mask);
        Probe probe(backend, seed, [&block, variable](const GraphBuilder& builder)
        {
            return attention_subject(builder, block, variable);
        });

        ASSERT_NO_FATAL_FAILURE(expect_gradient_matches(probe, seed + 1000));
    }
}

/**
 * @brief Tests the unfused rule against its closed form, with targets that are not one-hot.
 *
 * A SCALE by one sits between LOG_SOFTMAX and NLL_LOSS, so the fused rule does not apply and
 * the gradient is (softmax(x) * rowsum(Y) - Y) / N, which the host computes.
 */
TEST_F(Medium_AttentionGradient, AnUnfusedLogSoftmaxMatchesItsClosedForm)
{
    constexpr std::size_t K_ROWS = 3;
    constexpr std::size_t K_CLASSES = 4;
    const Shape shape{static_cast<int64_t>(K_ROWS), static_cast<int64_t>(K_CLASSES)};

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.parameter(shape, "x", DType::Float64);
    const ValueID targets = builder.input(shape, "targets", DType::Float64);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID detoured,
                            builder.scale(builder.log_softmax(x), 1.0));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss, builder.nll_loss(detoured, targets));

    const std::vector<ValueID> wrt = {x};
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<ValueID> gradients, Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())));
    ASSERT_NE(gradients.at(0), Graph::K_NO_VALUE);

    // The weights are positive and their rows do not sum to one.
    const std::vector<double> values = noise(K_ROWS * K_CLASSES, 7101);
    std::vector<double> weights = noise(K_ROWS * K_CLASSES, 7102);
    for (double& weight : weights)
        weight = std::abs(weight) + 0.25;

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), values);
    write_tensor(pool.tensor(targets), weights);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(backend(), graph, pool));
    SA_EXPECT_OK(step->run());
    SA_EXPECT_OK(step->run());

    const std::vector<double> got = read_tensor(pool.tensor(gradients[0]));
    ASSERT_EQ(got.size(), K_ROWS * K_CLASSES);
    for (std::size_t row = 0; row < K_ROWS; ++row)
    {
        double total = 0.0;
        double weight_sum = 0.0;
        for (std::size_t column = 0; column < K_CLASSES; ++column)
        {
            total += std::exp(values[row * K_CLASSES + column]);
            weight_sum += weights[row * K_CLASSES + column];
        }

        for (std::size_t column = 0; column < K_CLASSES; ++column)
        {
            const std::size_t i = row * K_CLASSES + column;
            const double expected = (std::exp(values[i]) / total * weight_sum - weights[i])
                                  / static_cast<double>(K_ROWS);
            EXPECT_NEAR(got[i], expected, K_CLOSED_FORM_TOLERANCE)
                << "at row " << row << " column " << column;
        }
    }
}

/** @brief Tests the gradient of a [3, 4] softmax that a matrix product reads. */
TEST_F(Medium_AttentionGradient, ASoftmaxGradientMatchesCentralDifferences)
{
    Probe probe(backend(), 7200, [](const GraphBuilder& builder)
    {
        const ValueID x = builder.parameter(Shape{3, 4}, "x", DType::Float64);
        const ValueID mix = builder.input(Shape{4, 3}, "mix", DType::Float64);
        // A lambda cannot ASSERT; a [3, 4] softmax times a [4, 3] matrix is well formed.
        return Subject{x, builder.matmul(builder.softmax(x), mix).value()};
    });

    ASSERT_NO_FATAL_FAILURE(expect_gradient_matches(probe, 7201));
}

/** @brief Tests the gradient of a softmax over the last axis of a [2, 3, 4] value. */
TEST_F(Medium_AttentionGradient, ASoftmaxOverRankThreeDifferentiatesThroughItsReshapes)
{
    Probe probe(backend(), 7300, [](const GraphBuilder& builder)
    {
        const ValueID x = builder.parameter(Shape{2, 3, 4}, "x", DType::Float64);
        // A lambda cannot ASSERT; the [2, 3, 4] softmax holds the 24 elements of [6, 4].
        return Subject{x, builder.reshape(builder.softmax(x), Shape{6, 4}).value()};
    });

    ASSERT_NO_FATAL_FAILURE(expect_gradient_matches(probe, 7301));
}

/** @brief Tests the gradient of a causal attention block at the sequence it reads. */
TEST_F(Medium_AttentionGradient, ACausalAttentionDifferentiatesAtItsInput)
{
    ASSERT_NO_FATAL_FAILURE(expect_attention_gradient_matches(
        backend(), AttentionMask::Causal, AttentionVariable::Input, 7400));
}

/** @brief Tests the gradient of a causal attention block at the weight of its qkv. */
TEST_F(Medium_AttentionGradient, ACausalAttentionDifferentiatesAtItsProjectionWeight)
{
    ASSERT_NO_FATAL_FAILURE(expect_attention_gradient_matches(
        backend(), AttentionMask::Causal, AttentionVariable::ProjectionWeight, 7500));
}

/** @brief Tests the gradient of an attention block without the mask at the sequence it reads. */
TEST_F(Medium_AttentionGradient, AnUnmaskedAttentionDifferentiatesAtItsInput)
{
    ASSERT_NO_FATAL_FAILURE(expect_attention_gradient_matches(
        backend(), AttentionMask::None, AttentionVariable::Input, 7600));
}

namespace
{
    /** @brief Provides the Engine and the element type of the mask kernel cases. */
    class Medium_CausalAttentionMask : public SaTest::EngineTest,
                                       public ::testing::WithParamInterface<DType>
    {

    };

    /** @brief Names a mask row by its element type. */
    std::string dtype_row_name(const ::testing::TestParamInfo<DType>& info)
    {
        return info.param == DType::Float32 ? "Float32" : "Float64";
    }

    /** @brief Holds the value the mask cases write above the diagonal. */
    constexpr double K_FILL = -1.0e9;

    /** @brief Holds T, the number of positions of the sequence the block cases read. */
    constexpr std::size_t K_MASK_LENGTH = 5;

    /** @brief Holds the width of the sequence the block cases read. */
    constexpr std::size_t K_MASK_DIM = 8;

    /** @brief Holds the first position whose input the block cases change. */
    constexpr std::size_t K_CHANGED_POSITION = 3;

    /** @brief Returns @p scores, [matrices, length, length], with the fill above each diagonal. */
    std::vector<double> masked_on_host(std::vector<double> scores, std::size_t length,
                                       double fill)
    {
        for (std::size_t i = 0; i < scores.size(); ++i)
        {
            const std::size_t row = (i / length) % length;
            const std::size_t column = i % length;
            if (column > row)
                scores[i] = fill;
        }

        return scores;
    }

    /** @brief Runs the mask over a [2, 4, 4] value of @p dtype and compares with the host. */
    void expect_mask_matches_host(SushiAI::ExecutionBackend& backend, DType dtype)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID scores = builder.input(Shape{2, 4, 4}, "scores", dtype);
        const Graph::Traced masked = builder.causal_mask(scores, K_FILL);
        SA_EXPECT_OK(masked);
        ASSERT_TRUE(masked.has_value());

        // Dyadic values survive the FLOAT32 round trip, so the kept entries compare exactly.
        std::vector<double> values(2 * 4 * 4);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = 0.25 * static_cast<double>(i) - 3.0;

        TensorPool pool(backend.engine());
        SA_EXPECT_OK(pool.materialise_leaves(graph));
        write_tensor(pool.tensor(scores), values);

        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(backend, graph, pool));
        SA_EXPECT_OK(step->run());
        SA_EXPECT_OK(step->run());

        const std::vector<double> got = read_tensor(pool.tensor(masked.value()));
        const std::vector<double> expected = masked_on_host(values, 4, K_FILL);
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_EQ(got[i], expected[i]) << "at element " << i;
    }

    /** @brief Holds what two runs of one block gave: before and after a later input changed. */
    struct TwoRuns
    {
        std::vector<double> before; /**< The [K_MASK_LENGTH, K_MASK_DIM] output of the first input. */
        std::vector<double> after;  /**< The output once positions from 3 on were changed. */
    };

    /**
     * @brief Evaluates @p graph on the host with @p input at @p x and every parameter filled from
     *        the host stream, and returns the elements of @p y.
     */
    std::vector<double> evaluate_on_host(const Graph::Graph& graph, ValueID x, ValueID y,
                                         const std::vector<double>& input)
    {
        Graph::HostEvaluator evaluator(graph);
        SA_EXPECT_OK(evaluator.write(x, input));
        for (ValueID id = 0; id < graph.value_count(); ++id)
        {
            if (graph.value(id).lifetime != Graph::Lifetime::Parameter)
                continue;

            const std::size_t count =
                static_cast<std::size_t>(graph.value(id).shape.num_elements());
            SA_EXPECT_OK(evaluator.write(id, noise(count, 6100 + id)));
        }

        SA_EXPECT_OK(evaluator.run());
        const Result<std::vector<double>> read = evaluator.read(y);
        SA_EXPECT_OK(read);
        return read.has_value() ? read.value() : std::vector<double>();
    }

    /**
     * @brief Traces @p block over a [1, K_MASK_LENGTH, K_MASK_DIM] input and evaluates it on the
     *        host on an input and on one changed from position 3 on.
     */
    template<typename Block>
    void run_twice(Block& block, TwoRuns& runs)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);
        const ValueID x = builder.input(
            Shape{1, static_cast<int64_t>(K_MASK_LENGTH), static_cast<int64_t>(K_MASK_DIM)}, "x");
        const NN::Trace trace = NN::Trace::evaluation(builder);
        const ValueID y = block.forward(trace, x);
        SA_EXPECT_OK(trace.status());
        ASSERT_FALSE(graph.has_refusal());

        std::vector<double> input = noise(K_MASK_LENGTH * K_MASK_DIM, 6001);
        runs.before = evaluate_on_host(graph, x, y, input);

        const std::vector<double> later = noise(K_MASK_LENGTH * K_MASK_DIM, 6002);
        for (std::size_t i = K_CHANGED_POSITION * K_MASK_DIM; i < input.size(); ++i)
            input[i] = later[i];

        runs.after = evaluate_on_host(graph, x, y, input);
    }

    /** @brief Counts the elements of positions [first, last) at which the two runs differ. */
    std::size_t changed_elements(const TwoRuns& runs, std::size_t first, std::size_t last)
    {
        std::size_t changed = 0;
        for (std::size_t i = first * K_MASK_DIM; i < last * K_MASK_DIM; ++i)
        {
            if (runs.before[i] != runs.after[i])
                ++changed;
        }

        return changed;
    }

    /** @brief Expects the two runs to hold K_MASK_LENGTH * K_MASK_DIM finite elements each. */
    void expect_whole_and_finite(const TwoRuns& runs)
    {
        ASSERT_EQ(runs.before.size(), K_MASK_LENGTH * K_MASK_DIM);
        ASSERT_EQ(runs.after.size(), K_MASK_LENGTH * K_MASK_DIM);
        for (const double value : runs.before)
            ASSERT_TRUE(std::isfinite(value));
    }
}

/**
 * @brief Tests that the mask keeps the lower triangle and writes the fill above, in FLOAT32 and
 *        in FLOAT64, where the mask vectors are converted from FLOAT32.
 */
TEST_P(Medium_CausalAttentionMask, TheMaskReplacesTheEntriesAboveTheDiagonal)
{
    ASSERT_NO_FATAL_FAILURE(expect_mask_matches_host(backend(), GetParam()));
}

INSTANTIATE_TEST_SUITE_P(Dtypes, Medium_CausalAttentionMask,
                         ::testing::Values(DType::Float32, DType::Float64), dtype_row_name);

/**
 * @brief Tests that the gradient of the scores is the incoming gradient on and below the
 *        diagonal and zero above it.
 *
 * The loss is the cross-entropy of the masked [4, 4] rows, so the incoming gradient is
 * (softmax - target) / 4, which the host computes from the masked scores.
 */
TEST(Small_BlockCausalSelfAttention, TheMaskPassesItsGradientThroughTheKeptEntriesAlone)
{
    constexpr std::size_t K_SIDE = 4;

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID scores = builder.parameter(Shape{4, 4}, "scores", DType::Float64);
    const ValueID targets = builder.input(Shape{4, 4}, "targets", DType::Float64);
    SA_ASSERT_OK_AND_ASSIGN(
        const ValueID loss,
        builder.nll_loss(builder.log_softmax(builder.causal_mask(scores, K_FILL)), targets));

    const std::vector<ValueID> wrt = {scores};
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<ValueID> gradients, Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())));
    ASSERT_NE(gradients.at(0), Graph::K_NO_VALUE);

    // Row r targets class 0, which every row keeps.
    const std::vector<double> values = noise(K_SIDE * K_SIDE, 6201);
    std::vector<double> one_hot(K_SIDE * K_SIDE, 0.0);
    for (std::size_t row = 0; row < K_SIDE; ++row)
        one_hot[row * K_SIDE] = 1.0;

    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(scores, values));
    SA_EXPECT_OK(evaluator.write(targets, one_hot));
    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> got, evaluator.read(gradients[0]));
    ASSERT_EQ(got.size(), K_SIDE * K_SIDE);
    for (std::size_t row = 0; row < K_SIDE; ++row)
    {
        double total = 0.0;
        for (std::size_t column = 0; column <= row; ++column)
            total += std::exp(values[row * K_SIDE + column]);

        for (std::size_t column = 0; column < K_SIDE; ++column)
        {
            const std::size_t i = row * K_SIDE + column;
            const double kept = (std::exp(values[i]) / total - one_hot[i]) / 4.0;
            if (column <= row)
                EXPECT_NEAR(got[i], kept, K_CLOSED_FORM_TOLERANCE)
                    << "at row " << row << " column " << column;
            else
                EXPECT_EQ(got[i], 0.0) << "at row " << row << " column " << column;
        }
    }
}

/** @brief Tests that positions 0 to 2 of a masked attention keep their output exactly. */
TEST(Small_BlockCausalSelfAttention, AnEarlierPositionIgnoresALaterInput)
{
    CausalSelfAttention block(static_cast<int64_t>(K_MASK_DIM), K_HEADS, AttentionMask::Causal);
    TwoRuns runs;
    ASSERT_NO_FATAL_FAILURE(run_twice(block, runs));
    ASSERT_NO_FATAL_FAILURE(expect_whole_and_finite(runs));

    EXPECT_EQ(changed_elements(runs, 0, K_CHANGED_POSITION), 0u);
    EXPECT_GT(changed_elements(runs, K_CHANGED_POSITION, K_MASK_LENGTH), 0u)
        << "the changed positions gave the output they gave before";
}

/** @brief Tests that without the mask an earlier position does read the later input. */
TEST(Small_BlockCausalSelfAttention, WithoutTheMaskAnEarlierPositionReadsALaterInput)
{
    CausalSelfAttention block(static_cast<int64_t>(K_MASK_DIM), K_HEADS, AttentionMask::None);
    TwoRuns runs;
    ASSERT_NO_FATAL_FAILURE(run_twice(block, runs));
    ASSERT_NO_FATAL_FAILURE(expect_whole_and_finite(runs));

    EXPECT_GT(changed_elements(runs, 0, K_CHANGED_POSITION), 0u);
}

/** @brief Tests that a whole causal transformer block keeps the same property. */
TEST(Small_BlockCausalSelfAttention, ATransformerBlockKeepsEarlierPositions)
{
    TransformerBlock block(static_cast<int64_t>(K_MASK_DIM), K_HEADS, 4 * static_cast<int64_t>(K_MASK_DIM),
                           AttentionMask::Causal, NN::K_LAYER_NORM_EPSILON);
    TwoRuns runs;
    ASSERT_NO_FATAL_FAILURE(run_twice(block, runs));
    ASSERT_NO_FATAL_FAILURE(expect_whole_and_finite(runs));

    EXPECT_EQ(changed_elements(runs, 0, K_CHANGED_POSITION), 0u);
    EXPECT_GT(changed_elements(runs, K_CHANGED_POSITION, K_MASK_LENGTH), 0u);
}
