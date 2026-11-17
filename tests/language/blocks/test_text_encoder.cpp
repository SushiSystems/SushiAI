/**************************************************************************/
/* test_text_encoder.cpp                                                  */
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
 * @file test_text_encoder.cpp
 * @brief Tests TextEncoder: its node, its trace and refusals, and what padding does to its
 *        result on an engine.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/blocks/text_encoder.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/padded_text_comparison.hpp>
#include "blocks_test_support.hpp"

using namespace SushiAI;
using Language::Blocks::TextEncoder;
using NN::Trace;
// SaTest stays reachable inside a TEST_F body; see tests/language/blocks/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /** @brief Holds the row count of the token table of every encoder. */
    constexpr int64_t K_VOCABULARY = 11;

    /** @brief Holds the width of every encoder. */
    constexpr int64_t K_DIM = 8;

    /** @brief Holds the head count of every encoder. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Holds the hidden width of the feed-forward of every block. */
    constexpr int64_t K_HIDDEN = 16;

    /** @brief Holds how many blocks every encoder stacks. */
    constexpr int64_t K_BLOCKS = 2;

    /** @brief Holds the row count of the position table, the longest text it serves. */
    constexpr int64_t K_TABLE_LENGTH = 6;

    /** @brief Holds T of the padded texts the numeric cases read. */
    constexpr int64_t K_PADDED_POSITIONS = 5;

    /** @brief Holds the length of the short text the padded batch holds first. */
    constexpr int64_t K_SHORT_POSITIONS = 3;

    /**
     * @brief Holds an upper bound of the rounded operations on the longest path to one value of
     *        the result: two blocks of a normalisation, a projection, a score, a softmax and a
     *        feed-forward each, then the pool.
     */
    constexpr std::size_t K_PATH_OPERATIONS = 512;

    /** @brief Names a row by its name field. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return info.param.name;
    }

    /** @brief Returns the encoder every case builds. */
    TextEncoder make_encoder()
    {
        return TextEncoder(K_VOCABULARY, K_DIM, K_TABLE_LENGTH, K_HEADS, K_HIDDEN, K_BLOCKS,
                           NN::K_LAYER_NORM_EPSILON);
    }

    /** @brief Returns the tokens of the short text. */
    std::vector<double> short_tokens()
    {
        return {1, 4, 7};
    }

    /** @brief Returns the tokens of the text that fills its sequence. */
    std::vector<double> full_tokens()
    {
        return {3, 8, 1, 10, 6};
    }

    /** @brief Returns the short text followed by @p padding, then the full text. */
    std::vector<double> padded_tokens(const std::vector<double>& padding)
    {
        return SaTest::padded_batch(short_tokens(), padding, full_tokens());
    }

    /** @brief Returns the lengths of the two texts of padded_tokens(). */
    std::vector<double> batch_lengths()
    {
        return {static_cast<double>(K_SHORT_POSITIONS), static_cast<double>(K_PADDED_POSITIONS)};
    }

    /** @brief Returns the names of the state @p encoder yields under "enc", in walk order. */
    std::vector<std::string> walked_names(TextEncoder& encoder)
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
        NN::walk_state(encoder, "enc", visitor);
        return names;
    }

    /** @brief Holds a factory row the factory refuses and two words of its message. */
    struct FactoryRow
    {
        std::string name;          /**< The row name. */
        std::string json;          /**< The attributes of the row. */
        std::vector<Shape> inputs; /**< The per-sample shapes the factory reads. */
        std::string first;         /**< A word of the message, after the context. */
        std::string second;        /**< A word of the message, or empty. */
    };

    /** @brief Runs one factory row the factory refuses. */
    class Small_BlockTextEncoderFactory : public ::testing::TestWithParam<FactoryRow>
    {

    };

    /** @brief Provides the Engine every numeric case runs on. */
    class Medium_BlockTextEncoder : public SaTest::EngineTest
    {

    };

    /**
     * @brief Holds a compiled graph of an encoder over @p texts texts of @p positions tokens,
     *        with the state write_state() of the blocks tests gives, and runs it.
     */
    class EncoderRun
    {
        public:
            /** @brief Traces the encoder, fills its state and compiles the step. */
            EncoderRun(SushiAI::ExecutionBackend& backend, int64_t texts, int64_t positions)
                : pool_(backend.engine()), encoder_(make_encoder())
            {
                const GraphBuilder builder(graph_);
                tokens_ = builder.input(Shape{texts, positions}, "tokens", DType::Int32);
                lengths_ = builder.input(Shape{texts}, "lengths");

                const Trace trace = Trace::evaluation(builder);
                encoded_ = encoder_.forward(trace, tokens_, lengths_);
                SA_EXPECT_OK(trace.status());

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                fill_state();
                // The graph is fixed here, so a refusal is a defect: require_value stops the test.
                step_ = SaTest::require_value(CompiledStep::compile(backend, graph_, pool_));
            }

            /** @brief Runs the step over @p tokens and @p lengths and returns the [N, dim] rows. */
            std::vector<double> encode(const std::vector<double>& tokens,
                                       const std::vector<double>& lengths)
            {
                write_tensor(pool_.tensor(tokens_), tokens);
                write_tensor(pool_.tensor(lengths_), lengths);
                SA_EXPECT_OK(step_->run());
                return read_tensor(pool_.tensor(encoded_));
            }

        private:
            /** @brief Writes eighths into every parameter, in walk order. */
            void fill_state()
            {
                int64_t offset = 0;
                const auto on_parameter = [&](const std::string&, NN::Parameter& parameter)
                {
                    write_tensor(pool_.tensor(parameter.id_in(graph_)),
                                 SaTest::eighths(parameter.shape().num_elements(), offset++));
                };
                const auto on_buffer = [](const std::string&, NN::Buffer&)
                {

                };
                const NN::StateVisitor visitor(on_parameter, on_buffer);
                NN::walk_state(encoder_, "", visitor);
            }

            Graph::Graph graph_;
            TensorPool pool_;
            TextEncoder encoder_;
            std::unique_ptr<CompiledStep> step_;
            ValueID tokens_ = Graph::K_NO_VALUE;
            ValueID lengths_ = Graph::K_NO_VALUE;
            ValueID encoded_ = Graph::K_NO_VALUE;
    };
} // namespace

/** @brief Tests that a trace over [N, T] tokens gives an [N, dim] value and records no failure. */
TEST(Small_BlockTextEncoder, ATraceGivesOneRowPerText)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID tokens =
        builder.input(Shape{2, K_PADDED_POSITIONS}, "tokens", DType::Int32);
    const Graph::ValueID lengths = builder.input(Shape{2}, "lengths");
    TextEncoder encoder = make_encoder();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID encoded = encoder.forward(trace, tokens, lengths);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(encoded).shape, (Shape{2, K_DIM}));
}

/** @brief Tests that a text longer than the position table fails and hands back the tokens. */
TEST(Small_BlockTextEncoder, ATextLongerThanThePositionTableFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID tokens =
        builder.input(Shape{1, K_TABLE_LENGTH + 1}, "tokens", DType::Int32);
    const Graph::ValueID lengths = builder.input(Shape{1}, "lengths");
    TextEncoder encoder = make_encoder();

    const Trace trace = Trace::evaluation(builder);
    EXPECT_EQ(encoder.forward(trace, tokens, lengths), tokens);

    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "7 positions", "6 rows");
}

/** @brief Tests that the per-sample shape of the result is the width of the encoder. */
TEST(Small_BlockTextEncoder, TheResultShapeIsTheWidth)
{
    const TextEncoder encoder = make_encoder();

    SA_ASSERT_OK_AND_ASSIGN(const Shape encoded, encoder.output_shape(Shape{K_PADDED_POSITIONS}));
    EXPECT_EQ(encoded, (Shape{K_DIM}));
    SA_EXPECT_FAILURE(encoder.output_shape(Shape{K_PADDED_POSITIONS, K_DIM}),
                      StatusCode::InvalidDescription, "per-sample [T] input", "[5, 8]");
}

/** @brief Tests that the walk names the token table, the positions and each block. */
TEST(Small_BlockTextEncoder, TheWalkNamesTheTableThePositionsAndEachBlock)
{
    TextEncoder encoder = make_encoder();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID tokens =
        builder.input(Shape{1, K_PADDED_POSITIONS}, "tokens", DType::Int32);
    const Graph::ValueID lengths = builder.input(Shape{1}, "lengths");
    const Trace trace = Trace::evaluation(builder);
    static_cast<void>(encoder.forward(trace, tokens, lengths));
    SA_EXPECT_OK(trace.status());

    const std::vector<std::string> names = walked_names(encoder);

    // Two tables, then per block two normalisations, qkv, proj, expand and project, each of two.
    ASSERT_EQ(names.size(), 2u + 12u * static_cast<std::size_t>(K_BLOCKS));
    EXPECT_EQ(names[0], "enc.embedding.weight");
    EXPECT_EQ(names[1], "enc.positions.weight");
    EXPECT_EQ(names[2], "enc.blocks.0.norm1.weight");
    EXPECT_EQ(names.back(), "enc.blocks.1.ff.project.bias");
}

/** @brief Tests that the factory builds a node of two inputs and one port, and traces it. */
TEST(Small_BlockTextEncoder, TheFactoryBuildsANodeOfTwoInputsAndOnePort)
{
    const std::vector<Shape> inputs = {Shape{K_PADDED_POSITIONS}, Shape{1}};
    const Config::Value attributes = SaTest::require_value(Config::parse_json(
        R"({"vocabulary": 11, "dim": 8, "heads": 2, "blocks": 2})", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_text_encoder_node(NN::ShapeSpan(inputs.data(), inputs.size()),
                                                 attributes, "layer 0"));

    EXPECT_EQ(node->input_count(), 2u);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].name, NN::K_SINGLE_PORT_NAME);
    EXPECT_EQ(node->ports()[0].shape, (Shape{K_DIM}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const std::vector<Graph::ValueID> values = {
        builder.input(Shape{2, K_PADDED_POSITIONS}, "tokens", DType::Int32),
        builder.input(Shape{2}, "lengths")};
    std::vector<Graph::ValueID> outputs;
    const Trace trace = Trace::evaluation(builder);
    SA_EXPECT_OK(node->forward(trace, NN::ValueSpan(values.data(), values.size()), outputs));

    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{2, K_DIM}));
}

/** @brief Tests that the factory refuses a stray key, a missing size and an input that misfits. */
TEST_P(Small_BlockTextEncoderFactory, TheFactoryRefusesARowThatDoesNotFit)
{
    const FactoryRow& row = GetParam();
    const Config::Value attributes =
        SaTest::require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_text_encoder_node(
            NN::ShapeSpan(row.inputs.data(), row.inputs.size()), attributes, "layer 3"),
        StatusCode::InvalidDescription, "layer 3", row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockTextEncoderFactory,
    ::testing::Values(
        FactoryRow{"StrayKey",
                   R"({"vocabulary": 11, "dim": 8, "heads": 2, "blocks": 2, "dropout": 0.1})",
                   {Shape{5}, Shape{1}}, "'dropout'", ""},
        FactoryRow{"NoVocabulary", R"({"dim": 8, "heads": 2, "blocks": 2})",
                   {Shape{5}, Shape{1}}, "'vocabulary'", "absent"},
        FactoryRow{"NoBlocks", R"({"vocabulary": 11, "dim": 8, "heads": 2})",
                   {Shape{5}, Shape{1}}, "'blocks'", "absent"},
        FactoryRow{"UnevenHeads", R"({"vocabulary": 11, "dim": 8, "heads": 3, "blocks": 2})",
                   {Shape{5}, Shape{1}}, "dim=8 heads=3", ""},
        FactoryRow{"LengthBelowTheTokens",
                   R"({"vocabulary": 11, "dim": 8, "heads": 2, "blocks": 2, "length": 4})",
                   {Shape{5}, Shape{1}}, "'length' as 4", "5 tokens"},
        FactoryRow{"OneInput", R"({"vocabulary": 11, "dim": 8, "heads": 2, "blocks": 2})",
                   {Shape{5}}, "tokens and the lengths", "1 inputs"},
        FactoryRow{"RankTwoTokens", R"({"vocabulary": 11, "dim": 8, "heads": 2, "blocks": 2})",
                   {Shape{5, 8}, Shape{1}}, "rank-1 [T] input", "[5, 8]"},
        FactoryRow{"TwoLengthsForOneText",
                   R"({"vocabulary": 11, "dim": 8, "heads": 2, "blocks": 2})",
                   {Shape{5}, Shape{2}}, "one length for each text", "[2]"}),
    row_name<FactoryRow>);

/**
 * @brief Tests that a text padded to T encodes as the same text at its own length.
 *
 * The oracle is the encoder run alone over each text: the short text at T of 3, the full text at
 * T of 5. The two texts share one padded batch, so the length of each row must reach its own
 * attention and its own pool.
 */
TEST_F(Medium_BlockTextEncoder, APaddedTextEncodesAsTheTextAtItsOwnLength)
{
    EncoderRun padded(backend(), 2, K_PADDED_POSITIONS);
    EncoderRun short_alone(backend(), 1, K_SHORT_POSITIONS);
    EncoderRun full_alone(backend(), 1, K_PADDED_POSITIONS);

    const std::vector<double> together = padded.encode(padded_tokens({2, 9}), batch_lengths());

    ASSERT_EQ(together.size(), static_cast<std::size_t>(2 * K_DIM));
    SaTest::expect_close_normwise<float>(
        SaTest::leading(together, 0, static_cast<std::size_t>(K_DIM)),
        short_alone.encode(short_tokens(), {static_cast<double>(K_SHORT_POSITIONS)}),
        K_PATH_OPERATIONS);
    SaTest::expect_close_normwise<float>(
        SaTest::leading(together, static_cast<std::size_t>(K_DIM), static_cast<std::size_t>(K_DIM)),
        full_alone.encode(full_tokens(), {static_cast<double>(K_PADDED_POSITIONS)}),
        K_PATH_OPERATIONS);
}

/** @brief Tests that changing a token beyond the length of its row changes no value of the result. */
TEST_F(Medium_BlockTextEncoder, ATokenBeyondTheLengthOfItsRowChangesNothing)
{
    EncoderRun run(backend(), 2, K_PADDED_POSITIONS);

    const std::vector<double> first = run.encode(padded_tokens({2, 9}), batch_lengths());
    const std::vector<double> second = run.encode(padded_tokens({5, 0}), batch_lengths());

    // The padded keys score exp(-1e9), which is exactly 0, so the rows are equal bit for bit.
    EXPECT_EQ(first, second);
}
