/**************************************************************************/
/* test_option_head.cpp                                                   */
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
 * @file test_option_head.cpp
 * @brief Tests OptionHead: its traces and refusals, and its scores, its fill and its gradient on
 *        an engine.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/blocks/option_head.hpp>
#include <SushiAI/language/blocks/text_encoder.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/padded_text_comparison.hpp>
#include <support/tolerance.hpp>
#include "blocks_test_support.hpp"

using namespace SushiAI;
using Language::Blocks::OptionHead;
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
    /** @brief Holds N, the number of questions of every case. */
    constexpr int64_t K_TEXTS = 2;

    /** @brief Holds K, the options of each question. */
    constexpr int64_t K_OPTIONS = 3;

    /** @brief Holds the width of the states the score cases read. */
    constexpr int64_t K_STATE_WIDTH = 4;

    /** @brief Holds the row count of the token table of the encoder. */
    constexpr int64_t K_VOCABULARY = 11;

    /** @brief Holds the width of the encoder. */
    constexpr int64_t K_DIM = 8;

    /** @brief Holds the head count of the encoder. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Holds the hidden width of the feed-forward of the encoder. */
    constexpr int64_t K_HIDDEN = 16;

    /** @brief Holds how many blocks the encoder stacks. */
    constexpr int64_t K_BLOCKS = 1;

    /** @brief Holds the row count of the position table of the encoder. */
    constexpr int64_t K_TABLE_LENGTH = 6;

    /** @brief Holds T of the option texts the encoding cases read. */
    constexpr int64_t K_POSITIONS = 4;

    /** @brief Holds the length of the option the encoding cases compare at its own length. */
    constexpr int64_t K_SHORT_POSITIONS = 2;

    /**
     * @brief Holds an upper bound of the rounded operations on the longest path to one value of
     *        an encoded option: one block of a normalisation, a projection, a score, a softmax
     *        and a feed-forward, then the pool.
     */
    constexpr std::size_t K_PATH_OPERATIONS = 512;

    /** @brief Names a row by its name field. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return info.param.name;
    }

    /** @brief Returns the encoder the encoding cases build. */
    TextEncoder make_encoder()
    {
        return TextEncoder(K_VOCABULARY, K_DIM, K_TABLE_LENGTH, K_HEADS, K_HIDDEN, K_BLOCKS,
                           NN::K_LAYER_NORM_EPSILON);
    }

    /** @brief Returns the softmax of one row of logits, worked on the host in double precision. */
    std::vector<double> softmax_of(const std::vector<double>& logits)
    {
        const double largest = *std::max_element(logits.begin(), logits.end());
        std::vector<double> weights;
        double sum = 0.0;
        for (const double logit : logits)
        {
            weights.push_back(std::exp(logit - largest));
            sum += weights.back();
        }

        for (double& weight : weights)
            weight /= sum;

        return weights;
    }

    /** @brief Returns the index of the first largest of @p values. */
    std::size_t argmax_of(const std::vector<double>& values)
    {
        return static_cast<std::size_t>(std::max_element(values.begin(), values.end())
                                        - values.begin());
    }

    /** @brief Holds the shapes of a score the trace refuses and two words of its message. */
    struct ScoreRow
    {
        std::string name;    /**< The row name. */
        Shape question;      /**< The shape of the question state. */
        Shape options;       /**< The shape of the option states. */
        Shape valid;         /**< The shape of the valid flags. */
        DType valid_type;    /**< The element type of the valid flags. */
        std::string first;   /**< A word of the message. */
        std::string second;  /**< A word of the message. */
    };

    /** @brief Runs one score the trace refuses. */
    class Small_BlockOptionHeadScore : public ::testing::TestWithParam<ScoreRow>
    {

    };

    /** @brief Holds the tokens and the lengths of an encoding the trace refuses. */
    struct EncodeRow
    {
        std::string name;    /**< The row name. */
        Shape tokens;        /**< The shape of the option tokens. */
        DType token_type;    /**< The element type of the option tokens. */
        Shape lengths;       /**< The shape of the lengths. */
        std::string first;   /**< A word of the message. */
        std::string second;  /**< A word of the message. */
    };

    /** @brief Runs one encoding the trace refuses. */
    class Small_BlockOptionHeadEncode : public ::testing::TestWithParam<EncodeRow>
    {

    };

    /** @brief Holds which options a question holds, for the cases that fill the rest. */
    struct MaskRow
    {
        std::string name;           /**< The row name. */
        std::vector<double> valid;  /**< The [2, 3] flags: 1 held, 0 padded. */
    };

    /** @brief Provides the Engine every numeric case runs on. */
    class Medium_BlockOptionHead : public SaTest::EngineTest
    {

    };

    /** @brief Runs the score and the fill over one mask of held options. */
    class Medium_BlockOptionHeadMask : public SaTest::EngineTest,
                                       public ::testing::WithParamInterface<MaskRow>
    {

    };

    /**
     * @brief Holds a FLOAT64 graph of a score over a [2, 4] question variable and a [2, 3, 4]
     *        option variable, whose logits, loss and gradient can all be evaluated at a point.
     *
     * The point is the question's 8 values followed by the options' 24. The loss is the
     * cross-entropy of the [2, 3] logits against the targets written with write_targets().
     */
    class ScoreProbe
    {
        public:
            /** @brief Builds the graph, differentiates it at both variables and compiles it. */
            explicit ScoreProbe(SushiAI::ExecutionBackend& backend) : pool_(backend.engine())
            {
                const GraphBuilder builder(graph_);
                question_ = builder.parameter(Shape{K_TEXTS, K_STATE_WIDTH}, "question",
                                              DType::Float64);
                options_ = builder.parameter(Shape{K_TEXTS, K_OPTIONS, K_STATE_WIDTH}, "options",
                                             DType::Float64);
                valid_ = builder.input(Shape{K_TEXTS, K_OPTIONS}, "valid", DType::Float64);

                OptionHead head;
                const Trace trace = Trace::evaluation(builder);
                logits_ = head.score(trace, question_, options_, valid_);
                SA_EXPECT_OK(trace.status());

                targets_ = builder.input(Shape{K_TEXTS, K_OPTIONS}, "targets", DType::Float64);
                // The extents are fixed here, so a refusal is a defect: value() stops the test.
                loss_ = builder.nll_loss(builder.log_softmax(logits_), targets_).value();

                const std::vector<ValueID> wrt = {question_, options_};
                // The graph is fixed here, so a refusal is a defect: value() stops the test.
                const std::vector<ValueID> gradients = Autograd::differentiate(
                    graph_, loss_,
                    SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).value();
                question_gradient_ = gradients.at(0);
                options_gradient_ = gradients.at(1);
                EXPECT_NE(question_gradient_, Graph::K_NO_VALUE);
                EXPECT_NE(options_gradient_, Graph::K_NO_VALUE);

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                // The graph is fixed here, so a refusal is a defect: require_value stops the test.
                step_ = SaTest::require_value(CompiledStep::compile(backend, graph_, pool_));
            }

            /** @brief Writes the valid flags, K per question. */
            void write_valid(const std::vector<double>& valid)
            {
                write_tensor(pool_.tensor(valid_), valid);
            }

            /** @brief Writes the targets: a one-hot row per question. */
            void write_targets(const std::vector<double>& targets)
            {
                write_tensor(pool_.tensor(targets_), targets);
            }

            /** @brief Runs the step at @p point and returns the [2, 3] logits. */
            std::vector<double> logits_at(SushiRuntime::span<const double> point)
            {
                run(point);
                return read_tensor(pool_.tensor(logits_));
            }

            /** @brief Returns the loss at @p point. */
            double value(SushiRuntime::span<const double> point)
            {
                run(point);
                return read_tensor(pool_.tensor(loss_))[0];
            }

            /** @brief Writes the gradient at @p point into @p out: question, then options. */
            void gradient(SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
            {
                run(point);

                std::vector<double> values = read_tensor(pool_.tensor(question_gradient_));
                const std::vector<double> options = read_tensor(pool_.tensor(options_gradient_));
                values.insert(values.end(), options.begin(), options.end());
                ASSERT_EQ(values.size(), out.size());
                std::copy(values.begin(), values.end(), out.begin());
            }

            /** @brief Returns the number of values in a point. */
            static constexpr std::size_t point_size()
            {
                return static_cast<std::size_t>(K_TEXTS * K_STATE_WIDTH
                                                + K_TEXTS * K_OPTIONS * K_STATE_WIDTH);
            }

        private:
            /** @brief Writes @p point into the two variables and runs the step. */
            void run(SushiRuntime::span<const double> point)
            {
                const std::size_t split = static_cast<std::size_t>(K_TEXTS * K_STATE_WIDTH);
                write_tensor(pool_.tensor(question_),
                             std::vector<double>(point.begin(), point.begin() + split));
                write_tensor(pool_.tensor(options_),
                             std::vector<double>(point.begin() + split, point.end()));
                SA_EXPECT_OK(step_->run());
            }

            Graph::Graph graph_;
            TensorPool pool_;
            std::unique_ptr<CompiledStep> step_;
            ValueID question_ = Graph::K_NO_VALUE;
            ValueID options_ = Graph::K_NO_VALUE;
            ValueID valid_ = Graph::K_NO_VALUE;
            ValueID logits_ = Graph::K_NO_VALUE;
            ValueID targets_ = Graph::K_NO_VALUE;
            ValueID loss_ = Graph::K_NO_VALUE;
            ValueID question_gradient_ = Graph::K_NO_VALUE;
            ValueID options_gradient_ = Graph::K_NO_VALUE;
    };

    /**
     * @brief Returns the point where a held option scores below zero and a padded one far above.
     *
     * Every question value is 1/2. A held option k holds -(k + 1)/16 in each channel, so it scores
     * -(k + 1)/8; a padded option holds 5/8 in each channel, so it would score 5/4 unmasked.
     */
    std::vector<double> point_where_padding_would_win(const std::vector<double>& valid)
    {
        std::vector<double> point(static_cast<std::size_t>(K_TEXTS * K_STATE_WIDTH), 0.5);
        for (int64_t text = 0; text < K_TEXTS; ++text)
        {
            for (int64_t option = 0; option < K_OPTIONS; ++option)
            {
                const bool held =
                    valid[static_cast<std::size_t>(text * K_OPTIONS + option)] >= 1.0;
                const double value =
                    held ? -static_cast<double>(option + 1) / 16.0 : 5.0 / 8.0;
                point.insert(point.end(), static_cast<std::size_t>(K_STATE_WIDTH), value);
            }
        }

        return point;
    }

    /** @brief Returns the first held option of each question as a one-hot row. */
    std::vector<double> first_held_targets(const std::vector<double>& valid)
    {
        std::vector<double> targets(static_cast<std::size_t>(K_TEXTS * K_OPTIONS), 0.0);
        for (int64_t text = 0; text < K_TEXTS; ++text)
        {
            for (int64_t option = 0; option < K_OPTIONS; ++option)
            {
                const std::size_t at = static_cast<std::size_t>(text * K_OPTIONS + option);
                if (valid[at] >= 1.0)
                {
                    targets[at] = 1.0;
                    break;
                }
            }
        }

        return targets;
    }

    /**
     * @brief Holds a compiled graph of an encoder over option texts, or over single texts, with
     *        the state of the encoding cases, and runs it.
     */
    class EncodeRun
    {
        public:
            /**
             * @brief Traces OptionHead::encode over [@p texts, @p options, @p positions] tokens
             *        when @p options is above 0, else the encoder alone over [@p texts,
             *        @p positions] tokens, and compiles the step.
             */
            EncodeRun(SushiAI::ExecutionBackend& backend, int64_t texts, int64_t options,
                      int64_t positions)
                : pool_(backend.engine()), encoder_(make_encoder())
            {
                const GraphBuilder builder(graph_);
                const bool grouped = options > 0;
                tokens_ = builder.input(grouped ? Shape{texts, options, positions}
                                                : Shape{texts, positions},
                                        "tokens", DType::Int32);
                lengths_ = builder.input(grouped ? Shape{texts, options} : Shape{texts},
                                         "lengths");

                const Trace trace = Trace::evaluation(builder);
                OptionHead head;
                encoded_ = grouped ? head.encode(trace, encoder_, tokens_, lengths_)
                                   : encoder_.forward(trace, tokens_, lengths_);
                SA_EXPECT_OK(trace.status());

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                fill_state();
                // The graph is fixed here, so a refusal is a defect: require_value stops the test.
                step_ = SaTest::require_value(CompiledStep::compile(backend, graph_, pool_));
            }

            /** @brief Runs the step and returns the encoded rows, the last axis fastest. */
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

/** @brief Tests that the encoding of [N, K, T] options is [N, K, dim] and the score [N, K]. */
TEST(Small_BlockOptionHead, ATraceGivesOneLogitPerOption)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID question = builder.input(Shape{K_TEXTS, K_DIM}, "question");
    const ValueID tokens =
        builder.input(Shape{K_TEXTS, K_OPTIONS, K_POSITIONS}, "tokens", DType::Int32);
    const ValueID lengths = builder.input(Shape{K_TEXTS, K_OPTIONS}, "lengths");
    const ValueID valid = builder.input(Shape{K_TEXTS, K_OPTIONS}, "valid");
    TextEncoder encoder = make_encoder();
    OptionHead head;

    const Trace trace = Trace::evaluation(builder);
    const ValueID states = head.encode(trace, encoder, tokens, lengths);
    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(states).shape, (Shape{K_TEXTS, K_OPTIONS, K_DIM}));

    const ValueID logits = head.score(trace, question, states, valid);
    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.value(logits).shape, (Shape{K_TEXTS, K_OPTIONS}));
}

/** @brief Tests that forward() is the encoding followed by the score. */
TEST(Small_BlockOptionHead, ForwardEncodesThenScores)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID question = builder.input(Shape{K_TEXTS, K_DIM}, "question");
    const ValueID tokens =
        builder.input(Shape{K_TEXTS, K_OPTIONS, K_POSITIONS}, "tokens", DType::Int32);
    const ValueID lengths = builder.input(Shape{K_TEXTS, K_OPTIONS}, "lengths");
    const ValueID valid = builder.input(Shape{K_TEXTS, K_OPTIONS}, "valid");
    TextEncoder encoder = make_encoder();
    OptionHead head;

    const Trace trace = Trace::evaluation(builder);
    const ValueID logits = head.forward(trace, encoder, question, tokens, lengths, valid);

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(logits).shape, (Shape{K_TEXTS, K_OPTIONS}));
}

/** @brief Tests that an operand of a score that does not fit fails and hands back the options. */
TEST_P(Small_BlockOptionHeadScore, AScoreThatDoesNotFitFails)
{
    const ScoreRow& row = GetParam();
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID question = builder.input(row.question, "question", DType::Float64);
    const ValueID options = builder.input(row.options, "options", DType::Float64);
    const ValueID valid = builder.input(row.valid, "valid", row.valid_type);
    OptionHead head;

    const Trace trace = Trace::evaluation(builder);
    EXPECT_EQ(head.score(trace, question, options, valid), options);

    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockOptionHeadScore,
    ::testing::Values(
        ScoreRow{"RankThreeQuestion", Shape{K_TEXTS, K_STATE_WIDTH, 1},
                 Shape{K_TEXTS, K_OPTIONS, K_STATE_WIDTH}, Shape{K_TEXTS, K_OPTIONS},
                 DType::Float64, "[N, dim] question state", "[2, 4, 1]"},
        ScoreRow{"AnotherWidth", Shape{K_TEXTS, K_STATE_WIDTH},
                 Shape{K_TEXTS, K_OPTIONS, K_STATE_WIDTH + 1}, Shape{K_TEXTS, K_OPTIONS},
                 DType::Float64, "batch and the width must agree", "[2, 3, 5]"},
        ScoreRow{"AnotherBatch", Shape{K_TEXTS + 1, K_STATE_WIDTH},
                 Shape{K_TEXTS, K_OPTIONS, K_STATE_WIDTH}, Shape{K_TEXTS, K_OPTIONS},
                 DType::Float64, "batch and the width must agree", "[3, 4]"},
        ScoreRow{"FlagsOfAnotherShape", Shape{K_TEXTS, K_STATE_WIDTH},
                 Shape{K_TEXTS, K_OPTIONS, K_STATE_WIDTH}, Shape{K_TEXTS, K_OPTIONS - 1},
                 DType::Float64, "one real floating valid flag", "[2, 2]"},
        ScoreRow{"IntegerFlags", Shape{K_TEXTS, K_STATE_WIDTH},
                 Shape{K_TEXTS, K_OPTIONS, K_STATE_WIDTH}, Shape{K_TEXTS, K_OPTIONS},
                 DType::Int32, "one real floating valid flag", "int32"}),
    row_name<ScoreRow>);

/** @brief Tests that an encoding that does not fit fails and hands back the tokens. */
TEST_P(Small_BlockOptionHeadEncode, AnEncodingThatDoesNotFitFails)
{
    const EncodeRow& row = GetParam();
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID tokens = builder.input(row.tokens, "tokens", row.token_type);
    const ValueID lengths = builder.input(row.lengths, "lengths");
    TextEncoder encoder = make_encoder();
    OptionHead head;

    const Trace trace = Trace::evaluation(builder);
    EXPECT_EQ(head.encode(trace, encoder, tokens, lengths), tokens);

    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockOptionHeadEncode,
    ::testing::Values(
        EncodeRow{"RankTwoTokens", Shape{K_TEXTS, K_POSITIONS}, DType::Int32,
                  Shape{K_TEXTS, K_OPTIONS}, "[N, K, T] INT32", "[2, 4]"},
        EncodeRow{"FloatTokens", Shape{K_TEXTS, K_OPTIONS, K_POSITIONS}, DType::Float32,
                  Shape{K_TEXTS, K_OPTIONS}, "[N, K, T] INT32", "float32"},
        EncodeRow{"OneLengthForEachText", Shape{K_TEXTS, K_OPTIONS, K_POSITIONS}, DType::Int32,
                  Shape{K_TEXTS}, "one length for each option", "[2]"},
        EncodeRow{"LongerThanThePositionTable", Shape{K_TEXTS, K_OPTIONS, K_TABLE_LENGTH + 1},
                  DType::Int32, Shape{K_TEXTS, K_OPTIONS}, "7 positions", "6 rows"}),
    row_name<EncodeRow>);

/** @brief Tests that the per-sample shape of the result is the option count. */
TEST(Small_BlockOptionHead, TheResultShapeIsTheOptionCount)
{
    const OptionHead head{};

    SA_ASSERT_OK_AND_ASSIGN(const Shape scored, head.output_shape(Shape{K_OPTIONS, K_POSITIONS}));
    EXPECT_EQ(scored, (Shape{K_OPTIONS}));
    SA_EXPECT_FAILURE(head.output_shape(Shape{K_POSITIONS}), StatusCode::InvalidDescription,
                      "per-sample [K, T] input", "[4]");
}

/**
 * @brief Tests that the logit of a held option is the dot product of the question state and the
 *        option state.
 *
 * The oracle is the product worked on the host in double precision from the same eighths. Every
 * option is held, so no fill enters.
 */
TEST_F(Medium_BlockOptionHead, ALogitIsTheDotProductOfTheQuestionAndTheOption)
{
    ScoreProbe probe(backend());
    const std::vector<double> valid(static_cast<std::size_t>(K_TEXTS * K_OPTIONS), 1.0);
    probe.write_valid(valid);
    probe.write_targets(first_held_targets(valid));
    const std::vector<double> point = SaTest::eighths(
        static_cast<int64_t>(ScoreProbe::point_size()), 3);

    const std::vector<double> logits =
        probe.logits_at(SushiRuntime::span<const double>(point.data(), point.size()));

    ASSERT_EQ(logits.size(), static_cast<std::size_t>(K_TEXTS * K_OPTIONS));
    const std::size_t options_at = static_cast<std::size_t>(K_TEXTS * K_STATE_WIDTH);
    for (int64_t text = 0; text < K_TEXTS; ++text)
    {
        for (int64_t option = 0; option < K_OPTIONS; ++option)
        {
            double dot = 0.0;
            double magnitude = 0.0;
            for (int64_t channel = 0; channel < K_STATE_WIDTH; ++channel)
            {
                const double q = point[static_cast<std::size_t>(text * K_STATE_WIDTH + channel)];
                const double o = point[options_at + static_cast<std::size_t>(
                    (text * K_OPTIONS + option) * K_STATE_WIDTH + channel)];
                dot += q * o;
                magnitude += std::fabs(q * o);
            }

            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<double>(
                logits[static_cast<std::size_t>(text * K_OPTIONS + option)], dot, magnitude,
                static_cast<std::size_t>(K_STATE_WIDTH))))
                << "at question " << text << " option " << option;
        }
    }
}

/**
 * @brief Tests that a padded option holds the fill, that a softmax weighs it exactly 0 and that
 *        the argmax of each row is a held option.
 *
 * The point gives every padded option the largest score of its row, so a mask that failed to
 * replace it would show as an argmax outside the list and a weight above 0.
 */
TEST_P(Medium_BlockOptionHeadMask, APaddedOptionWeighsZeroAndTheArgmaxIsHeld)
{
    const MaskRow& row = GetParam();
    ScoreProbe probe(backend());
    probe.write_valid(row.valid);
    probe.write_targets(first_held_targets(row.valid));
    const std::vector<double> point = point_where_padding_would_win(row.valid);

    const std::vector<double> logits =
        probe.logits_at(SushiRuntime::span<const double>(point.data(), point.size()));

    ASSERT_EQ(logits.size(), static_cast<std::size_t>(K_TEXTS * K_OPTIONS));
    for (int64_t text = 0; text < K_TEXTS; ++text)
    {
        const std::size_t first = static_cast<std::size_t>(text * K_OPTIONS);
        const std::vector<double> scores(logits.begin() + static_cast<std::ptrdiff_t>(first),
                                         logits.begin()
                                             + static_cast<std::ptrdiff_t>(first + K_OPTIONS));
        const std::vector<double> weights = softmax_of(scores);

        double held_sum = 0.0;
        for (std::size_t option = 0; option < static_cast<std::size_t>(K_OPTIONS); ++option)
        {
            if (row.valid[first + option] >= 1.0)
            {
                held_sum += weights[option];
                continue;
            }

            // The fill is exp(-1e9 - largest), which underflows to 0 in every real type.
            EXPECT_EQ(scores[option], Language::Blocks::K_MASKED_ATTENTION_SCORE)
                << "at question " << text << " option " << option;
            EXPECT_EQ(weights[option], 0.0) << "at question " << text << " option " << option;
        }

        EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<double>(
            held_sum, 1.0, 1.0, static_cast<std::size_t>(K_OPTIONS))))
            << "at question " << text;
        EXPECT_GE(row.valid[first + argmax_of(scores)], 1.0) << "at question " << text;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Masks, Medium_BlockOptionHeadMask,
    ::testing::Values(MaskRow{"NothingPadded", {1, 1, 1, 1, 1, 1}},
                      MaskRow{"LastPadded", {1, 1, 0, 1, 1, 0}},
                      MaskRow{"FirstPadded", {0, 1, 1, 0, 1, 1}},
                      MaskRow{"OneHeld", {1, 0, 0, 0, 0, 1}}),
    row_name<MaskRow>);

/**
 * @brief Tests that the gradient of a cross-entropy over the logits matches central differences
 *        at the question state and at the option states.
 *
 * The second question holds one padded option, so the gradient crosses the fill.
 */
TEST_F(Medium_BlockOptionHead, TheScoreDifferentiatesAtTheQuestionAndTheOptions)
{
    ScoreProbe probe(backend());
    const std::vector<double> valid = {1, 1, 1, 1, 0, 1};
    probe.write_valid(valid);
    probe.write_targets(first_held_targets(valid));
    std::vector<double> point = SaTest::eighths(static_cast<int64_t>(ScoreProbe::point_size()), 3);

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

/** @brief Tests that a padded option receives a gradient of exactly 0. */
TEST_F(Medium_BlockOptionHead, APaddedOptionReceivesNoGradient)
{
    ScoreProbe probe(backend());
    const std::vector<double> valid = {1, 1, 0, 1, 0, 0};
    probe.write_valid(valid);
    probe.write_targets(first_held_targets(valid));
    const std::vector<double> point = point_where_padding_would_win(valid);
    std::vector<double> gradient(ScoreProbe::point_size(), 1.0);

    probe.gradient(SushiRuntime::span<const double>(point.data(), point.size()),
                   SushiRuntime::span<double>(gradient.data(), gradient.size()));

    const std::size_t options_at = static_cast<std::size_t>(K_TEXTS * K_STATE_WIDTH);
    for (std::size_t option = 0; option < static_cast<std::size_t>(K_TEXTS * K_OPTIONS); ++option)
    {
        if (valid[option] >= 1.0)
            continue;

        for (std::size_t channel = 0; channel < static_cast<std::size_t>(K_STATE_WIDTH);
             ++channel)
        {
            // The select passes the held logits their gradient and the padded ones none.
            EXPECT_EQ(gradient[options_at + option * static_cast<std::size_t>(K_STATE_WIDTH)
                               + channel],
                      0.0)
                << "at option " << option << " channel " << channel;
        }
    }
}

/**
 * @brief Tests that an option encodes as the same text run alone at its own length.
 *
 * The oracle is the encoder run over the one option. Option (0, 0) holds 2 of 4 tokens, so its
 * padding must not reach its state; option (1, 2) fills its text.
 */
TEST_F(Medium_BlockOptionHead, AnOptionEncodesAsItsTextAlone)
{
    EncodeRun grouped(backend(), K_TEXTS, K_OPTIONS, K_POSITIONS);
    EncodeRun short_alone(backend(), 1, 0, K_SHORT_POSITIONS);
    EncodeRun full_alone(backend(), 1, 0, K_POSITIONS);
    const std::vector<double> tokens = {1, 4, 9, 9,  3, 8, 1, 2,  5, 5, 5, 5,
                                        7, 7, 6, 6,  2, 4, 6, 8,  3, 8, 1, 10};
    const std::vector<double> lengths = {2, 4, 3, 4, 1, 4};

    const std::vector<double> states = grouped.encode(tokens, lengths);

    ASSERT_EQ(states.size(), static_cast<std::size_t>(K_TEXTS * K_OPTIONS * K_DIM));
    SaTest::expect_close_normwise<float>(
        SaTest::row_of(states, 0, static_cast<std::size_t>(K_DIM)),
        short_alone.encode({1, 4}, {static_cast<double>(K_SHORT_POSITIONS)}), K_PATH_OPERATIONS);
    SaTest::expect_close_normwise<float>(
        SaTest::row_of(states, 5, static_cast<std::size_t>(K_DIM)),
        full_alone.encode({3, 8, 1, 10}, {static_cast<double>(K_POSITIONS)}), K_PATH_OPERATIONS);
}
