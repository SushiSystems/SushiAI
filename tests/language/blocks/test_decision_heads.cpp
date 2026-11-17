/**************************************************************************/
/* test_decision_heads.cpp                                                */
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
 * @file test_decision_heads.cpp
 * @brief Tests DecisionHeads and make_decision_node: the node, its three ports and refusals, and
 *        the choice, the yes-or-no logit and the number on an engine.
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
#include <SushiAI/language/blocks/decision_heads.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/blocks/text_encoder.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>
#include "blocks_test_support.hpp"

using namespace SushiAI;
using Language::Blocks::DecisionHeads;
using Language::Blocks::DecisionTrace;
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

    /** @brief Holds the row count of the token table of the encoder. */
    constexpr int64_t K_VOCABULARY = 11;

    /** @brief Holds the width of the encoder and of the states the heads read. */
    constexpr int64_t K_DIM = 8;

    /** @brief Holds the head count of the encoder. */
    constexpr int64_t K_HEADS = 2;

    /** @brief Holds the hidden width of the feed-forward of the encoder. */
    constexpr int64_t K_HIDDEN = 16;

    /** @brief Holds how many blocks the encoder stacks. */
    constexpr int64_t K_BLOCKS = 1;

    /** @brief Holds the row count of the position table of the encoder. */
    constexpr int64_t K_TABLE_LENGTH = 6;

    /** @brief Holds T of the question texts. */
    constexpr int64_t K_QUESTION_POSITIONS = 5;

    /** @brief Holds T of the option texts. */
    constexpr int64_t K_OPTION_POSITIONS = 4;

    /** @brief Holds the parameters of the two linear heads: a weight and a bias each. */
    constexpr int64_t K_HEAD_PARAMETERS = 4;

    /** @brief Holds the attributes of the row every case builds. */
    constexpr const char* K_ROW =
        R"({"vocabulary": 11, "dim": 8, "heads": 2, "blocks": 1, "options": 3})";

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

    /** @brief Returns the five per-sample shapes of the row every case builds. */
    std::vector<Shape> row_inputs()
    {
        return {Shape{K_QUESTION_POSITIONS}, Shape{1}, Shape{K_OPTIONS, K_OPTION_POSITIONS},
                Shape{K_OPTIONS}, Shape{K_OPTIONS}};
    }

    /** @brief Returns the names of the state @p heads yields under "dec", in walk order. */
    std::vector<std::string> walked_names(DecisionHeads& heads)
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
        NN::walk_state(heads, "dec", visitor);
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
    class Small_BlockDecisionHeadsFactory : public ::testing::TestWithParam<FactoryRow>
    {

    };

    /** @brief Provides the Engine every numeric case runs on. */
    class Medium_BlockDecisionHeads : public SaTest::EngineTest
    {

    };

    /** @brief Names the head a probe differentiates. */
    enum class Head
    {
        Choice, /**< The cross-entropy of the choice logits, at both states. */
        YesNo,  /**< The cross-entropy of the yes-or-no logits, at the question. */
    };

    /**
     * @brief Holds a FLOAT64 graph of the three heads over a [2, 8] question state and [2, 3, 8]
     *        option states, whose values, loss and gradient can all be evaluated at a point.
     *
     * The yes-or-no weight holds the eighths from offset 0, its bias from 1, the number's weight
     * from 2 and its bias from 3. For Head::Choice the point is the question's 16 values then the
     * options' 48; for Head::YesNo it is the question's alone, and the options hold fixed eighths.
     */
    class HeadsProbe
    {
        public:
            /** @brief Builds the graph, differentiates it for @p head and compiles it. */
            HeadsProbe(SushiAI::ExecutionBackend& backend, Head head)
                : head_(head), pool_(backend.engine()), heads_(make_encoder())
            {
                const GraphBuilder builder(graph_);
                question_ = builder.parameter(Shape{K_TEXTS, K_DIM}, "question", DType::Float64);
                const Shape option_shape{K_TEXTS, K_OPTIONS, K_DIM};
                options_ = head == Head::Choice
                    ? builder.parameter(option_shape, "options", DType::Float64)
                    : builder.input(option_shape, "options", DType::Float64);
                valid_ = builder.input(Shape{K_TEXTS, K_OPTIONS}, "valid", DType::Float64);

                const Trace trace = Trace::evaluation(builder);
                const Result<DecisionTrace> traced =
                    heads_.trace_heads(trace, question_, options_, valid_);
                SA_EXPECT_OK(traced);
                SA_EXPECT_OK(trace.status());
                traced_ = traced.value();

                targets_ = builder.input(head == Head::Choice ? Shape{K_TEXTS, K_OPTIONS}
                                                              : Shape{1, K_TEXTS},
                                         "targets", DType::Float64);
                // The extents are fixed here, so a refusal is a defect: value() stops the test.
                loss_ = (head == Head::Choice
                             ? builder.nll_loss(builder.log_softmax(traced_.choice), targets_)
                             : builder.nll_loss(
                                   builder.log_softmax(builder.reshape(traced_.yes_no,
                                                                       Shape{1, K_TEXTS})),
                                   targets_)).value();

                const std::vector<ValueID> wrt = head == Head::Choice
                    ? std::vector<ValueID>{question_, options_}
                    : std::vector<ValueID>{question_};
                // The graph is fixed here, so a refusal is a defect: value() stops the test.
                gradients_ = Autograd::differentiate(
                    graph_, loss_,
                    SushiRuntime::span<const ValueID>(wrt.data(), wrt.size())).value();
                for (const ValueID gradient : gradients_)
                    EXPECT_NE(gradient, Graph::K_NO_VALUE);

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                fill_heads();
                // The graph is fixed here, so a refusal is a defect: require_value stops the test.
                step_ = SaTest::require_value(CompiledStep::compile(backend, graph_, pool_));
            }

            /** @brief Writes the valid flags and the one-hot targets of the loss. */
            void write_inputs(const std::vector<double>& valid, const std::vector<double>& targets)
            {
                write_tensor(pool_.tensor(valid_), valid);
                write_tensor(pool_.tensor(targets_), targets);
            }

            /** @brief Runs the step at @p point and returns the [2, 1] yes-or-no logits. */
            std::vector<double> yes_no_at(SushiRuntime::span<const double> point)
            {
                run(point);
                return read_tensor(pool_.tensor(traced_.yes_no));
            }

            /** @brief Runs the step at @p point and returns the [2, 1] numbers. */
            std::vector<double> number_at(SushiRuntime::span<const double> point)
            {
                run(point);
                return read_tensor(pool_.tensor(traced_.number));
            }

            /** @brief Returns the loss at @p point. */
            double value(SushiRuntime::span<const double> point)
            {
                run(point);
                return read_tensor(pool_.tensor(loss_))[0];
            }

            /** @brief Writes the gradient at @p point into @p out, in the order of the point. */
            void gradient(SushiRuntime::span<const double> point, SushiRuntime::span<double> out)
            {
                run(point);

                std::vector<double> values;
                for (const ValueID id : gradients_)
                {
                    const std::vector<double> part = read_tensor(pool_.tensor(id));
                    values.insert(values.end(), part.begin(), part.end());
                }

                ASSERT_EQ(values.size(), out.size());
                std::copy(values.begin(), values.end(), out.begin());
            }

            /** @brief Returns the number of values in a point of @p head. */
            static std::size_t point_size(Head head)
            {
                const int64_t question = K_TEXTS * K_DIM;
                const int64_t options = K_TEXTS * K_OPTIONS * K_DIM;
                return static_cast<std::size_t>(head == Head::Choice ? question + options
                                                                     : question);
            }

        private:
            /** @brief Writes eighths into the two linear heads, in walk order. */
            void fill_heads()
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
                NN::walk_state(heads_.yes_no, "", visitor);
                NN::walk_state(heads_.number, "", visitor);
                EXPECT_EQ(offset, K_HEAD_PARAMETERS);
            }

            /** @brief Writes @p point into the variables and runs the step. */
            void run(SushiRuntime::span<const double> point)
            {
                const std::size_t split = static_cast<std::size_t>(K_TEXTS * K_DIM);
                write_tensor(pool_.tensor(question_),
                             std::vector<double>(point.begin(), point.begin() + split));
                write_tensor(pool_.tensor(options_),
                             head_ == Head::Choice
                                 ? std::vector<double>(point.begin() + split, point.end())
                                 : SaTest::eighths(K_TEXTS * K_OPTIONS * K_DIM, 5));
                SA_EXPECT_OK(step_->run());
            }

            Head head_;
            Graph::Graph graph_;
            TensorPool pool_;
            DecisionHeads heads_;
            DecisionTrace traced_{};
            std::unique_ptr<CompiledStep> step_;
            ValueID question_ = Graph::K_NO_VALUE;
            ValueID options_ = Graph::K_NO_VALUE;
            ValueID valid_ = Graph::K_NO_VALUE;
            ValueID targets_ = Graph::K_NO_VALUE;
            ValueID loss_ = Graph::K_NO_VALUE;
            std::vector<ValueID> gradients_;
    };

    /**
     * @brief Holds a compiled graph of the whole decision model over [2, 5] questions and
     *        [2, 3, 4] options, with eighths in every parameter, and runs it.
     */
    class DecisionRun
    {
        public:
            /** @brief Traces the heads, fills the state and compiles the step. */
            explicit DecisionRun(SushiAI::ExecutionBackend& backend)
                : pool_(backend.engine()), heads_(make_encoder())
            {
                const GraphBuilder builder(graph_);
                question_tokens_ = builder.input(Shape{K_TEXTS, K_QUESTION_POSITIONS},
                                                 "question_tokens", DType::Int32);
                question_lengths_ = builder.input(Shape{K_TEXTS}, "question_lengths");
                option_tokens_ = builder.input(Shape{K_TEXTS, K_OPTIONS, K_OPTION_POSITIONS},
                                               "option_tokens", DType::Int32);
                option_lengths_ = builder.input(Shape{K_TEXTS, K_OPTIONS}, "option_lengths");
                option_valid_ = builder.input(Shape{K_TEXTS, K_OPTIONS}, "option_valid");

                const Trace trace = Trace::evaluation(builder);
                const Result<DecisionTrace> traced = heads_.trace_decision(
                    trace, question_tokens_, question_lengths_, option_tokens_, option_lengths_,
                    option_valid_);
                SA_EXPECT_OK(traced);
                SA_EXPECT_OK(trace.status());
                traced_ = traced.value();

                SA_EXPECT_OK(pool_.materialise_leaves(graph_));
                fill_state();
                // The graph is fixed here, so a refusal is a defect: require_value stops the test.
                step_ = SaTest::require_value(CompiledStep::compile(backend, graph_, pool_));
            }

            /** @brief Runs the step over @p option_tokens and the fixed questions and flags. */
            void run(const std::vector<double>& option_tokens, const std::vector<double>& valid)
            {
                write_tensor(pool_.tensor(question_tokens_),
                             std::vector<double>{1, 4, 7, 2, 2,  3, 8, 1, 10, 6});
                write_tensor(pool_.tensor(question_lengths_), std::vector<double>{3, 5});
                write_tensor(pool_.tensor(option_tokens_), option_tokens);
                write_tensor(pool_.tensor(option_lengths_), std::vector<double>{2, 4, 1, 3, 4, 2});
                write_tensor(pool_.tensor(option_valid_), valid);
                SA_EXPECT_OK(step_->run());
            }

            /** @brief Returns the [2, 3] choice logits of the last run. */
            std::vector<double> choice()
            {
                return read_tensor(pool_.tensor(traced_.choice));
            }

            /** @brief Returns the [2, 1] yes-or-no logits of the last run. */
            std::vector<double> yes_no()
            {
                return read_tensor(pool_.tensor(traced_.yes_no));
            }

            /** @brief Returns the [2, 1] numbers of the last run. */
            std::vector<double> number()
            {
                return read_tensor(pool_.tensor(traced_.number));
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
                NN::walk_state(heads_, "", visitor);
            }

            Graph::Graph graph_;
            TensorPool pool_;
            DecisionHeads heads_;
            DecisionTrace traced_{};
            std::unique_ptr<CompiledStep> step_;
            ValueID question_tokens_ = Graph::K_NO_VALUE;
            ValueID question_lengths_ = Graph::K_NO_VALUE;
            ValueID option_tokens_ = Graph::K_NO_VALUE;
            ValueID option_lengths_ = Graph::K_NO_VALUE;
            ValueID option_valid_ = Graph::K_NO_VALUE;
    };

    /** @brief Returns the logit a linear head of eighths gives row @p text of @p point. */
    double linear_head_on_host(const std::vector<double>& point, int64_t text,
                               int64_t weight_offset, double& magnitude)
    {
        const std::vector<double> weight = SaTest::eighths(K_DIM, weight_offset);
        const double bias = SaTest::eighths(1, weight_offset + 1)[0];
        double sum = bias;
        magnitude = std::fabs(bias);
        for (int64_t channel = 0; channel < K_DIM; ++channel)
        {
            const double product = weight[static_cast<std::size_t>(channel)]
                                   * point[static_cast<std::size_t>(text * K_DIM + channel)];
            sum += product;
            magnitude += std::fabs(product);
        }

        return sum;
    }

    /** @brief Returns the one-hot targets of the yes-or-no loss: the first question says yes. */
    std::vector<double> yes_targets()
    {
        return {1.0, 0.0};
    }

    /** @brief Returns the one-hot rows of the choice loss: option 0, then option 1. */
    std::vector<double> choice_targets()
    {
        return {1, 0, 0, 0, 1, 0};
    }
} // namespace

/** @brief Tests that the factory builds a node of five inputs and the three named ports. */
TEST(Small_BlockDecisionHeads, TheFactoryBuildsANodeOfFiveInputsAndThreePorts)
{
    const std::vector<Shape> inputs = row_inputs();
    const Config::Value attributes =
        SaTest::require_value(Config::parse_json(K_ROW, "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_decision_node(NN::ShapeSpan(inputs.data(), inputs.size()),
                                             attributes, "layer 0"));

    EXPECT_EQ(node->input_count(), 5u);
    ASSERT_EQ(node->ports().size(), 3u);
    EXPECT_EQ(node->ports()[0].name, Language::Blocks::K_CHOICE_PORT);
    EXPECT_EQ(node->ports()[0].shape, (Shape{K_OPTIONS}));
    EXPECT_EQ(node->ports()[1].name, Language::Blocks::K_YES_NO_PORT);
    EXPECT_EQ(node->ports()[1].shape, (Shape{1}));
    EXPECT_EQ(node->ports()[2].name, Language::Blocks::K_NUMBER_PORT);
    EXPECT_EQ(node->ports()[2].shape, (Shape{1}));
}

/** @brief Tests that tracing the node appends the choice, the yes-or-no logit and the number. */
TEST(Small_BlockDecisionHeads, ATraceOfTheNodeAppendsThreeValuesOfTheStatedShapes)
{
    const std::vector<Shape> inputs = row_inputs();
    const Config::Value attributes =
        SaTest::require_value(Config::parse_json(K_ROW, "<test>"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_decision_node(NN::ShapeSpan(inputs.data(), inputs.size()),
                                             attributes, "layer 0"));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const std::vector<ValueID> values = {
        builder.input(Shape{K_TEXTS, K_QUESTION_POSITIONS}, "question", DType::Int32),
        builder.input(Shape{K_TEXTS}, "question_lengths"),
        builder.input(Shape{K_TEXTS, K_OPTIONS, K_OPTION_POSITIONS}, "options", DType::Int32),
        builder.input(Shape{K_TEXTS, K_OPTIONS}, "option_lengths"),
        builder.input(Shape{K_TEXTS, K_OPTIONS}, "option_valid")};
    std::vector<ValueID> outputs;
    const Trace trace = Trace::evaluation(builder);
    SA_EXPECT_OK(node->forward(trace, NN::ValueSpan(values.data(), values.size()), outputs));

    ASSERT_EQ(outputs.size(), 3u);
    EXPECT_EQ(graph.value(outputs[0]).shape, (Shape{K_TEXTS, K_OPTIONS}));
    EXPECT_EQ(graph.value(outputs[1]).shape, (Shape{K_TEXTS, 1}));
    EXPECT_EQ(graph.value(outputs[2]).shape, (Shape{K_TEXTS, 1}));
}

/** @brief Tests that the node refuses four values and leaves its outputs unchanged. */
TEST(Small_BlockDecisionHeads, TheNodeRefusesFourValues)
{
    const std::vector<Shape> inputs = row_inputs();
    const Config::Value attributes =
        SaTest::require_value(Config::parse_json(K_ROW, "<test>"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_decision_node(NN::ShapeSpan(inputs.data(), inputs.size()),
                                             attributes, "layer 0"));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const std::vector<ValueID> values = {
        builder.input(Shape{K_TEXTS, K_QUESTION_POSITIONS}, "question", DType::Int32),
        builder.input(Shape{K_TEXTS}, "question_lengths"),
        builder.input(Shape{K_TEXTS, K_OPTIONS, K_OPTION_POSITIONS}, "options", DType::Int32),
        builder.input(Shape{K_TEXTS, K_OPTIONS}, "option_lengths")};
    std::vector<ValueID> outputs;
    const Trace trace = Trace::evaluation(builder);

    SA_EXPECT_FAILURE(node->forward(trace, NN::ValueSpan(values.data(), values.size()), outputs),
                      StatusCode::InvalidDescription, "input count is 5", "given is 4");
    EXPECT_TRUE(outputs.empty());
}

/** @brief Tests that flags of the wrong shape fail the trace and return the failure. */
TEST(Small_BlockDecisionHeads, FlagsOfTheWrongShapeFailTheTrace)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    DecisionHeads heads(make_encoder());
    const ValueID question = builder.input(Shape{K_TEXTS, K_QUESTION_POSITIONS}, "question",
                                           DType::Int32);
    const ValueID question_lengths = builder.input(Shape{K_TEXTS}, "question_lengths");
    const ValueID options =
        builder.input(Shape{K_TEXTS, K_OPTIONS, K_OPTION_POSITIONS}, "options", DType::Int32);
    const ValueID option_lengths = builder.input(Shape{K_TEXTS, K_OPTIONS}, "option_lengths");
    const ValueID flags = builder.input(Shape{K_TEXTS, K_OPTIONS - 1}, "flags");

    const Trace trace = Trace::evaluation(builder);

    SA_EXPECT_FAILURE(heads.trace_decision(trace, question, question_lengths, options,
                                           option_lengths, flags),
                      StatusCode::InvalidDescription, "one real floating valid flag", "[2, 2]");
}

/** @brief Tests that the walk names the encoder and the two linear heads, in that order. */
TEST(Small_BlockDecisionHeads, TheWalkNamesTheEncoderAndTheTwoHeads)
{
    DecisionHeads heads(make_encoder());
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const ValueID question = builder.input(Shape{1, K_QUESTION_POSITIONS}, "question",
                                           DType::Int32);
    const ValueID question_lengths = builder.input(Shape{1}, "question_lengths");
    const ValueID options =
        builder.input(Shape{1, K_OPTIONS, K_OPTION_POSITIONS}, "options", DType::Int32);
    const ValueID option_lengths = builder.input(Shape{1, K_OPTIONS}, "option_lengths");
    const ValueID flags = builder.input(Shape{1, K_OPTIONS}, "flags");
    const Trace trace = Trace::evaluation(builder);
    SA_EXPECT_OK(heads.trace_decision(trace, question, question_lengths, options, option_lengths,
                                      flags));

    const std::vector<std::string> names = walked_names(heads);

    // Two tables and twelve entries of the one block, then a weight and a bias for each head.
    ASSERT_EQ(names.size(), 2u + 12u * static_cast<std::size_t>(K_BLOCKS) + 4u);
    EXPECT_EQ(names.front(), "dec.encoder.embedding.weight");
    EXPECT_EQ(names[14], "dec.yes_no.weight");
    EXPECT_EQ(names[15], "dec.yes_no.bias");
    EXPECT_EQ(names[16], "dec.number.weight");
    EXPECT_EQ(names.back(), "dec.number.bias");
}

/** @brief Tests that the factory refuses a stray key, a bad "options" and an input that misfits. */
TEST_P(Small_BlockDecisionHeadsFactory, TheFactoryRefusesARowThatDoesNotFit)
{
    const FactoryRow& row = GetParam();
    const Config::Value attributes =
        SaTest::require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_decision_node(NN::ShapeSpan(row.inputs.data(), row.inputs.size()),
                                             attributes, "layer 3"),
        StatusCode::InvalidDescription, "layer 3", row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockDecisionHeadsFactory,
    ::testing::Values(
        FactoryRow{"StrayKey",
                   R"({"vocabulary": 11, "dim": 8, "heads": 2, "blocks": 1, "options": 3,
                       "top_k": 1})",
                   row_inputs(), "'top_k'", ""},
        FactoryRow{"NoOptions", R"({"vocabulary": 11, "dim": 8, "heads": 2, "blocks": 1})",
                   row_inputs(), "'options'", "absent"},
        FactoryRow{"OptionsAboveTheTokens",
                   R"({"vocabulary": 11, "dim": 8, "heads": 2, "blocks": 1, "options": 4})",
                   row_inputs(), "'options' as 4", "3 options"},
        FactoryRow{"NoVocabulary", R"({"dim": 8, "heads": 2, "blocks": 1, "options": 3})",
                   row_inputs(), "'vocabulary'", "absent"},
        FactoryRow{"LengthBelowTheLongerText",
                   R"({"vocabulary": 11, "dim": 8, "heads": 2, "blocks": 1, "options": 3,
                       "length": 4})",
                   row_inputs(), "'length' as 4", "5 tokens"},
        FactoryRow{"FourInputs", K_ROW,
                   {Shape{5}, Shape{1}, Shape{3, 4}, Shape{3}}, "the question, its length",
                   "4 inputs"},
        FactoryRow{"RankTwoQuestion", K_ROW,
                   {Shape{5, 8}, Shape{1}, Shape{3, 4}, Shape{3}, Shape{3}}, "rank-1 [T] input",
                   "[5, 8]"},
        FactoryRow{"TwoQuestionLengths", K_ROW,
                   {Shape{5}, Shape{2}, Shape{3, 4}, Shape{3}, Shape{3}}, "one length, [1]", "[2]"},
        FactoryRow{"RankOneOptions", K_ROW,
                   {Shape{5}, Shape{1}, Shape{3}, Shape{3}, Shape{3}}, "rank-2 [K, T] input",
                   "[3]"},
        FactoryRow{"TwoOptionLengths", K_ROW,
                   {Shape{5}, Shape{1}, Shape{3, 4}, Shape{2}, Shape{3}},
                   "one length for each of its 3 options", "[2]"},
        FactoryRow{"TwoOptionFlags", K_ROW,
                   {Shape{5}, Shape{1}, Shape{3, 4}, Shape{3}, Shape{2}},
                   "one flag for each of its 3 options", "[2]"}),
    row_name<FactoryRow>);

/**
 * @brief Tests that the yes-or-no logit and the number are the linear heads of the question state.
 *
 * The oracle is w . q + b worked on the host in double precision from the eighths each head holds.
 */
TEST_F(Medium_BlockDecisionHeads, TheScalarHeadsAreLinearInTheQuestionState)
{
    HeadsProbe probe(backend(), Head::YesNo);
    probe.write_inputs({1, 1, 1, 1, 1, 1}, yes_targets());
    const std::vector<double> point = SaTest::eighths(K_TEXTS * K_DIM, 3);
    const SushiRuntime::span<const double> at(point.data(), point.size());

    const std::vector<double> yes_no = probe.yes_no_at(at);
    const std::vector<double> number = probe.number_at(at);

    ASSERT_EQ(yes_no.size(), static_cast<std::size_t>(K_TEXTS));
    ASSERT_EQ(number.size(), static_cast<std::size_t>(K_TEXTS));
    for (int64_t text = 0; text < K_TEXTS; ++text)
    {
        double yes_magnitude = 0.0;
        double number_magnitude = 0.0;
        const double yes_reference = linear_head_on_host(point, text, 0, yes_magnitude);
        const double number_reference = linear_head_on_host(point, text, 2, number_magnitude);

        EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<double>(
            yes_no[static_cast<std::size_t>(text)], yes_reference, yes_magnitude,
            static_cast<std::size_t>(K_DIM) + 1)))
            << "yes_no at question " << text;
        EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<double>(
            number[static_cast<std::size_t>(text)], number_reference, number_magnitude,
            static_cast<std::size_t>(K_DIM) + 1)))
            << "number at question " << text;
    }
}

/**
 * @brief Tests that the gradient of a cross-entropy over the choice matches central differences
 *        at the question state and at the option states.
 *
 * The second question holds one padded option, so the gradient crosses the fill.
 */
TEST_F(Medium_BlockDecisionHeads, TheChoiceDifferentiatesAtTheQuestionAndTheOptions)
{
    HeadsProbe probe(backend(), Head::Choice);
    probe.write_inputs({1, 1, 1, 1, 1, 0}, choice_targets());
    std::vector<double> point = SaTest::eighths(
        static_cast<int64_t>(HeadsProbe::point_size(Head::Choice)), 3);

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

/**
 * @brief Tests that the gradient of a cross-entropy over the yes-or-no logits of the two questions
 *        matches central differences at the question state.
 */
TEST_F(Medium_BlockDecisionHeads, TheYesNoLogitDifferentiatesAtTheQuestion)
{
    HeadsProbe probe(backend(), Head::YesNo);
    probe.write_inputs({1, 1, 1, 1, 1, 1}, yes_targets());
    std::vector<double> point = SaTest::eighths(
        static_cast<int64_t>(HeadsProbe::point_size(Head::YesNo)), 3);

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

/**
 * @brief Tests that the whole model holds the fill at each padded option, scores each held option
 *        with a finite logit and picks a held option.
 */
TEST_F(Medium_BlockDecisionHeads, ThePaddedOptionsOfAModelHoldTheFill)
{
    DecisionRun run(backend());
    const std::vector<double> valid = {1, 1, 0, 1, 0, 1};
    const std::vector<double> tokens = {1, 4, 9, 9,  3, 8, 1, 2,  5, 5, 5, 5,
                                        7, 7, 6, 6,  2, 4, 6, 8,  3, 8, 1, 10};

    run.run(tokens, valid);
    const std::vector<double> choice = run.choice();

    ASSERT_EQ(choice.size(), static_cast<std::size_t>(K_TEXTS * K_OPTIONS));
    for (std::size_t option = 0; option < choice.size(); ++option)
    {
        if (valid[option] >= 1.0)
            EXPECT_TRUE(std::isfinite(choice[option]) && choice[option] > -1.0e8)
                << "at option " << option;
        else
            EXPECT_EQ(choice[option], Language::Blocks::K_MASKED_ATTENTION_SCORE)
                << "at option " << option;
    }

    for (std::size_t text = 0; text < static_cast<std::size_t>(K_TEXTS); ++text)
    {
        const auto first = choice.begin() + static_cast<std::ptrdiff_t>(text * K_OPTIONS);
        const std::size_t best = static_cast<std::size_t>(
            std::max_element(first, first + K_OPTIONS) - first);
        EXPECT_GE(valid[text * static_cast<std::size_t>(K_OPTIONS) + best], 1.0)
            << "at question " << text;
    }
}

/** @brief Tests that the yes-or-no logit and the number read the question and no option. */
TEST_F(Medium_BlockDecisionHeads, TheScalarHeadsReadNoOption)
{
    DecisionRun run(backend());
    const std::vector<double> valid = {1, 1, 1, 1, 1, 1};

    run.run({1, 4, 9, 9,  3, 8, 1, 2,  5, 5, 5, 5,  7, 7, 6, 6,  2, 4, 6, 8,  3, 8, 1, 10}, valid);
    const std::vector<double> yes_no = run.yes_no();
    const std::vector<double> number = run.number();
    const std::vector<double> choice = run.choice();
    run.run({2, 2, 2, 2,  9, 9, 9, 9,  4, 3, 2, 1,  6, 6, 6, 6,  8, 7, 6, 5,  1, 1, 1, 1}, valid);

    // The heads read the question state alone, so another option text moves the choice only.
    EXPECT_EQ(run.yes_no(), yes_no);
    EXPECT_EQ(run.number(), number);
    EXPECT_NE(run.choice(), choice);
}
