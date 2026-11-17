/**************************************************************************/
/* test_decoupled_head.cpp                                                */
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
 * @file test_decoupled_head.cpp
 * @brief Tests DecoupledHead's ports, outputs and state names against the YOLOX head's layout.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
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
#include <SushiAI/vision/blocks/decoupled_head.hpp>
#include <SushiAI/vision/detect/yolox/yolox_head_contract.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using NN::NormOptions;
using NN::Trace;
using Vision::Blocks::DecoupledHead;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Holds the epsilon and the momentum of the YOLOX norms, 1e-3 and 0.03. */
    constexpr NormOptions K_YOLOX_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 30 'decoupled_head'";

    /** @brief Holds the class count of the fixture. */
    constexpr int64_t K_CLASSES = 3;

    /** @brief Holds the width of the fixture's stems and towers. */
    constexpr int64_t K_HIDDEN = 8;

    /** @brief Holds the number of levels of the fixture, as the YOLOX heads have. */
    constexpr std::size_t K_LEVELS = 3;

    /** @brief Holds the channels of each level's feature. */
    constexpr std::array<int64_t, K_LEVELS> K_WIDTHS = {16, 32, 64};

    /** @brief Holds the H and the W of each level. */
    constexpr std::array<int64_t, K_LEVELS> K_SIDES = {8, 4, 2};

    /** @brief Holds the batch extent of a traced graph. */
    constexpr int64_t K_BATCH = 2;

    /** @brief Writes one piece of state as "name [extents]", for a readable mismatch. */
    std::string record(const std::string& name, const Shape& shape)
    {
        return name + " " + shape.to_string();
    }

    /** @brief Lists every piece of state under @p module, in the order the walk yields. */
    template<typename M>
    std::vector<std::string> walk(M& module, std::string_view prefix)
    {
        std::vector<std::string> records;
        const auto on_parameter = [&records](const std::string& name, NN::Parameter& state)
        {
            records.push_back(record(name, state.shape()));
        };
        const auto on_buffer = [&records](const std::string& name, NN::Buffer& state)
        {
            records.push_back(record(name, state.shape()));
        };
        const NN::StateVisitor visitor(on_parameter, on_buffer);
        NN::walk_state(module, prefix, visitor);

        return records;
    }

    /** @brief Appends the five records of a Conv and its norm under @p prefix. */
    void append_conv_norm(std::vector<std::string>& records, const std::string& prefix,
                          int64_t out_channels, int64_t kernel, int64_t in_channels)
    {
        const Shape channels{out_channels};
        records.push_back(record(prefix + ".conv.weight",
                                 Shape{out_channels, kernel, kernel, in_channels}));
        records.push_back(record(prefix + ".bn.weight", channels));
        records.push_back(record(prefix + ".bn.bias", channels));
        records.push_back(record(prefix + ".bn.running_mean", channels));
        records.push_back(record(prefix + ".bn.running_var", channels));
    }

    /** @brief Appends the weight and the bias of a biased 1x1 under @p prefix. */
    void append_prediction(std::vector<std::string>& records, const std::string& prefix,
                           int64_t out_channels)
    {
        records.push_back(record(prefix + ".weight", Shape{out_channels, 1, 1, K_HIDDEN}));
        records.push_back(record(prefix + ".bias", Shape{out_channels}));
    }

    /** @brief Appends the towers of every level under @p prefix, "<prefix>.<level>.<index>". */
    void append_towers(std::vector<std::string>& records, const std::string& prefix)
    {
        for (std::size_t level = 0; level < K_LEVELS; ++level)
        {
            const std::string tower = prefix + "." + std::to_string(level);
            append_conv_norm(records, tower + ".0", K_HIDDEN, 3, K_HIDDEN);
            append_conv_norm(records, tower + ".1", K_HIDDEN, 3, K_HIDDEN);
        }
    }

    /** @brief Appends the prediction of each level under @p prefix, "<prefix>.<level>". */
    void append_predictions(std::vector<std::string>& records, const std::string& prefix,
                            int64_t out_channels)
    {
        for (std::size_t level = 0; level < K_LEVELS; ++level)
            append_prediction(records, prefix + "." + std::to_string(level), out_channels);
    }

    /** @brief Lists the records of the whole head in the order of the YOLOX state dict. */
    std::vector<std::string> expected_state(const std::string& head)
    {
        std::vector<std::string> records;
        append_towers(records, head + ".cls_convs");
        append_towers(records, head + ".reg_convs");
        append_predictions(records, head + ".cls_preds", K_CLASSES);
        append_predictions(records, head + ".reg_preds", 4);
        append_predictions(records, head + ".obj_preds", 1);
        for (std::size_t level = 0; level < K_LEVELS; ++level)
            append_conv_norm(records, head + ".stems." + std::to_string(level), K_HIDDEN, 1,
                             K_WIDTHS[level]);

        return records;
    }

    /** @brief Lists the per-sample [side, side, width] shape of each level. */
    std::vector<Shape> level_shapes()
    {
        std::vector<Shape> shapes;
        for (std::size_t level = 0; level < K_LEVELS; ++level)
            shapes.push_back(Shape{K_SIDES[level], K_SIDES[level], K_WIDTHS[level]});

        return shapes;
    }

    /** @brief Views @p shapes as the inputs of a row. */
    NN::ShapeSpan span_of(const std::vector<Shape>& shapes)
    {
        return NN::ShapeSpan(shapes.data(), shapes.size());
    }

    /** @brief Holds a graph with the three features of a batch of K_BATCH declared in it. */
    struct FeatureGraph
    {
        /** @brief Holds the graph a head is traced into. */
        Graph::Graph graph;

        /** @brief Holds the builder over the graph. */
        Graph::GraphBuilder builder{graph};

        /** @brief Holds the id of each level's [K_BATCH, side, side, width] input. */
        std::vector<Graph::ValueID> features;

        /** @brief Declares one input per level of K_SIDES and K_WIDTHS. */
        FeatureGraph()
        {
            for (std::size_t level = 0; level < K_LEVELS; ++level)
                features.push_back(builder.input(
                    Shape{K_BATCH, K_SIDES[level], K_SIDES[level], K_WIDTHS[level]}, "feature"));
        }

        /** @brief Views the three inputs as the values a head reads. */
        NN::ValueSpan inputs() const
        {
            return NN::ValueSpan(features.data(), features.size());
        }
    };

    /** @brief Builds the fixture's head, or fails the test. */
    std::unique_ptr<DecoupledHead> make_head()
    {
        const std::vector<Shape> levels = level_shapes();
        Result<std::unique_ptr<DecoupledHead>> made =
            DecoupledHead::create(span_of(levels), K_CLASSES, K_HIDDEN, K_YOLOX_NORM);
        EXPECT_TRUE(Test::status_is_ok(made.status()));
        if (!made.status().is_ok())
            return nullptr;

        return std::move(made).value();
    }

    /** @brief Parses @p text as the attributes of one row. */
    Config::Value attributes_of(std::string_view text)
    {
        return require_value(Config::parse_json(text, "<test>"));
    }

    /** @brief Returns the constant the bias named @p name starts from, or NaN when absent. */
    double bias_constant(DecoupledHead& head, std::string_view name)
    {
        double found = std::nan("");
        const auto on_parameter = [&](const std::string& state_name, NN::Parameter& state)
        {
            if (state_name != name)
                return;

            EXPECT_EQ(state.init().kind, NN::ParameterInit::Kind::Constant) << state_name;
            found = state.init().constant;
        };
        const auto on_buffer = [](const std::string&, NN::Buffer&) {};
        const NN::StateVisitor visitor(on_parameter, on_buffer);
        NN::walk_state(head, "", visitor);

        return found;
    }
} // namespace

/** @brief Tests that three levels give three ports each, named by level and role. */
TEST(Small_BlockDecoupledHead, ThreeLevelsPublishNinePortsByLevelAndRole)
{
    const std::unique_ptr<DecoupledHead> head = make_head();
    ASSERT_NE(head, nullptr);

    ASSERT_EQ(head->input_count(), K_LEVELS);
    ASSERT_EQ(head->ports().size(), 3 * K_LEVELS);
    const std::array<std::string_view, 3> roles = {"classes", "boxes", "objectness"};
    const std::array<int64_t, 3> channels = {K_CLASSES, 4, 1};
    for (std::size_t level = 0; level < K_LEVELS; ++level)
    {
        for (std::size_t role = 0; role < roles.size(); ++role)
        {
            const NN::Port& port = head->ports()[3 * level + role];
            EXPECT_EQ(port.name, "level" + std::to_string(level) + "." + std::string(roles[role]));
            EXPECT_EQ(port.shape, (Shape{K_SIDES[level], K_SIDES[level], channels[role]}));
        }
    }
}

/** @brief Tests that the port names are the ones the YOLOX decoding looks up. */
TEST(Small_BlockDecoupledHead, PortNamesAreTheContractsNames)
{
    const std::unique_ptr<DecoupledHead> head = make_head();
    ASSERT_NE(head, nullptr);

    using namespace Vision::Detect::Yolox;
    EXPECT_EQ(head->ports()[0].name, yolox_port_name(0, K_YOLOX_CLASSES));
    EXPECT_EQ(head->ports()[1].name, yolox_port_name(0, K_YOLOX_BOXES));
    EXPECT_EQ(head->ports()[2].name, yolox_port_name(0, K_YOLOX_OBJECTNESS));
    EXPECT_EQ(head->ports()[8].name, yolox_port_name(2, K_YOLOX_OBJECTNESS));
}

/** @brief Tests that a trace appends nine values with each port's shape under the batch. */
TEST(Small_BlockDecoupledHead, ATraceAppendsOneValueOfEachPortsShape)
{
    const std::unique_ptr<DecoupledHead> head = make_head();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_OK(head->forward(Trace::evaluation(traced.builder), traced.inputs(), outputs));

    ASSERT_EQ(outputs.size(), head->ports().size());
    for (std::size_t at = 0; at < outputs.size(); ++at)
    {
        const Shape& port = head->ports()[at].shape;
        const Shape& found = traced.graph.value(outputs[at]).shape;
        EXPECT_EQ(found, (Shape{K_BATCH, port[0], port[1], port[2]})) << "port " << at;
    }
}

/**
 * @brief Tests the first level's outputs by their shapes as the task states them: with 3 classes
 *        and a batch of 2 the 8 by 8 level yields [2, 8, 8, 3], [2, 8, 8, 4] and [2, 8, 8, 1].
 */
TEST(Small_BlockDecoupledHead, TheFirstLevelYieldsClassesBoxesAndObjectness)
{
    const std::unique_ptr<DecoupledHead> head = make_head();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_OK(head->forward(Trace::evaluation(traced.builder), traced.inputs(), outputs));

    EXPECT_EQ(traced.graph.value(outputs[0]).shape, (Shape{2, 8, 8, 3}));
    EXPECT_EQ(traced.graph.value(outputs[1]).shape, (Shape{2, 8, 8, 4}));
    EXPECT_EQ(traced.graph.value(outputs[2]).shape, (Shape{2, 8, 8, 1}));
}

/** @brief Tests that the state carries the names of the YOLOX state dict, in its order. */
TEST(Small_BlockDecoupledHead, StateCarriesTheYoloxRecordNames)
{
    const std::unique_ptr<DecoupledHead> head = make_head();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(head->forward(Trace::evaluation(traced.builder), traced.inputs(), outputs));

    const std::vector<std::string> expected = expected_state("head");

    // Per level: two towers of 2 * 5 records, three predictions of 2 and a stem of 5.
    ASSERT_EQ(expected.size(), K_LEVELS * (2 * 10 + 3 * 2 + 5));
    EXPECT_EQ(walk(*head, "head"), expected);
}

/** @brief Tests that the class and objectness biases start from the prior and the boxes at 0. */
TEST(Small_BlockDecoupledHead, TheBiasesStartFromThePriorAndZero)
{
    const std::unique_ptr<DecoupledHead> head = make_head();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(head->forward(Trace::evaluation(traced.builder), traced.inputs(), outputs));

    const double prior = Vision::Detect::Yolox::yolox_prior_bias();
    for (std::size_t level = 0; level < K_LEVELS; ++level)
    {
        const std::string index = std::to_string(level);
        // A constant initialiser stores the value it is given, so the comparison is exact.
        EXPECT_EQ(bias_constant(*head, "cls_preds." + index + ".bias"), prior);
        EXPECT_EQ(bias_constant(*head, "obj_preds." + index + ".bias"), prior);
        EXPECT_EQ(bias_constant(*head, "reg_preds." + index + ".bias"), 0.0);
    }
}

/** @brief Tests that forward given two values for three levels fails and appends nothing. */
TEST(Small_BlockDecoupledHead, ForwardRefusesAWrongInputCount)
{
    const std::unique_ptr<DecoupledHead> head = make_head();
    ASSERT_NE(head, nullptr);
    FeatureGraph traced;
    std::vector<Graph::ValueID> outputs;

    SA_EXPECT_FAILURE(head->forward(Trace::evaluation(traced.builder),
                                    NN::ValueSpan(traced.features.data(), 2), outputs),
                      StatusCode::InvalidDescription, "input count is 3", "given is 2");
    EXPECT_TRUE(outputs.empty());
}

/** @brief Tests that create over no level fails. */
TEST(Small_BlockDecoupledHead, CreateRefusesNoLevel)
{
    SA_EXPECT_FAILURE(DecoupledHead::create(NN::ShapeSpan(), K_CLASSES, K_HIDDEN, K_YOLOX_NORM),
                      StatusCode::InvalidDescription, "at least one feature level");
}

/** @brief Tests that create refuses a level of rank 2, naming the level and its shape. */
TEST(Small_BlockDecoupledHead, CreateRefusesALevelOfRankTwo)
{
    const std::vector<Shape> levels = {Shape{8, 8, 16}, Shape{4, 32}};

    SA_EXPECT_FAILURE(DecoupledHead::create(span_of(levels), K_CLASSES, K_HIDDEN, K_YOLOX_NORM),
                      StatusCode::InvalidDescription, "level 1", "[4, 32]", "rank-3");
}

/** @brief Tests that create refuses zero classes and a zero hidden width. */
TEST(Small_BlockDecoupledHead, CreateRefusesZeroClassesAndZeroHidden)
{
    const std::vector<Shape> levels = level_shapes();

    SA_EXPECT_FAILURE(DecoupledHead::create(span_of(levels), 0, K_HIDDEN, K_YOLOX_NORM),
                      StatusCode::InvalidDescription, "at least one class", "0");
    SA_EXPECT_FAILURE(DecoupledHead::create(span_of(levels), K_CLASSES, 0, K_YOLOX_NORM),
                      StatusCode::InvalidDescription, "hidden width", "0");
}

/** @brief Tests that the factory builds the head from "classes" and "hidden". */
TEST(Small_BlockDecoupledHead, TheFactoryBuildsTheHead)
{
    const std::vector<Shape> levels = level_shapes();
    const Config::Value attributes = attributes_of(
        R"({"classes": 3, "hidden": 8, "bn_epsilon": 0.001, "bn_momentum": 0.03})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_decoupled_head_node(span_of(levels), attributes,
                                                                     K_CONTEXT));

    EXPECT_EQ(node->input_count(), K_LEVELS);
    ASSERT_EQ(node->ports().size(), 3 * K_LEVELS);
    EXPECT_EQ(node->ports()[0].name, "level0.classes");
    EXPECT_EQ(node->ports()[8].shape, (Shape{2, 2, 1}));
}

/** @brief Tests that the factory refuses a key the row does not take. */
TEST(Small_BlockDecoupledHead, TheFactoryRefusesAnUnknownKey)
{
    const std::vector<Shape> levels = level_shapes();
    const Config::Value attributes = attributes_of(R"({"classes": 3, "hidden": 8, "width": 1})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_decoupled_head_node(span_of(levels), attributes,
                                                               K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 30 'decoupled_head'", "'width'");
}

/** @brief Tests that the factory refuses a row without its class count or its hidden width. */
TEST(Small_BlockDecoupledHead, TheFactoryRefusesAnAbsentKey)
{
    const std::vector<Shape> levels = level_shapes();

    SA_EXPECT_FAILURE(Vision::Blocks::make_decoupled_head_node(
                          span_of(levels), attributes_of(R"({"hidden": 8})"), K_CONTEXT),
                      StatusCode::InvalidDescription, "'classes'", "absent");
    SA_EXPECT_FAILURE(Vision::Blocks::make_decoupled_head_node(
                          span_of(levels), attributes_of(R"({"classes": 3})"), K_CONTEXT),
                      StatusCode::InvalidDescription, "'hidden'", "absent");
}

/** @brief Tests that the factory names its row when the levels do not fit. */
TEST(Small_BlockDecoupledHead, TheFactoryRefusesNoLevel)
{
    SA_EXPECT_FAILURE(Vision::Blocks::make_decoupled_head_node(
                          NN::ShapeSpan(), attributes_of(R"({"classes": 3, "hidden": 8})"),
                          K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 30 'decoupled_head'",
                      "at least one feature level");
}
