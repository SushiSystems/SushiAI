/**************************************************************************/
/* test_focus.cpp                                                         */
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
 * @file test_focus.cpp
 * @brief Tests Focus's phase stack against values worked by hand, its state names and its factory.
 * @author Mustafa Garip
 */

#include <cstddef>
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
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/focus.hpp>
#include "host_state_support.hpp"

using namespace SushiAI;
using NN::NormOptions;
using NN::Trace;
using Vision::Blocks::Focus;

namespace SaTest = SushiAI::Test;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Holds the epsilon and the momentum of the YOLOX norms, 1e-3 and 0.03. */
    constexpr NormOptions K_YOLOX_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 0 'focus'";

    /** @brief Lists every piece of state under @p module as "name [extents]", in walk order. */
    template<typename M>
    std::vector<std::string> walk(M& module, std::string_view prefix)
    {
        std::vector<std::string> records;
        const auto on_parameter = [&records](const std::string& name, NN::Parameter& state)
        {
            records.push_back(name + " " + state.shape().to_string());
        };
        const auto on_buffer = [&records](const std::string& name, NN::Buffer& state)
        {
            records.push_back(name + " " + state.shape().to_string());
        };
        const NN::StateVisitor visitor(on_parameter, on_buffer);
        NN::walk_state(module, prefix, visitor);

        return records;
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

    /**
     * @brief Stacks the phases of @p values, laid out as @p input, on the host.
     * @return The values of the stack, or an empty list after a failed expectation.
     */
    std::vector<double> stack_on_host(const Shape& input, const std::vector<double>& values)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(input, "x");
        Result<Graph::ValueID> stacked = Focus::stack_phases(builder, x);
        EXPECT_TRUE(Test::status_is_ok(stacked.status()));
        if (!stacked.status().is_ok())
            return {};

        Graph::HostEvaluator evaluator(graph);
        SA_EXPECT_OK(evaluator.write(x, values));
        SA_EXPECT_OK(evaluator.run());
        Result<std::vector<double>> result = evaluator.read(stacked.value());
        EXPECT_TRUE(Test::status_is_ok(result.status()));
        if (!result.status().is_ok())
            return {};

        return std::move(result).value();
    }
} // namespace

/**
 * @brief Tests the channel order on a 1x4x4x1 input holding 1..16 row by row. Each 2x2 window is
 *        read as top-left, bottom-left, top-right, bottom-right, so the window of rows 0 and 1 and
 *        columns 0 and 1 (1 2 / 5 6) is 1, 5, 2, 6.
 */
TEST(Small_BlockFocus, TheStackFollowsTheYoloxPhaseOrder)
{
    const std::vector<double> result = stack_on_host(Shape{1, 4, 4, 1}, SaTest::ramp(16));

    const std::vector<double> expected = {1, 5, 2, 6, 3, 7, 4, 8, 9, 13, 10, 14, 11, 15, 12, 16};
    EXPECT_EQ(result, expected);
}

/**
 * @brief Tests that a phase keeps its channels together: a 1x2x2x2 input of 1..8 has the pixels
 *        (1, 2), (3, 4) over (5, 6), (7, 8), and stacks as top-left, bottom-left, top-right,
 *        bottom-right.
 */
TEST(Small_BlockFocus, APhaseKeepsItsChannelsTogether)
{
    const std::vector<double> result = stack_on_host(Shape{1, 2, 2, 2}, SaTest::ramp(8));

    const std::vector<double> expected = {1, 2, 5, 6, 3, 4, 7, 8};
    EXPECT_EQ(result, expected);
}

/** @brief Tests that each image of a batch stacks on its own: the second image holds 5..8. */
TEST(Small_BlockFocus, EachImageOfABatchStacksOnItsOwn)
{
    const std::vector<double> result = stack_on_host(Shape{2, 2, 2, 1}, SaTest::ramp(8));

    const std::vector<double> expected = {1, 3, 2, 4, 5, 7, 6, 8};
    EXPECT_EQ(result, expected);
}

/** @brief Tests that the stack halves H and W and multiplies the channels by four. */
TEST(Small_BlockFocus, TheStackHalvesTheExtentsAndQuadruplesTheChannels)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 6, 8, 3}, "x");

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID stacked, Focus::stack_phases(builder, x));

    EXPECT_EQ(graph.value(stacked).shape, (Shape{2, 3, 4, 12}));
}

/** @brief Tests that an odd height or width is refused, naming the shape. */
TEST(Small_BlockFocus, TheStackRefusesAnOddExtent)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID odd_height = builder.input(Shape{1, 3, 4, 1}, "odd_height");
    const Graph::ValueID odd_width = builder.input(Shape{1, 4, 3, 1}, "odd_width");

    SA_EXPECT_FAILURE(Focus::stack_phases(builder, odd_height), StatusCode::InvalidDescription,
                      "even H and W", "[1, 3, 4, 1]");
    SA_EXPECT_FAILURE(Focus::stack_phases(builder, odd_width), StatusCode::InvalidDescription,
                      "even H and W", "[1, 4, 3, 1]");
}

/** @brief Tests that the YOLOX-X stem maps a 640x640 image of 3 channels to 320x320x80. */
TEST(Small_BlockFocus, TheOutputShapeOfTheYoloxXStem)
{
    const Focus stem(3, 80, 3, K_YOLOX_NORM);

    EXPECT_EQ(stem.output_shape(Shape{640, 640, 3}).value(), (Shape{320, 320, 80}));
}

/** @brief Tests that output_shape refuses an odd extent. */
TEST(Small_BlockFocus, TheOutputShapeRefusesAnOddExtent)
{
    const Focus stem(3, 80, 3, K_YOLOX_NORM);

    SA_EXPECT_FAILURE(stem.output_shape(Shape{639, 640, 3}), StatusCode::InvalidDescription,
                      "even H and W", "[639, 640, 3]");
}

/** @brief Tests that a traced stem yields the batch, half the extents and the output width. */
TEST(Small_BlockFocus, ATracedStemYieldsTheOutputWidthAtHalfTheExtents)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 8, 8, 3}, "x");
    Focus stem(3, 16, 3, K_YOLOX_NORM);

    const Graph::ValueID y = stem.forward(Trace::evaluation(builder), x);

    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4, 4, 16}));
}

/** @brief Tests that a stem over an odd input records the failure in the trace. */
TEST(Small_BlockFocus, ATracedStemOverAnOddInputFailsTheTrace)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 5, 4, 3}, "x");
    Focus stem(3, 16, 3, K_YOLOX_NORM);
    const Trace trace = Trace::evaluation(builder);

    static_cast<void>(stem.forward(trace, x));

    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, "even H and W");
}

/**
 * @brief Tests the state names of the YOLOX stem: the convolution reads the 12 stacked channels
 *        and the record names are "conv.conv.weight" and "conv.bn.*", as the checkpoint's
 *        "stem.conv" holds them.
 */
TEST(Small_BlockFocus, StateHoldsTheConvolutionOverTwelveChannels)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 4, 4, 3}, "x");
    Focus stem(3, 80, 3, K_YOLOX_NORM);
    static_cast<void>(stem.forward(Trace::evaluation(builder), x));

    const std::vector<std::string> expected = {
        "stem.conv.conv.weight [80, 3, 3, 12]",
        "stem.conv.bn.weight [80]",
        "stem.conv.bn.bias [80]",
        "stem.conv.bn.running_mean [80]",
        "stem.conv.bn.running_var [80]",
    };
    EXPECT_EQ(walk(stem, "stem"), expected);
}

/** @brief Tests that make_focus_node builds the stem from "filters" and a default kernel of 3. */
TEST(Small_BlockFocus, TheFactoryBuildsTheStemWithAKernelOfThree)
{
    const Shape input{8, 8, 3};
    const Config::Value attributes = attributes_of(R"({"filters": 16})");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_focus_node(one_input(input), attributes,
                                                            K_CONTEXT));

    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{4, 4, 16}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 8, 8, 3}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(node->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));
    const std::vector<std::string> state = walk(*node, "");
    ASSERT_FALSE(state.empty());
    EXPECT_EQ(state[0], "conv.conv.weight [16, 3, 3, 12]");
}

/** @brief Tests that the factory refuses an even kernel, which would change the halved extents. */
TEST(Small_BlockFocus, TheFactoryRefusesAnEvenKernel)
{
    const Shape input{8, 8, 3};
    const Config::Value attributes = attributes_of(R"({"filters": 16, "kernel": 2})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_focus_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 0 'focus'", "'kernel'", "odd", "2");
}

/** @brief Tests that the factory refuses an input whose height is odd. */
TEST(Small_BlockFocus, TheFactoryRefusesAnOddInput)
{
    const Shape input{7, 8, 3};
    const Config::Value attributes = attributes_of(R"({"filters": 16})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_focus_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "even H and W", "[7, 8, 3]");
}

/** @brief Tests that the factory refuses a key the row does not take. */
TEST(Small_BlockFocus, TheFactoryRefusesAnUnknownKey)
{
    const Shape input{8, 8, 3};
    const Config::Value attributes = attributes_of(R"({"filters": 16, "stride": 2})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_focus_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "layer 0 'focus'", "'stride'");
}

/** @brief Tests that the factory refuses a row without its output width. */
TEST(Small_BlockFocus, TheFactoryRefusesAnAbsentFilterCount)
{
    const Shape input{8, 8, 3};
    const Config::Value attributes = attributes_of(R"({"kernel": 3})");

    SA_EXPECT_FAILURE(Vision::Blocks::make_focus_node(one_input(input), attributes, K_CONTEXT),
                      StatusCode::InvalidDescription, "'filters'", "absent");
}
