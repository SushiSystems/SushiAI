/**************************************************************************/
/* test_segment_head.cpp                                                  */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/


/**
 * @file test_segment_head.cpp
 * @brief Tests SegmentHead's trace, its shape, its state names and its factory.
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
#include <SushiAI/vision/blocks/segment_head.hpp>

using namespace SushiAI;
using NN::BufferUpdateList;
using NN::NormOptions;
using NN::Trace;
using Vision::Blocks::SegmentHead;

namespace
{
    /** @brief Holds the epsilon and the momentum the tests give every normalisation. */
    constexpr NormOptions K_NORM{1e-3, 0.03};

    /** @brief Names the row a factory is asked to build, as a diagnostic would. */
    constexpr std::string_view K_CONTEXT = "layer 3 'segment_head'";

    /** @brief Lists the name and the shape of every piece of state under @p module. */
    template<typename M>
    std::vector<std::string> walk(M& module)
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
        NN::walk_state(module, "", visitor);

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

    /** @brief Views one shape as the inputs of a one-input row. */
    NN::ShapeSpan one_input(const Shape& shape)
    {
        return NN::ShapeSpan(&shape, static_cast<std::size_t>(1));
    }
} // namespace

/** @brief Tests that the head multiplies H and W by the scale and ends at the class count. */
TEST(Small_BlockSegmentHead, OutputShapeScalesTheExtentsAndSetsTheClasses)
{
    const SegmentHead head(16, 8, 5, 4, K_NORM);
    const SegmentHead unscaled(16, 8, 5, 1, K_NORM);

    EXPECT_EQ(head.output_shape(Shape{6, 10, 16}).value(), (Shape{24, 40, 5}));
    EXPECT_EQ(unscaled.output_shape(Shape{6, 10, 16}).value(), (Shape{6, 10, 5}));
}

/** @brief Tests that the trace is two convolutions, one norm, one SiLU and one upsample. */
TEST(Small_BlockSegmentHead, TheTraceEndsAtTheUpsample)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 16}, "x");
    SegmentHead head(16, 8, 5, 2, K_NORM);
    BufferUpdateList list;

    const Graph::ValueID y = head.forward(Trace::training(builder, list), x);

    EXPECT_EQ(count_nodes(graph, Ops::K_BATCHNORM2D.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_SILU.id), 1u);
    EXPECT_EQ(count_nodes(graph, Ops::K_UPSAMPLE2D.id), 1u);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 8, 8, 5}));
    EXPECT_EQ(graph.node(graph.value(y).producer).op, Ops::K_UPSAMPLE2D.id);
}

/** @brief Tests the seven pieces of state: cv1's five, then the weight and the bias of cv2. */
TEST(Small_BlockSegmentHead, StateIsListedUnderCv1AndCv2)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 16}, "x");
    SegmentHead head(16, 8, 5, 2, K_NORM);
    BufferUpdateList list;
    static_cast<void>(head.forward(Trace::training(builder, list), x));

    const std::vector<std::string> expected = {
        "cv1.conv.weight [8, 3, 3, 16]", "cv1.bn.weight [8]",       "cv1.bn.bias [8]",
        "cv1.bn.running_mean [8]",       "cv1.bn.running_var [8]", "cv2.weight [5, 1, 1, 8]",
        "cv2.bias [5]",
    };

    EXPECT_EQ(walk(head), expected);
}

/** @brief Tests that the factory reads the row and that "hidden" defaults to the input width. */
TEST(Small_BlockSegmentHead, TheFactoryBuildsTheRow)
{
    const Shape input{4, 4, 16};
    const Config::Value stated = Config::parse_json(
        R"({"classes": 5, "hidden": 8, "scale": 2})", "<test>");
    const Config::Value bare = Config::parse_json(R"({"classes": 5})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            Vision::Blocks::make_segment_head_node(one_input(input), stated,
                                                                   K_CONTEXT));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> plain,
                            Vision::Blocks::make_segment_head_node(one_input(input), bare,
                                                                   K_CONTEXT));

    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{8, 8, 5}));
    ASSERT_EQ(plain->ports().size(), 1u);
    EXPECT_EQ(plain->ports()[0].shape, (Shape{4, 4, 5}));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 16}, "x");
    std::vector<Graph::ValueID> outputs;
    SA_EXPECT_OK(plain->forward(Trace::evaluation(builder), NN::ValueSpan(&x, 1), outputs));
    const std::vector<std::string> state = walk(*plain);
    ASSERT_EQ(state.size(), 7u);
    EXPECT_EQ(state[0], "cv1.conv.weight [16, 3, 3, 16]");
}

/** @brief Tests that the factory refuses an unknown key, a missing class count and scale 0. */
TEST(Small_BlockSegmentHead, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{4, 4, 16};
    const Config::Value unknown = Config::parse_json(R"({"classes": 5, "mode": 1})", "<test>");
    const Config::Value no_classes = Config::parse_json(R"({"scale": 2})", "<test>");
    const Config::Value zero_scale =
        Config::parse_json(R"({"classes": 5, "scale": 0})", "<test>");

    SA_EXPECT_FAILURE(
        Vision::Blocks::make_segment_head_node(one_input(input), unknown, K_CONTEXT),
        StatusCode::InvalidDescription, "layer 3 'segment_head'", "'mode'");
    SA_EXPECT_FAILURE(
        Vision::Blocks::make_segment_head_node(one_input(input), no_classes, K_CONTEXT),
        StatusCode::InvalidDescription, "'classes'", "absent");
    SA_EXPECT_FAILURE(
        Vision::Blocks::make_segment_head_node(one_input(input), zero_scale, K_CONTEXT),
        StatusCode::InvalidDescription, "'scale'", "positive");
}
